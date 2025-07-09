import logging
import time
from dataclasses import dataclass
from textwrap import dedent
from typing import List, Optional, Dict

import torch
from pydantic import BaseModel
from tqdm import tqdm
from transformers import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from .config import BenchmarkConfig
from .utils import get_device, get_model_and_tokenizer

logger = logging.getLogger(__name__)


class Stats(BaseModel):
    start_time: float = 0
    end_time: float = 0
    sample_start_times: List[float] = []
    sample_end_times: List[float] = []
    prefill_tokens: int = 0
    generated_tokens: int = 0

    @property
    def run_time(self):
        return self.end_time - self.start_time

    def print(self):
        assert len(self.sample_start_times) == len(self.sample_end_times)
        n = len(self.sample_start_times)
        print(
            dedent(
                f"""
                Run time: {round(self.run_time, 2)}
                Prefill tokens: {self.prefill_tokens} tok, {self.prefill_tokens / self.run_time} tok/s
                Generated tokens: {self.generated_tokens} tok, {self.generated_tokens / self.run_time} tok/s
                Per sample latency from global start: {sum(self.sample_end_times) / n - self.start_time} s
                Per sample latency from sample start: {sum([e - s for e, s in zip(self.sample_end_times, self.sample_start_times)]) / n}
                """
            )
        )


class Batcher:
    def __init__(
        self,
        config: BenchmarkConfig,
    ):
        self.model, self.tokenizer = get_model_and_tokenizer(config.model)
        self.config = config
        self.stats = Stats()
        self.dtype = next(self.model.parameters()).dtype

    def __call__(self, texts: List[str]):
        raise NotImplementedError()


class SynchronousBatcher(Batcher):
    @torch.no_grad()
    def __call__(self, texts: List[str]) -> List[str]:
        self.stats.start_time = time.time()
        self.stats.generated_tokens = 0
        self.stats.prefill_tokens = 0
        batch_size = self.config.batch_size
        num_batches = (len(texts) + batch_size - 1) // batch_size
        logger.info(f"Synchronous batching working on {num_batches} batches")
        generated_ids_global = []
        for batch_idx in tqdm(range(num_batches)):
            batch = texts[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            self.stats.sample_start_times += [time.time()] * len(batch)
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                max_length=self.config.max_prefix_len,
                truncation="longest_first",
            )
            inputs = inputs.to(get_device())
            self.stats.prefill_tokens += inputs["attention_mask"].sum().item()
            prefix_len = inputs["input_ids"].size(1)
            gens = self.model.generate(
                **inputs,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.config.max_new_tokens,
                use_cache=True,
            )
            generated_ids = gens[:, prefix_len:].cpu()
            self.stats.sample_end_times += [time.time()] * len(batch)
            generated_ids_global.append(generated_ids)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.stats.end_time = time.time()
        generated_texts = []
        for generated_ids in generated_ids_global:
            self.stats.generated_tokens += (
                (generated_ids != self.tokenizer.pad_token_id).sum().item()
            )
            generated_texts += self.tokenizer.batch_decode(generated_ids)
        return generated_texts


class ContinuousBatcher(Batcher):
    @dataclass
    class _Batch:
        text_ids: List[int]
        texts_decoding: List[str]
        texts_waiting: List[str]
        input_ids: Optional[torch.LongTensor] = None
        generated_tokens: Optional[List[List[int]]] = None
        position_ids: Optional[torch.LongTensor] = None
        attention_mask: Optional[torch.LongTensor] = None
        past_key_values: Optional[DynamicCache] = None
        generated_tokens_counter: Optional[torch.LongTensor] = None

    @torch.no_grad()
    def __call__(self, texts: List[str]):
        assert self.config.fraction > 0.0
        self.stats.start_time = time.time()
        self.stats.sample_start_times = [None] * len(texts)
        self.stats.sample_end_times = [None] * len(texts)
        results = [None] * len(texts)
        batch_size = self.config.batch_size
        idx = batch_size
        batch = ContinuousBatcher._Batch(
            text_ids=list(range(idx)),
            texts_decoding=[],
            texts_waiting=texts[:batch_size],
        )
        pbar = tqdm(total=len(texts), mininterval=1)
        while idx < len(texts) or batch.texts_decoding or batch.texts_waiting:
            if (
                len(batch.texts_decoding) < batch_size and 
                len(batch.texts_waiting) >= len(batch.texts_decoding) * self.config.fraction
            ):
                self._update_prefill(batch)
            self._step(batch)
            text_ids, finished_samples = self._get_finished_samples(batch)
            if finished_samples:
                for tid, fs in zip(text_ids, finished_samples):
                    results[tid] = fs
                batch.texts_waiting += texts[idx : idx + len(finished_samples)]
                batch.text_ids += list(range(idx, idx + len(finished_samples)))
                idx += len(finished_samples)
                pbar.update(len(finished_samples))

        self.stats.end_time = time.time()
        return results

    def _update_prefill(self, batch: _Batch):
        start_time = time.time()
        for i in range(
            len(batch.texts_decoding),
            len(batch.texts_decoding) + len(batch.texts_waiting),
        ):
            self.stats.sample_start_times[batch.text_ids[i]] = start_time
        inputs = self.tokenizer(
            batch.texts_waiting,
            return_tensors="pt",
            padding="longest",
            # padding="max_length",
            max_length=self.config.max_prefix_len,
            truncation="longest_first",
        )
        attention_mask = inputs["attention_mask"]
        self.stats.prefill_tokens += inputs["attention_mask"].sum().item()
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        inputs["position_ids"] = position_ids
        prefill_data: CausalLMOutputWithCrossAttentions = self.model(
            **inputs, use_cache=True
        )
        self.stats.generated_tokens += attention_mask.size(0)

        if not batch.texts_decoding:
            attention_mask = torch.cat(
                (attention_mask, torch.ones_like(attention_mask[:, 0:1])), dim=1
            )
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

            batch.texts_decoding = batch.texts_waiting
            batch.texts_waiting = []
            input_ids = prefill_data.logits[:, -1].argmax(dim=1, keepdim=True)
            batch.input_ids = input_ids
            batch.past_key_values = prefill_data.past_key_values
            batch.attention_mask = attention_mask
            batch.position_ids = position_ids[:, -1].unsqueeze(1)
            batch.generated_tokens_counter = torch.ones(len(batch.texts_decoding)).to(
                input_ids.device
            )
            batch.generated_tokens = input_ids.tolist()
        else:
            self._expand_batch(batch, prefill_data, inputs)

    def _expand_batch(
        self, batch: _Batch, prefill_data: CausalLMOutputWithCrossAttentions, inputs: Dict[str, torch.Tensor]
    ):
        batch.texts_decoding += batch.texts_waiting
        batch.texts_waiting = []

        seqlen_to_add = batch.past_key_values.key_cache[0].size(2) - prefill_data.past_key_values.key_cache[0].size(2)
        input_ids = prefill_data.logits[:, -1].argmax(dim=1, keepdim=True)
        batch.input_ids = torch.cat((batch.input_ids, input_ids), dim=0)
        batch.generated_tokens += input_ids.tolist()
        generated_tokens_counter = torch.ones_like(input_ids).view(-1)
        batch.generated_tokens_counter = torch.cat(
            (batch.generated_tokens_counter, generated_tokens_counter), dim=0
        )

        batch_seqlen = batch.attention_mask.size(1)
        inputs_seqlen = inputs["input_ids"].size(1)
        input_paddings = torch.zeros((inputs["input_ids"].size(0), batch_seqlen - inputs_seqlen - 1), device=get_device())
        input_attn_mask = torch.cat((input_paddings, inputs["attention_mask"], torch.ones_like(inputs["attention_mask"][:, 0:1])), dim=1)
        batch.attention_mask = torch.cat((batch.attention_mask, input_attn_mask), dim=0)
        attention_mask = batch.attention_mask
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        batch.position_ids = position_ids[:, -1].unsqueeze(1)

        # batch, n_heads, seq_len, d_head
        pkv = prefill_data.past_key_values.key_cache[0]
        padding_tensor = torch.zeros(
            pkv.size(0), pkv.size(1), seqlen_to_add, pkv.size(3), dtype=self.dtype, device=get_device()
        )

        for layer in range(self.model.config.num_hidden_layers):
            # key
            padded_tensor = torch.cat(
                (padding_tensor, prefill_data.past_key_values.key_cache[layer]),
                dim=2
            )
            batch.past_key_values.key_cache[layer] = torch.cat(
                (batch.past_key_values.key_cache[layer], padded_tensor),
                dim=0
            )

            # value
            padded_tensor = torch.cat(
                (padding_tensor, prefill_data.past_key_values.value_cache[layer]),
                dim=2
            )
            batch.past_key_values.value_cache[layer] = torch.cat(
                (batch.past_key_values.value_cache[layer], padded_tensor),
                dim=0
            )

    def _step(self, batch: _Batch):
        # iids = batch.input_ids.clone()
        step_data: CausalLMOutputWithCrossAttentions = self.model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            position_ids=batch.position_ids,
            past_key_values=batch.past_key_values,
            use_cache=True,
        )
        self.stats.generated_tokens += step_data.logits.size(0)
        batch.input_ids = step_data.logits[:, 0].argmax(dim=1, keepdim=True)
        # if iids.view(-1).tolist() == [10601, 311, 1249]:
        #     input_ids = torch.cat((self.tokenizer(batch.texts_decoding[-1], return_tensors="pt").input_ids, torch.LongTensor([[1249]])), 1)
        #     attention_mask = batch.attention_mask[1:2, -32:]
        #     for layer in range(self.model.config.num_hidden_layers):
        #         batch.past_key_values.key_cache[layer] = batch.past_key_values.key_cache[layer][2:3, :, -32:-1]
        #         batch.past_key_values.value_cache[layer] = batch.past_key_values.value_cache[layer][2:3, :, -32:-1]
        #     am = self.model(input_ids=input_ids, attention_mask=attention_mask).logits.argmax(-1)
        #     print(am)
        #     print()

        attention_mask = torch.cat(
            (batch.attention_mask, torch.ones_like(batch.attention_mask[:, 0:1])), dim=1
        )
        batch.attention_mask = attention_mask
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        batch.position_ids = position_ids[:, -1].unsqueeze(1)
        batch.generated_tokens_counter += 1

        input_ids = batch.input_ids.view(-1).tolist()
        for generated_tokens, token_id in zip(batch.generated_tokens, input_ids):
            generated_tokens.append(token_id)

    def _get_finished_samples(self, batch: _Batch):
        finished = (
            (
                (batch.input_ids == self.tokenizer.eos_token_id)
                | (
                    batch.generated_tokens_counter.unsqueeze(1)
                    >= self.config.max_new_tokens
                )
            )
            .view(-1)
            .long()
        )
        finished_indices = finished.nonzero().view(-1).tolist()
        if not finished_indices:
            return [], []
        mask = torch.ones_like(batch.input_ids, dtype=bool).view(-1)
        mask[finished_indices] = False
        include_indices = torch.arange(batch.input_ids.size(0)).to(mask.device)[mask]

        batch.input_ids = batch.input_ids.index_select(0, include_indices)
        batch.position_ids = batch.position_ids.index_select(0, include_indices)
        batch.attention_mask = batch.attention_mask.index_select(0, include_indices)
        for layer in range(self.model.config.num_hidden_layers):
            batch.past_key_values.key_cache[layer] = batch.past_key_values.key_cache[layer].index_select(0, include_indices)
            batch.past_key_values.value_cache[layer] = batch.past_key_values.value_cache[layer].index_select(0, include_indices)

        batch.generated_tokens_counter = batch.generated_tokens_counter.index_select(
            0, include_indices
        )
        # parse text results
        result = []
        text_ids = []
        end_time = time.time()
        for remove_index in sorted(finished_indices, reverse=True):
            self.stats.sample_end_times[batch.text_ids[remove_index]] = end_time
            batch.texts_decoding.pop(remove_index)
            text_ids.append(batch.text_ids.pop(remove_index))
            suffix = self.tokenizer.decode(batch.generated_tokens.pop(remove_index))
            result.append(suffix)

        return text_ids, result
