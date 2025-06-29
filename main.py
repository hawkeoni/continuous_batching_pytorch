import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
BATCH_SIZE = 3
MAX_NEW_TOKENS = 32
MAX_PREFIX_LENGTH = 512


def _move_to_device(inputs):
    for k, v in inputs.items():
        inputs[k] = v.to(device)
    return inputs


def get_model_and_tokenizer(padding_side="left"):
    logger.info("Start loading the model & tokenizer")
    model_name = "Qwen/Qwen3-1.7B"
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )
        .eval()
        .to(device)
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side=padding_side)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    logger.info("Finished loading the model & tokenizer")
    return model, tokenizer


def get_dataset() -> List[str]:
    file = Path("shakespeare.txt")
    if file.exists():
        logger.info("Using cached text")
        text = file.read_text()
    else:
        logger.info("Downloading text")
        text = requests.get(
            "https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt"
        ).content.decode()
        file.write_text(text)
    random.seed(4)
    samples = []
    start = 0
    for _ in range(100):
        l = random.randint(5, 250)
        samples.append(text[start : start + l])
        start += l
    return samples


@torch.no_grad()
def synchronous_batching():
    model, tokenizer = get_model_and_tokenizer()
    dataset = get_dataset()
    logger.info("Start synchronous batching")
    results = []
    start_time = time.time()
    prefill_tokens = 0
    generated_tokens = 0
    num_batches = (len(dataset) + BATCH_SIZE - 1) // BATCH_SIZE
    latencies = []
    logger.info(f"Synchronous batching working on {num_batches} batches")
    for batch_idx in range(num_batches):
        lstart = time.time()
        batch = dataset[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors="pt", padding=True)
        prefill_tokens += inputs["attention_mask"].sum().item()
        prefix_len = inputs["input_ids"].size(1)
        inputs = _move_to_device(inputs)
        gens = model.generate(
            **inputs,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
        )
        generated_ids = gens[:, prefix_len:]
        generated_tokens += (generated_ids != tokenizer.pad_token_id).sum().item()
        generated_texts = tokenizer.batch_decode(generated_ids.cpu())
        latencies += [time.time() - lstart] * BATCH_SIZE
        for prefix, generation in zip(batch, generated_texts):
            results.append({"prefix": prefix, "generation": generation})
    run_time = time.time() - start_time
    logger.info(f"RUN TIME: {run_time}")
    logger.info(f"Prefill Tokens per second: {prefill_tokens / run_time}")
    logger.info(f"Generated Tokens per second: {generated_tokens / run_time}")
    logger.info(f"Average Latency: {sum(latencies) / len(latencies)}")
    return results


BAD = "es with thee.\n\n\n                     4\n  Unthrifty loveliness why dost thou spend,\n  Upon thy self thy beauty's legacy?\n  Nature's beques"


class ContinousBatcher:

    @dataclass
    class _Batch:
        texts_decoding: List[str]
        texts_waiting: List[str]
        input_ids: Optional[torch.LongTensor] = None
        generated_tokens: Optional[List[List[int]]] = None
        position_ids: Optional[torch.LongTensor] = None
        attention_mask: Optional[torch.LongTensor] = None
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        generated_tokens_counter: Optional[torch.LongTensor] = None
        start_times: Optional[List[float]] = None

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self._prefill_tokens = 0
        self._generated_tokens = 0

    def _update_prefill(self, batch: _Batch):
        inputs = self.tokenizer(
            batch.texts_waiting,
            return_tensors="pt",
            padding="max_length",
            max_length=MAX_PREFIX_LENGTH,
        )
        inputs = _move_to_device(inputs)
        attention_mask = inputs["attention_mask"]
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        inputs["position_ids"] = position_ids
        prefill_data: CausalLMOutputWithCrossAttentions = self.model(**inputs, use_cache=True)

        self._prefill_tokens += attention_mask.sum().item()

        if not batch.texts_decoding:
            attention_mask = torch.cat((attention_mask, torch.ones_like(attention_mask[:, 0:1])), dim=1)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

            batch.texts_decoding = batch.texts_waiting
            batch.texts_waiting = []
            batch.start_times = [time.time()] * len(batch.texts_decoding)
            input_ids = prefill_data.logits[:, -1].argmax(dim=1, keepdim=True)
            batch.input_ids = input_ids
            batch.past_key_values = prefill_data.past_key_values
            batch.attention_mask = attention_mask
            batch.position_ids = position_ids[:, -1].unsqueeze(1)
            batch.generated_tokens_counter = torch.ones(len(batch.texts_decoding)).to(input_ids.device)
            batch.generated_tokens = input_ids.tolist()
        else:
            self._expand_batch(batch, prefill_data)

    def _expand_batch(self, batch: _Batch, prefill_data: CausalLMOutputWithCrossAttentions):
        batch.texts_decoding += batch.texts_waiting
        batch.texts_waiting = []

        input_ids = prefill_data.logits[:, -1].argmax(dim=1, keepdim=True)
        batch.input_ids = torch.cat((batch.input_ids, input_ids), dim=0)
        batch.generated_tokens += input_ids.tolist()
        generated_tokens_counter = torch.ones_like(input_ids).view(-1)
        batch.generated_tokens_counter = torch.cat((batch.generated_tokens_counter, generated_tokens_counter), dim=0)

        attention_mask = batch.attention_mask
        attention_mask = torch.cat((attention_mask, torch.ones_like(attention_mask[:, 0:1])), dim=1)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        batch.position_ids = position_ids[:, -1].unsqueeze(1)
        batch.attention_mask = attention_mask

        new_past_key_values = []
        # batch, n_heads, seq_len, d_head
        seqlen_to_add = batch.past_key_values[0][0].size(2) - prefill_data.past_key_values.size(2)
        pkv = prefill_data.past_key_values
        padding_tensor = torch.zeros(pkv.size(0), pkv.size(1), seqlen_to_add, pkv.size(3))
        for layer in range(len(batch.past_key_values)):
            layer_kv = []
            for key_value_idx in range(2):
                padded_tensor = torch.cat((padding_tensor, prefill_data.past_key_values), dim=2)
                new_kv = torch.cat((batch.past_key_values[layer][key_value_idx], padded_tensor))
                layer_kv.append(new_kv)
            new_past_key_values.append(tuple(layer_kv))

        batch.past_key_values = tuple(new_past_key_values)

    def _step(self, batch: _Batch):
        step_data: CausalLMOutputWithCrossAttentions = self.model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            position_ids=batch.position_ids,
            past_key_values=batch.past_key_values,
            use_cache=True,
        )
        batch.input_ids = step_data.logits[:, 0].argmax(dim=1, keepdim=True)
        batch.past_key_values = step_data.past_key_values

        attention_mask = torch.cat((batch.attention_mask, torch.ones_like(batch.attention_mask[:, 0:1])), dim=1)
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
                | (batch.generated_tokens_counter.unsqueeze(1) >= MAX_NEW_TOKENS)
            )
            .view(-1)
            .long()
        )
        finished_indices = finished.nonzero().view(-1).tolist()
        if not finished_indices:
            return []
        mask = torch.ones_like(batch.input_ids, dtype=bool).view(-1)
        mask[finished_indices] = False
        include_indices = torch.arange(batch.input_ids.size(0)).to(mask.device)[mask]

        batch.input_ids = batch.input_ids.index_select(0, include_indices)
        batch.position_ids = batch.position_ids.index_select(0, include_indices)
        batch.attention_mask = batch.attention_mask.index_select(0, include_indices)
        new_past_key_values = []
        for layer in range(len(batch.past_key_values)):
            layer_kv = []
            for key_value_idx in range(2):
                layer_kv.append(batch.past_key_values[layer][key_value_idx].index_select(0, include_indices))
            new_past_key_values.append(tuple(layer_kv))
        batch.past_key_values = tuple(new_past_key_values)
        batch.generated_tokens_counter = batch.generated_tokens_counter.index_select(0, include_indices)

        # parse text results
        result = []
        for remove_index in sorted(finished_indices, reverse=True):
            prefix = batch.texts_decoding.pop(remove_index)
            suffix = self.tokenizer.decode(batch.generated_tokens.pop(remove_index))
            result.append({"prefix": prefix, "generation": suffix})

        return result

    def process(self, dataset: List[str]):
        idx = BATCH_SIZE
        prefill_tokens = 0
        generated_tokens = 0
        latencies = []
        results = []
        batch = ContinousBatcher._Batch(texts_decoding=[], texts_waiting=dataset[:BATCH_SIZE])
        while idx < len(dataset) or batch.texts_decoding or batch.texts_waiting:
            if len(batch.texts_waiting) >= BATCH_SIZE // 2 or len(batch.texts_decoding) == 0:
                self._update_prefill(batch)
            self._step(batch)
            self._generated_tokens += len(batch.texts_decoding)
            finished_samples = self._get_finished_samples(batch)
            if finished_samples:
                results += finished_samples
                latencies += [time.time()] * len(finished_samples)
                batch.start_times += [time.time()] * len(finished_samples)
                batch.texts_waiting += dataset[idx : idx + len(finished_samples)]
                idx += len(finished_samples)

        return results, latencies


@torch.no_grad()
def continous_batching():
    model, tokenizer = get_model_and_tokenizer()
    batcher = ContinousBatcher(model, tokenizer)
    dataset = get_dataset()
    start_time = time.time()
    logger.info("Start continous batching")
    results, latencies = batcher.process(dataset)
    run_time = time.time() - start_time
    logger.info(f"RUN TIME: {run_time}")
    logger.info(f"Prefill Tokens per second: {batcher._prefill_tokens / run_time}")
    logger.info(f"Generated Tokens per second: {batcher._generated_tokens / run_time}")
    # logger.info(f"Average Latency: {sum(latencies) / len(latencies)}")
    return results


if __name__ == "__main__":
    sync_results = sorted(synchronous_batching(), key=lambda x: x["prefix"])
    cont_results = sorted(continous_batching(), key=lambda x: x["prefix"])
    dataset = set(get_dataset())

    cont_prefixes = set([sample["prefix"] for sample in cont_results])
    print(dataset.difference(cont_prefixes))

    wrong = 0
    correct = 0
    for s, c in zip(sync_results, cont_results):
        assert s["prefix"] == c["prefix"]
        if s["generation"] != c["generation"]:
            # print(f"""`{s["generation"]}`""")
            # print(f"""`{c["generation"]}`""")
            # print("------")
            wrong += 1
        else:
            correct += 1
            # break
    print("Correct %s Wrong %s" % (correct, wrong))
