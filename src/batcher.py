import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from .config import BenchmarkConfig
from .stats import Stats
from .utils import get_device, get_model_and_tokenizer, get_profiler, get_record_function

logger = logging.getLogger(__name__)


class Batcher:
    """Base class for batch text generation."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.model, self.tokenizer = get_model_and_tokenizer(config.model)
        self.stats = Stats()
        self.dtype = next(self.model.parameters()).dtype
        self._profiler = get_profiler(config.do_profile)
        self._record_function = get_record_function(config.do_profile)

    def __call__(self, texts: List[str]) -> List[str]:
        raise NotImplementedError("Subclasses must implement __call__")


class SynchronousBatcher(Batcher):
    """Processes texts in synchronous batches for generation."""
    
    @torch.no_grad()
    def __call__(self, texts: List[str]) -> List[str]:
        """
        Generate text for all input texts using synchronous batching.
        
        Args:
            texts: List of input texts to generate from
            
        Returns:
            List of generated text strings
        """
        self._initialize_stats()
        batches = self._create_batches(texts)
        
        logger.info(f"Processing {len(batches)} batches synchronously")
        
        with self._profiler as prof:
            all_generated_ids = self._process_all_batches(batches)
            self._synchronize_device()
            self._finalize_stats()
            generated_texts = self._decode_outputs(all_generated_ids)
        
        self._export_profile_if_needed()
        return generated_texts
    
    def _initialize_stats(self) -> None:
        """Reset stats for a new generation run."""
        self.stats.start_time = time.time()
        self.stats.generated_tokens = 0
        self.stats.prefill_tokens = 0
    
    def _create_batches(self, texts: List[str]) -> List[List[str]]:
        """Split texts into batches based on configured batch size."""
        batch_size = self.config.batch_size
        return [
            texts[i:i + batch_size] 
            for i in range(0, len(texts), batch_size)
        ]
    
    def _process_all_batches(self, batches: List[List[str]]) -> List[torch.Tensor]:
        """Process each batch and collect generated token IDs."""
        all_generated_ids = []
        
        for batch_idx, batch in enumerate(tqdm(batches)):
            generated_ids = self._process_single_batch(batch, batch_idx)
            all_generated_ids.append(generated_ids)
        
        return all_generated_ids
    
    def _process_single_batch(self, batch: List[str], batch_idx: int) -> torch.Tensor:
        """Process a single batch through tokenization and generation."""
        # Track timing for each sample
        batch_start_time = time.time()
        self.stats.sample_start_times.extend([batch_start_time] * len(batch))
        
        # Tokenize and prepare inputs
        inputs = self._tokenize_batch(batch)
        prefix_len = inputs["input_ids"].size(1)
        
        # Generate tokens
        with self._record_function(f"batch {batch_idx}"):
            full_outputs = self._generate_for_batch(inputs)
        
        # Extract only the newly generated tokens (excluding prefix)
        generated_ids = full_outputs[:, prefix_len:].cpu()
        
        # Track timing
        batch_end_time = time.time()
        self.stats.sample_end_times.extend([batch_end_time] * len(batch))
        
        return generated_ids
    
    def _tokenize_batch(self, batch: List[str]) -> dict:
        """Tokenize a batch of texts and move to device."""
        inputs = self.tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            max_length=self.config.max_prefix_len,
            truncation="longest_first",
        )
        inputs = inputs.to(get_device())
        
        # Track prefill tokens (tokens processed before generation)
        self.stats.prefill_tokens += inputs["attention_mask"].sum().item()
        
        return inputs
    
    def _generate_for_batch(self, inputs: dict) -> torch.Tensor:
        """Run generation on a batch of tokenized inputs."""
        return self.model.generate(
            **inputs,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=self.config.max_new_tokens,
            use_cache=True,
        )
    
    def _synchronize_device(self) -> None:
        """Synchronize CUDA device if available."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    def _finalize_stats(self) -> None:
        """Record final timing statistics."""
        self.stats.end_time = time.time()
    
    def _decode_outputs(self, all_generated_ids: List[torch.Tensor]) -> List[str]:
        """
        Decode all generated token IDs to text strings.
        
        Also updates stats with the total number of generated tokens.
        """
        generated_texts = []
        
        for generated_ids in all_generated_ids:
            # Count non-padding tokens
            non_pad_tokens = (generated_ids != self.tokenizer.pad_token_id).sum().item()
            self.stats.generated_tokens += non_pad_tokens
            
            # Decode to text
            batch_texts = self.tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )
            generated_texts.extend(batch_texts)
        
        return generated_texts
    
    def _export_profile_if_needed(self) -> None:
        """Export profiling trace if profiling is enabled."""
        if self.config.do_profile:
            self._profiler.export_chrome_trace("trace.json")

class ContinuousBatcher(Batcher):
    """
    Processes texts using continuous batching, where new samples can be added
    as existing samples complete, maximizing GPU utilization.
    """
    
    @dataclass
    class _Batch:
        """Maintains state for an active batch during continuous generation."""
        # Tracking
        text_ids: List[int]  # Original indices of texts in this batch
        texts_decoding: List[str]  # Texts currently being generated
        texts_waiting: List[str]  # Texts waiting to be added to batch
        
        # Model inputs/outputs
        input_ids: Optional[torch.LongTensor] = None
        position_ids: Optional[torch.LongTensor] = None
        attention_mask: Optional[torch.LongTensor] = None
        past_key_values: Optional[DynamicCache] = None
        
        # Generation tracking
        generated_tokens: Optional[List[List[int]]] = None  # Token IDs per sample
        generated_tokens_counter: Optional[torch.LongTensor] = None  # Count per sample

    @torch.no_grad()
    def __call__(self, texts: List[str]) -> List[str]:
        """
        Generate text for all inputs using continuous batching.
        
        Args:
            texts: List of input texts to generate from
            
        Returns:
            List of generated text strings in original order
        """
        self._validate_config()
        self._initialize_stats(len(texts))
        
        results = [None] * len(texts)
        batch = self._create_initial_batch(texts)
        
        with tqdm(total=len(texts), mininterval=1) as pbar:
            with self._profiler as prof:
                self._run_generation_loop(texts, batch, results, pbar)
        
        self._finalize_and_export(prof)
        return results
    
    def _validate_config(self) -> None:
        """Ensure configuration is valid for continuous batching."""
        assert self.config.fraction > 0.0, "Fraction must be positive for continuous batching"
    
    def _initialize_stats(self, num_texts: int) -> None:
        """Initialize statistics tracking for the generation run."""
        self.stats.start_time = time.time()
        self.stats.sample_start_times = [None] * num_texts
        self.stats.sample_end_times = [None] * num_texts
    
    def _create_initial_batch(self, texts: List[str]) -> _Batch:
        """Create the initial batch with the first batch_size texts waiting."""
        batch_size = self.config.batch_size
        return ContinuousBatcher._Batch(
            text_ids=list(range(batch_size)),
            texts_decoding=[],
            texts_waiting=texts[:batch_size],
        )
    
    def _run_generation_loop(
        self, 
        texts: List[str], 
        batch: _Batch, 
        results: List[str],
        pbar: tqdm
    ) -> None:
        """
        Main generation loop: prefill waiting texts, step generation, 
        and swap in new texts as samples complete.
        """
        next_text_idx = self.config.batch_size
        
        while self._should_continue_generation(batch, next_text_idx, len(texts)):
            # Add waiting texts to batch if threshold met
            if self._should_prefill(batch):
                self._prefill_waiting_texts(batch)
            
            # Generate one token for all active samples
            self._generate_one_step(batch)
            
            # Check for completed samples and swap in new ones
            finished_text_ids, finished_texts = self._collect_finished_samples(batch)
            
            if finished_texts:
                self._save_results(results, finished_text_ids, finished_texts)
                next_text_idx = self._add_new_waiting_texts(
                    batch, texts, next_text_idx, len(finished_texts)
                )
                pbar.update(len(finished_texts))
    
    def _should_continue_generation(
        self, batch: _Batch, next_idx: int, total_texts: int
    ) -> bool:
        """Check if there are still texts to process or generate."""
        has_more_inputs = next_idx < total_texts
        has_active_texts = batch.texts_decoding or batch.texts_waiting
        return has_more_inputs or has_active_texts
    
    def _should_prefill(self, batch: _Batch) -> bool:
        """
        Determine if we should add waiting texts to the batch.
        
        Adds texts when:
        - Batch has room (below batch_size)
        - Waiting texts meet the fraction threshold relative to active texts
        """
        has_capacity = len(batch.texts_decoding) < self.config.batch_size
        meets_threshold = (
            len(batch.texts_waiting) >= 
            len(batch.texts_decoding) * self.config.fraction
        )
        return has_capacity and meets_threshold
    
    def _save_results(
        self, 
        results: List[str], 
        text_ids: List[int], 
        finished_texts: List[str]
    ) -> None:
        """Save completed generations to results list at original indices."""
        for text_id, text in zip(text_ids, finished_texts):
            results[text_id] = text
    
    def _add_new_waiting_texts(
        self, 
        batch: _Batch, 
        texts: List[str], 
        current_idx: int, 
        num_to_add: int
    ) -> int:
        """Add new texts to waiting queue and return updated index."""
        end_idx = current_idx + num_to_add
        new_texts = texts[current_idx:end_idx]
        new_ids = list(range(current_idx, end_idx))
        
        batch.texts_waiting.extend(new_texts)
        batch.text_ids.extend(new_ids)
        
        return end_idx
    
    def _prefill_waiting_texts(self, batch: _Batch) -> None:
        """
        Process waiting texts through prefill phase:
        - Tokenize inputs
        - Run forward pass to generate KV cache
        - Initialize or expand batch state
        """
        self._record_prefill_start_times(batch)
        
        # Tokenize waiting texts
        inputs = self._tokenize_waiting_texts(batch.texts_waiting)
        
        # Run prefill forward pass
        with self._record_function(f"Prefill {len(batch.texts_waiting)}"):
            prefill_outputs = self.model(**inputs, use_cache=True)
        
        self.stats.generated_tokens += inputs["attention_mask"].size(0)
        
        # Initialize or expand the batch
        if self._is_first_prefill(batch):
            self._initialize_batch_from_prefill(batch, prefill_outputs, inputs)
        else:
            self._expand_batch_with_prefill(batch, prefill_outputs, inputs)
    
    def _record_prefill_start_times(self, batch: _Batch) -> None:
        """Record start time for all texts being added to the batch."""
        start_time = time.time()
        start_idx = len(batch.texts_decoding)
        end_idx = start_idx + len(batch.texts_waiting)
        
        for i in range(start_idx, end_idx):
            text_id = batch.text_ids[i]
            self.stats.sample_start_times[text_id] = start_time
    
    def _tokenize_waiting_texts(self, texts: List[str]) -> dict:
        """Tokenize waiting texts with padding and truncation."""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            max_length=self.config.max_prefix_len,
            truncation="longest_first",
        ).to(get_device())
        
        self.stats.prefill_tokens += inputs["attention_mask"].sum().item()
        
        # Calculate position IDs from attention mask
        attention_mask = inputs["attention_mask"]
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        inputs["position_ids"] = position_ids
        
        return inputs
    
    def _is_first_prefill(self, batch: _Batch) -> bool:
        """Check if this is the first prefill (batch is empty)."""
        return not batch.texts_decoding
    
    def _initialize_batch_from_prefill(
        self, 
        batch: _Batch, 
        prefill_outputs: CausalLMOutputWithCrossAttentions,
        inputs: Dict[str, torch.Tensor]
    ) -> None:
        """Initialize batch state from first prefill."""
        attention_mask = inputs["attention_mask"]
        
        # Extend attention mask for next token
        with self._record_function("Prefill attention & positions"):
            attention_mask = torch.cat(
                (attention_mask, torch.ones_like(attention_mask[:, 0:1])), 
                dim=1
            )
            position_ids = self._compute_position_ids(attention_mask)
        
        # Get first generated token from logits
        next_token_ids = prefill_outputs.logits[:, -1].argmax(dim=1, keepdim=True)
        
        # Initialize batch state
        batch.texts_decoding = batch.texts_waiting
        batch.texts_waiting = []
        batch.input_ids = next_token_ids
        batch.past_key_values = prefill_outputs.past_key_values
        batch.attention_mask = attention_mask
        batch.position_ids = position_ids[:, -1].unsqueeze(1)
        batch.generated_tokens_counter = torch.ones(
            len(batch.texts_decoding), device=next_token_ids.device
        )
        batch.generated_tokens = next_token_ids.tolist()
    
    def _expand_batch_with_prefill(
        self,
        batch: _Batch,
        prefill_outputs: CausalLMOutputWithCrossAttentions,
        inputs: Dict[str, torch.Tensor]
    ) -> None:
        """Expand existing batch with newly prefilled samples."""
        # Move waiting texts to decoding
        batch.texts_decoding.extend(batch.texts_waiting)
        batch.texts_waiting = []
        
        # Get first generated tokens for new samples
        next_token_ids = prefill_outputs.logits[:, -1].argmax(dim=1, keepdim=True)
        
        # Concatenate input_ids and generated tokens
        batch.input_ids = torch.cat((batch.input_ids, next_token_ids), dim=0)
        batch.generated_tokens.extend(next_token_ids.tolist())
        
        # Initialize token counter for new samples
        new_counters = torch.ones_like(next_token_ids).view(-1)
        batch.generated_tokens_counter = torch.cat(
            (batch.generated_tokens_counter, new_counters), dim=0
        )
        
        # Expand attention mask and position IDs
        self._expand_attention_mask(batch, inputs)
        
        # Expand KV cache with new samples
        self._expand_kv_cache(batch, prefill_outputs)
    
    def _expand_attention_mask(
        self, 
        batch: _Batch, 
        inputs: Dict[str, torch.Tensor]
    ) -> None:
        """Expand attention mask to include new samples with proper padding."""
        batch_seqlen = batch.attention_mask.size(1)
        inputs_seqlen = inputs["input_ids"].size(1)
        
        with self._record_function("Expand attention"):
            # Pad new samples to match existing sequence length
            padding_size = batch_seqlen - inputs_seqlen - 1
            input_padding = torch.zeros(
                (inputs["input_ids"].size(0), padding_size),
                device=get_device(),
            )
            
            # Concatenate: padding + input attention + next token
            new_attention_mask = torch.cat(
                (
                    input_padding,
                    inputs["attention_mask"],
                    torch.ones_like(inputs["attention_mask"][:, 0:1]),
                ),
                dim=1,
            )
        
        # Add to batch
        batch.attention_mask = torch.cat(
            (batch.attention_mask, new_attention_mask), dim=0
        )
        
        # Recompute position IDs
        batch.position_ids = self._compute_position_ids(batch.attention_mask)
        batch.position_ids = batch.position_ids[:, -1].unsqueeze(1)
    
    def _expand_kv_cache(
        self,
        batch: _Batch,
        prefill_outputs: CausalLMOutputWithCrossAttentions
    ) -> None:
        """Expand KV cache to include new samples with proper padding."""
        # Calculate padding needed for new samples
        existing_seqlen = batch.past_key_values.layers[0].keys.size(2)
        new_seqlen = prefill_outputs.past_key_values.layers[0].keys.size(2)
        padding_seqlen = existing_seqlen - new_seqlen
        
        # Create padding template
        sample_kv = prefill_outputs.past_key_values.layers[0].keys
        padding_template = torch.zeros(
            sample_kv.size(0),  # batch
            sample_kv.size(1),  # num_heads
            padding_seqlen,     # seq_len (padding)
            sample_kv.size(3),  # head_dim
            dtype=self.dtype,
            device=get_device(),
        )
        
        with self._record_function("Expand kv cache"):
            for layer_idx in range(self.model.config.num_hidden_layers):
                self._expand_layer_kv_cache(
                    batch, prefill_outputs, layer_idx, padding_template
                )
    
    def _expand_layer_kv_cache(
        self,
        batch: _Batch,
        prefill_outputs: CausalLMOutputWithCrossAttentions,
        layer_idx: int,
        padding_template: torch.Tensor
    ) -> None:
        """Expand KV cache for a single layer."""
        layer_cache = batch.past_key_values.layers[layer_idx]
        new_layer_cache = prefill_outputs.past_key_values.layers[layer_idx]
        
        # Expand keys: [existing | padding | new]
        padded_keys = torch.cat(
            (padding_template, new_layer_cache.keys), dim=2
        )
        layer_cache.keys = torch.cat(
            (layer_cache.keys, padded_keys), dim=0
        )
        
        # Expand values: [existing | padding | new]
        padded_values = torch.cat(
            (padding_template, new_layer_cache.values), dim=2
        )
        layer_cache.values = torch.cat(
            (layer_cache.values, padded_values), dim=0
        )
    
    def _generate_one_step(self, batch: _Batch) -> None:
        """Generate one token for all active samples in the batch."""
        with self._record_function("Step"):
            step_outputs = self.model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                position_ids=batch.position_ids,
                past_key_values=batch.past_key_values,
                use_cache=True,
            )
        
        self.stats.generated_tokens += step_outputs.logits.size(0)
        
        # Get next tokens from logits
        batch.input_ids = step_outputs.logits[:, 0].argmax(dim=1, keepdim=True)
        
        # Update attention mask and position IDs
        batch.attention_mask = torch.cat(
            (batch.attention_mask, torch.ones_like(batch.attention_mask[:, 0:1])),
            dim=1
        )
        batch.position_ids = self._compute_position_ids(batch.attention_mask)
        batch.position_ids = batch.position_ids[:, -1].unsqueeze(1)
        
        # Increment generation counter
        batch.generated_tokens_counter += 1
        
        # Append new tokens to generated sequences
        new_token_ids = batch.input_ids.view(-1).tolist()
        for token_list, token_id in zip(batch.generated_tokens, new_token_ids):
            token_list.append(token_id)
    
    def _compute_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute position IDs from attention mask."""
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        return position_ids
    
    def _collect_finished_samples(self, batch: _Batch) -> tuple[List[int], List[str]]:
        """
        Identify and remove finished samples from the batch.
        
        Returns:
            Tuple of (text_ids, generated_texts) for finished samples
        """
        finished_indices = self._find_finished_indices(batch)
        
        if not finished_indices:
            return [], []
        
        # Remove finished samples from batch tensors
        keep_indices = self._get_keep_indices(batch, finished_indices)
        self._remove_samples_from_batch(batch, keep_indices)
        
        # Extract and decode finished samples
        text_ids, generated_texts = self._extract_finished_texts(
            batch, finished_indices
        )
        
        return text_ids, generated_texts
    
    def _find_finished_indices(self, batch: _Batch) -> List[int]:
        """Find indices of samples that have finished generating."""
        with self._record_function("Finished check"):
            # Check if sample hit EOS or max length
            is_eos = batch.input_ids == self.tokenizer.eos_token_id
            is_max_length = (
                batch.generated_tokens_counter.unsqueeze(1) >= 
                self.config.max_new_tokens
            )
            
            finished_mask = (is_eos | is_max_length).view(-1).long()
            finished_indices = finished_mask.nonzero().view(-1).tolist()
        
        return finished_indices
    
    def _get_keep_indices(
        self, batch: _Batch, finished_indices: List[int]
    ) -> torch.Tensor:
        """Get indices of samples to keep in the batch."""
        mask = torch.ones(batch.input_ids.size(0), dtype=torch.bool)
        mask[finished_indices] = False
        
        device = batch.input_ids.device
        keep_indices = torch.arange(batch.input_ids.size(0), device=device)[mask]
        
        return keep_indices
    
    def _remove_samples_from_batch(
        self, batch: _Batch, keep_indices: torch.Tensor
    ) -> None:
        """Remove finished samples from all batch tensors."""
        with self._record_function("Finished index_select"):
            # Remove from main tensors
            batch.input_ids = batch.input_ids.index_select(0, keep_indices)
            batch.position_ids = batch.position_ids.index_select(0, keep_indices)
            batch.attention_mask = batch.attention_mask.index_select(0, keep_indices)
            batch.generated_tokens_counter = batch.generated_tokens_counter.index_select(
                0, keep_indices
            )
            
            # Remove from KV cache
            for layer_idx in range(self.model.config.num_hidden_layers):
                layer_cache = batch.past_key_values.layers[layer_idx]
                layer_cache.keys = layer_cache.keys.index_select(0, keep_indices)
                layer_cache.values = layer_cache.values.index_select(0, keep_indices)
    
    def _extract_finished_texts(
        self, batch: _Batch, finished_indices: List[int]
    ) -> tuple[List[int], List[str]]:
        """Extract and decode finished samples, updating metadata."""
        text_ids = []
        generated_texts = []
        end_time = time.time()
        
        # Process in reverse order to maintain indices during removal
        for batch_idx in sorted(finished_indices, reverse=True):
            # Record completion time
            original_text_id = batch.text_ids[batch_idx]
            self.stats.sample_end_times[original_text_id] = end_time
            
            # Remove from batch metadata
            batch.texts_decoding.pop(batch_idx)
            text_ids.append(batch.text_ids.pop(batch_idx))
            
            # Decode generated tokens
            token_ids = batch.generated_tokens.pop(batch_idx)
            generated_text = self.tokenizer.decode(
                token_ids, skip_special_tokens=True
            )
            generated_texts.append(generated_text)
        
        return text_ids, generated_texts
    
    def _finalize_and_export(self, prof) -> None:
        """Finalize statistics and export profiling if enabled."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        self.stats.end_time = time.time()
        
        if self.config.do_profile:
            prof.export_chrome_trace("trace.json")
