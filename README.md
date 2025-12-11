# Continuous Batching in PyTorch & Transformers

> **⚠️ DISCLAIMER**  
> This code relies heavily on the internal structure of `transformers.DynamicCache` and is tested on version 4.57.1.  
> Compatibility with other versions is not guaranteed.

## Overview

This repository implements a simple continuous batching pipeline for Large Language Models (LLMs) using pure PyTorch and Hugging Face Transformers. 

**Purpose:** Educational demonstration of continuous batching concepts.

**For production use:** Consider mature inference frameworks like [vLLM](https://github.com/vllm-project/vllm), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), or [SGLang](https://github.com/sgl-project/sglang).

## Quick Start

Run generation with:

```bash
python main.py -c configs/test_qwen3_8b_cont.json -o cont.json
```

This takes a configuration file, runs text generation, and saves results with timing statistics to the output file.


## Configuration Parameters
Example configuration (configs/test_qwen3_8b_cont.json):


```json
{
    "model": "Qwen/Qwen3-8B",
    "continuous_batching": true,
    "batch_size": 3,
    "max_prefix_len": 512,
    "max_new_tokens": 64,
    "dataset_size": 100,
    "fraction": 0.5,
    "do_profile": false
}

```

| Parameter | Description |
|-----------|-------------|
| `model` | Hugging Face model identifier |
| `continuous_batching` | Enable continuous batching (`true`) or use synchronous batching (`false`) |
| `batch_size` | Maximum number of samples to process simultaneously |
| `max_prefix_len` | Maximum input length; longer texts will be truncated |
| `max_new_tokens` | Maximum number of tokens to generate per sample |
| `dataset_size` | Total number of samples to generate |
| `fraction` | Prefill threshold ratio (see below) |
| `do_profile` | Generate PyTorch profiling trace for performance analysis |

### Understanding the `fraction` Parameter

The `fraction` parameter controls when to start prefilling new samples during continuous batching.

**Formula:** Prefill starts when `waiting_samples > generating_samples × fraction`

**Examples:**
- **Batch size:** 20, **Fraction:** 0.5
  - **Scenario 1:** 1 waiting, 19 generating → **No prefill** (1 < 19 × 0.5 = 9.5)
  - **Scenario 2:** 10 waiting, 10 generating → **Prefill starts** (10 ≥ 10 × 0.5 = 5)

Lower fractions allow prefilling sooner, higher fractions wait for more samples to accumulate.

## Dataset

The benchmark uses a mix of two datasets to highlight continuous batching benefits:

- **[tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)** – Long-form responses
- **[cais/mmlu](https://huggingface.co/datasets/cais/mmlu)** – Short answers

This combination creates varied generation lengths, demonstrating how continuous batching efficiently handles heterogeneous workloads.

## Performance Profiling

When `do_profile: true`, the script generates a trace file that can be visualized at:
- [Perfetto UI](https://ui.perfetto.dev)
- Chrome browser: `chrome://tracing`

## Benchmark Results

Results for **Qwen3-8B** model (100 samples):

    Correctness: 99/100 samples matched between methods

| Metric | Continuous Batching | Synchronous Batching | Improvement |
|--------|--------------------:|---------------------:|------------:|
| **Total Runtime** | 64.6s | 107.8s | **40% faster** |
| **Prefill Tokens** | 5,395 | 5,395 | – |
| **Generated Tokens** | 3,172 | 3,113 | – |
| **Generation Speed** | 49.1 tok/s | 28.9 tok/s | **70% faster** |
| **Global Latency** | 35.2s | 56.9s | **38% lower** |
| **Per-Sample Latency** | 1.94s | 3.19s | **39% lower** |

### Key Takeaways

✅ **1.7× throughput improvement** – Continuous batching generates useful tokens 70% faster  
✅ **40% reduction in per-sample latency** – Users get results faster  
✅ **Efficient GPU utilization** – New samples fill gaps left by completed generations  
✅ **High accuracy** – 99% agreement between batching methods

## How It Works

**Synchronous Batching:**
- Processes fixed batches sequentially
- All samples must complete before starting the next batch
- GPU may be underutilized when samples finish at different times

**Continuous Batching:**
- Dynamically adds new samples as existing ones complete
- Maintains high batch occupancy throughout generation
- Better GPU utilization, especially with varied output lengths

---

**Questions or issues?** Please open an issue on GitHub or contact me at ilyadimov98@gmail.com
