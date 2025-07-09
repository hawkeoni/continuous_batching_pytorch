import random
import logging
from typing import List
from contextlib import nullcontext, contextmanager

import torch
from torch.profiler import profile, ProfilerActivity
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def get_model_and_tokenizer(model_name: str):
    logger.info("Start loading the model & tokenizer")
    cuda_kwargs = {}
    if torch.cuda.is_available():
        cuda_kwargs = {"attn_implementation": "flash_attention_2"}
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            **cuda_kwargs,
        )
        .eval()
        .to(get_device())
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    logger.info("Finished loading the model & tokenizer")
    return model, tokenizer


def get_alpaca_dataset(dataset_size: int, tokenizer: AutoTokenizer) -> List[str]:
    dataset = load_dataset("tatsu-lab/alpaca", split=f"train[:{dataset_size}]")
    texts = []
    for sample in dataset:
        messages = [
            {"role": "user", "content": sample["instruction"] + " " + sample["input"]}
        ]
        if tokenizer.chat_template is None:
            texts.append(sample["instruction"] + " " + sample["input"] + "Answer:\n")
        else:
            texts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
    return texts


def get_mmlu_dataset(dataset_size: int, tokenizer: AutoTokenizer) -> List[str]:
    dataset = load_dataset("cais/mmlu", "all", split=f"test[:{dataset_size}]")
    texts = []
    for sample in dataset:
        prompt = sample["question"] + "\nChoices:" + " ".join(sample["choices"]) + "\nWithout any reasoning give me an answer from one of the choices."
        messages = [
            {"role": "user", "content": prompt}
        ]
        if tokenizer.chat_template is None:
            texts.append(prompt)
        else:
            texts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
    return texts


def get_benchmark_dataset(dataset_size: int, tokenizer: AutoTokenizer) -> List[str]:
    random.seed(0)
    texts = get_alpaca_dataset(dataset_size // 2, tokenizer) + get_mmlu_dataset(dataset_size // 2, tokenizer)
    random.shuffle(texts)
    return texts


_DEVICE = None

def get_device():
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE
    _DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return _DEVICE


def get_profiler(enabled: bool):
    if enabled:
        return profile(
            activities=[ProfilerActivity.CPU]
        )
    else:
        return nullcontext()

def get_record_function(enabled: bool):
    if enabled:
        return torch.profiler.record_function
    else:
        @contextmanager
        def dummy_record_function(_):
            yield
        return dummy_record_function