import logging
from pathlib import Path
from typing import List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def get_model_and_tokenizer(model_name: str):
    logger.info("Start loading the model & tokenizer")
    model_name = "Qwen/Qwen3-1.7B"
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )
        .eval()
        .cuda()
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    logger.info("Finished loading the model & tokenizer")
    return model, tokenizer


def get_dataset(dataset_size: str, tokenizer: AutoTokenizer) -> List[str]:
    dataset = load_dataset("tatsu-lab/alpaca", split=f"train[:{dataset_size}]")
    texts = []
    for sample in dataset:
        messages = [
            {"role": "user", "content": sample["instruction"] + " " + sample["input"]}
        ]
        texts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
    return texts
