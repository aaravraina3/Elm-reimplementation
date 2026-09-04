"""Shared fixtures.

The loss and eval tests need a real model interface (forward passes, a
tokenizer, peft's disable_adapter) but not real capability. So they run against
a randomly initialised tiny Qwen2 with the real tokenizer: no download beyond
the tokenizer files, and a full test pass in seconds rather than minutes.
"""

import dataclasses

import pytest
import torch
from peft import get_peft_model
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from elm.config import ELMConfig
from elm.train import lora_config

TOKENIZER_ID = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_ID)


@pytest.fixture(scope="session")
def cfg():
    # Short sequences and few generated tokens: these tests assert behaviour,
    # not quality.
    return dataclasses.replace(
        ELMConfig(), max_len=64, fluency_max_new_tokens=4, top_k=8
    )


@pytest.fixture
def model(tokenizer, cfg):
    """A tiny randomly initialised Qwen2 with a fresh ELM LoRA adapter attached.

    Eight layers, because ELMConfig targets layers 4-7.
    """
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    base = Qwen2ForCausalLM(config)
    base.requires_grad_(False)
    return get_peft_model(base, lora_config(cfg))


@pytest.fixture
def text():
    return "Buffer overflows occur when a program writes past the end of a buffer."


@pytest.fixture
def personas():
    return (
        "Here is a text written by an expert in the field of exploit development:\n",
        "The text is written by a novice, with no knowledge about exploit development:\n",
    )
