"""Tests for the evaluation code.

A randomly initialised model cannot be asserted to score well on anything, so
these check the things that must hold regardless of the model: prompt shape,
answer-format handling, bounds, and that perplexity is exactly exp(mean NLL).
"""

import math

import pytest
import torch

from elm.evaluate import LETTERS, format_mcq, mcq_accuracy, measure, perplexity

QUESTIONS = [
    {"question": "Which call allocates heap memory?",
     "choices": ["malloc", "printf", "chmod", "fork"], "answer": 0},
    {"question": "What does ASLR randomise?",
     "choices": ["file names", "address layout", "packet order", "clock skew"], "answer": 1},
    {"question": "Which tool disassembles binaries?",
     "choices": ["curl", "sed", "objdump", "tar"], "answer": 2},
]

TEXTS = [
    "Buffer overflows occur when a program writes past the end of a buffer.",
    "Static analysis inspects code without executing it.",
]


# --- prompt formatting -------------------------------------------------------

def test_format_mcq_lists_every_choice_and_ends_with_the_answer_cue():
    prompt = format_mcq(QUESTIONS[0]["question"], QUESTIONS[0]["choices"])
    for letter, choice in zip(LETTERS, QUESTIONS[0]["choices"]):
        assert f"{letter}. {choice}" in prompt
    assert prompt.rstrip().endswith("Answer:")
    assert QUESTIONS[0]["question"] in prompt


# --- accuracy ----------------------------------------------------------------

def test_mcq_accuracy_is_a_fraction(model, tokenizer):
    acc = mcq_accuracy(model, tokenizer, QUESTIONS)
    assert 0.0 <= acc <= 1.0
    assert acc * len(QUESTIONS) == pytest.approx(round(acc * len(QUESTIONS)))


def test_mcq_accuracy_treats_letter_and_index_answers_identically(model, tokenizer):
    """The WMDP files use integer answers; other sources use letters. Both must
    score the same, otherwise a format change silently shifts the metric."""
    as_letters = [{**q, "answer": LETTERS[q["answer"]]} for q in QUESTIONS]
    assert mcq_accuracy(model, tokenizer, QUESTIONS) == mcq_accuracy(
        model, tokenizer, as_letters
    )


def test_mcq_accuracy_respects_the_limit(model, tokenizer):
    acc = mcq_accuracy(model, tokenizer, QUESTIONS, limit=1)
    assert acc in (0.0, 1.0)


def test_mcq_accuracy_on_an_empty_set_does_not_divide_by_zero(model, tokenizer):
    assert mcq_accuracy(model, tokenizer, [], limit=None) == 0.0


# --- perplexity --------------------------------------------------------------

def test_perplexity_is_finite_and_above_one(model, tokenizer):
    ppl = perplexity(model, tokenizer, TEXTS, max_len=64)
    assert math.isfinite(ppl) and ppl > 1.0


def test_perplexity_equals_exp_of_pooled_nll(model, tokenizer):
    """Definitional check: perplexity must be exp(total NLL / total tokens),
    pooled across passages rather than averaged per-passage."""
    import torch.nn.functional as F

    total_nll, total_tokens = 0.0, 0
    for text in TEXTS:
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            logits = model(**ids).logits
        total_nll += F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            ids.input_ids[:, 1:].reshape(-1),
            reduction="sum",
        ).item()
        total_tokens += ids.input_ids.shape[1] - 1

    expected = math.exp(total_nll / total_tokens)
    assert perplexity(model, tokenizer, TEXTS, max_len=64) == pytest.approx(expected, rel=1e-4)


def test_perplexity_skips_passages_too_short_to_score(model, tokenizer):
    """A single-token passage has no next-token to predict; it must be skipped
    rather than contributing a zero-token division."""
    ppl = perplexity(model, tokenizer, ["a"] + TEXTS, max_len=64)
    assert math.isfinite(ppl) and ppl > 1.0


# --- the bundle --------------------------------------------------------------

def test_measure_returns_all_three_criteria(model, tokenizer):
    result = measure(model, tokenizer, QUESTIONS, TEXTS, TEXTS, mcq_limit=2, ppl_limit=2)
    assert set(result) == {"wmdp_cyber_acc", "retain_ppl", "forget_ppl"}
    assert all(math.isfinite(v) for v in result.values())
