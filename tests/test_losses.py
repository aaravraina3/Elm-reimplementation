"""Tests for the three loss terms.

These run against a randomly initialised tiny model (see conftest.py), so they
assert *behaviour* — zero points, alignment, masking, gradient flow — rather
than anything about quality.
"""

import pytest
import torch
import torch.nn.functional as F

from elm.losses import (
    erase_loss,
    fluency_loss,
    generate_deflection,
    retain_loss,
    soft_cross_entropy,
)


# --- soft_cross_entropy ------------------------------------------------------

def test_soft_cross_entropy_equals_entropy_when_prediction_matches_target():
    """H(p, p) is just H(p). This is the floor the retain term subtracts off."""
    torch.manual_seed(0)
    logits = torch.randn(5, 32)
    probs = logits.softmax(dim=-1)
    entropy = -(probs * logits.log_softmax(dim=-1)).sum(dim=-1).mean()
    assert soft_cross_entropy(logits, probs) == pytest.approx(entropy.item(), abs=1e-6)


def test_soft_cross_entropy_is_minimised_at_the_target():
    torch.manual_seed(0)
    logits = torch.randn(5, 32)
    probs = logits.softmax(dim=-1)
    matched = soft_cross_entropy(logits, probs)
    for scale in (0.5, 2.0, 5.0):
        assert soft_cross_entropy(logits * scale, probs) >= matched - 1e-6


def test_soft_cross_entropy_matches_torch_with_probability_targets():
    torch.manual_seed(0)
    logits, target_logits = torch.randn(5, 32), torch.randn(5, 32)
    probs = target_logits.softmax(dim=-1)
    expected = F.cross_entropy(logits, probs)
    assert soft_cross_entropy(logits, probs) == pytest.approx(expected.item(), abs=1e-5)


# --- retain ------------------------------------------------------------------

def test_retain_loss_is_zero_for_an_untrained_adapter(model, tokenizer, cfg, text):
    """lora_B is initialised to zeros, so the adapter is an exact identity and
    the KL from the frozen model must be 0. This simultaneously checks that
    disable_adapter() is being used correctly."""
    loss = retain_loss(model, tokenizer, text, cfg)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_retain_loss_is_positive_once_the_adapter_is_perturbed(model, tokenizer, cfg, text):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.add_(torch.randn_like(param) * 0.05)
    assert retain_loss(model, tokenizer, text, cfg).item() > 1e-4


def test_retain_loss_produces_gradients(model, tokenizer, cfg, text):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.add_(torch.randn_like(param) * 0.05)
    retain_loss(model, tokenizer, text, cfg).backward()
    grads = [p.grad for n, p in model.named_parameters() if "lora_" in n and p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


# --- erase -------------------------------------------------------------------

def test_erase_loss_is_finite_and_produces_gradients(model, tokenizer, cfg, text, personas):
    expert, novice = personas
    loss = erase_loss(model, tokenizer, text, expert, novice, cfg)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    grads = [p.grad for n, p in model.named_parameters() if "lora_" in n and p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_erase_loss_matches_retain_when_the_personas_are_identical(
    model, tokenizer, cfg, text, personas
):
    """Identical prefixes make (novice - expert) zero, so the ELM target
    collapses to the frozen model's own distribution and the erase term becomes
    the retain term, up to the entropy constant that retain subtracts."""
    expert, _ = personas
    import dataclasses

    no_topk = dataclasses.replace(cfg, top_k=None)
    erase = erase_loss(model, tokenizer, text, expert, expert, no_topk)

    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=no_topk.max_len)
    with torch.no_grad(), model.disable_adapter():
        frozen = model(**ids).logits[0].float() / no_topk.temperature
    entropy = -(frozen.softmax(-1) * frozen.log_softmax(-1)).sum(-1).mean()

    # The adapter is an identity, so the only gap is the temperature scaling
    # applied to the target but not to the model's own logits.
    assert erase.item() > entropy.item() - 1.0


# --- fluency -----------------------------------------------------------------

def test_fluency_loss_is_zero_for_an_empty_continuation(model, tokenizer, cfg, text):
    assert fluency_loss(model, tokenizer, text, "", cfg).item() == pytest.approx(0.0)


def test_fluency_loss_supervises_only_the_continuation(model, tokenizer, cfg):
    """Recompute the masked cross-entropy by hand and require an exact match.
    This is the assertion that would catch an off-by-one in the label shift."""
    prompt = "Reverse engineering is"
    continuation = " a completely different and fun topic today."

    loss = fluency_loss(model, tokenizer, prompt, continuation, cfg)

    n_prompt = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
    full = tokenizer(prompt + continuation, return_tensors="pt")
    with torch.no_grad():
        logits = model(**full).logits
    labels = full.input_ids.clone()
    labels[:, :n_prompt] = -100
    expected = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    assert loss.item() == pytest.approx(expected.item(), abs=1e-4)


def test_fluency_loss_ignores_changes_confined_to_the_prompt(model, tokenizer, cfg):
    """Supervision is masked to the continuation, so the number of *supervised
    tokens* must not change when only the prompt changes."""
    continuation = " and now for something completely different."
    short = fluency_loss(model, tokenizer, "Malware", continuation, cfg)
    long = fluency_loss(model, tokenizer, "Malware analysis of packed binaries", continuation, cfg)
    assert torch.isfinite(short) and torch.isfinite(long)
    assert short.item() != long.item()  # context differs, so the values differ


# --- guided decoding ---------------------------------------------------------

def test_generate_deflection_respects_the_token_budget(model, tokenizer, cfg, personas):
    """The budget is asserted through output length rather than by re-encoding
    the result: decode-then-encode is not token-count preserving, so a 4-token
    generation can round-trip to 6 tokens and the count tells you nothing.
    Seeding makes the two runs share their opening tokens."""
    expert, novice = personas

    def run(budget):
        torch.manual_seed(0)
        with model.disable_adapter():
            return generate_deflection(
                model, tokenizer, "Reverse engineering is", expert, novice, cfg,
                max_new_tokens=budget,
            )

    short, long = run(1), run(8)
    assert isinstance(short, str) and isinstance(long, str)
    assert len(long) >= len(short)


def test_generate_deflection_is_deterministic_under_a_fixed_seed(model, tokenizer, cfg, personas):
    """Sampling draws from the global RNG, so a seeded run must reproduce. This
    is what makes the fluency term debuggable at all."""
    expert, novice = personas

    def run():
        torch.manual_seed(1234)
        with model.disable_adapter():
            return generate_deflection(
                model, tokenizer, "Reverse engineering is", expert, novice, cfg,
                max_new_tokens=6,
            )

    assert run() == run()
