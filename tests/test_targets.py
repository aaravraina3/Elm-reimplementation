"""Property tests for the ELM objective.

The target construction has properties that follow directly from the
derivation, so they can be asserted rather than eyeballed. No model is loaded:
`combine_logprobs` is pure tensor math, so these run in milliseconds offline.
"""

import pytest
import torch

from elm.targets import combine_logprobs


def logprobs(n_positions: int, vocab: int, seed: int) -> torch.Tensor:
    """A plausible (n_positions, vocab) log-probability tensor."""
    g = torch.Generator().manual_seed(seed)
    return torch.log_softmax(torch.randn(n_positions, vocab, generator=g), dim=-1)


def triple(n_positions=6, vocab=64):
    return (
        logprobs(n_positions, vocab, 0),
        logprobs(n_positions, vocab, 1),
        logprobs(n_positions, vocab, 2),
    )


def kl_from_original(target: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    """Per-position KL(target || exp(original))."""
    safe = target.clamp_min(1e-12)
    return (target * (safe.log() - original)).sum(dim=-1)


def test_eta_zero_recovers_the_original_distribution():
    """eta=0 removes the guidance term entirely, so P' must equal P exactly."""
    original, expert, novice = triple()
    target = combine_logprobs(original, expert, novice, 0.0, 0.0, top_k=None)
    assert torch.allclose(target, original.exp(), atol=1e-6)


def test_rows_are_probability_distributions():
    original, expert, novice = triple()
    target = combine_logprobs(original, expert, novice, 1.0, 1000.0, top_k=50)
    assert torch.allclose(target.sum(dim=-1), torch.ones(target.shape[0]), atol=1e-5)
    assert (target >= 0).all()


@pytest.mark.parametrize("top_k", [None, 50])
def test_no_nan_or_inf_at_extreme_eta(top_k):
    """The reference ramps eta to 1000, so the arithmetic has to survive it."""
    original, expert, novice = triple()
    target = combine_logprobs(original, expert, novice, 1.0, 1000.0, top_k=top_k)
    assert torch.isfinite(target).all()


def test_divergence_grows_with_eta():
    """eta is the erasure strength dial, so the target must move monotonically
    further from the original as it increases."""
    original, expert, novice = triple()
    divergences = []
    for eta in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]:
        target = combine_logprobs(original, expert, novice, eta, eta, top_k=None)
        divergences.append(kl_from_original(target, original).sum().item())
    assert divergences == sorted(divergences)
    assert divergences[0] == pytest.approx(0.0, abs=1e-6)
    assert divergences[-1] > divergences[0]


def test_eta_ramp_edits_later_positions_harder():
    """The reference ramps eta across token positions rather than using a
    constant, so early tokens are nearly untouched and late tokens are pushed
    hard. That ramp is in the code but not in the paper's equations.

    To isolate the ramp we hold the guidance direction identical at every
    position, so the only thing varying down the sequence is eta. Comparing
    positions with different guidance vectors would confound the two.
    """
    n = 8
    row_o, row_e, row_n = (logprobs(1, 64, s) for s in (0, 1, 2))
    original, expert, novice = (r.repeat(n, 1) for r in (row_o, row_e, row_n))

    target = combine_logprobs(original, expert, novice, 0.0, 8.0, top_k=None)
    entropy = -(target.clamp_min(1e-30) * target.clamp_min(1e-30).log()).sum(dim=-1)

    # Higher eta concentrates the target, so entropy must fall down the sequence.
    assert torch.all(entropy[1:] <= entropy[:-1] + 1e-6)
    assert entropy[-1] < entropy[0]
    # Position 0 has eta=0, so it is exactly the original.
    assert torch.allclose(target[0], row_o.exp()[0], atol=1e-6)


def test_target_saturates_to_one_hot_at_the_reference_eta():
    """Finding, not a spec: the reference default is --eta 1000, and by eta
    around 50 the target is numerically one-hot. Past that point the "soft"
    erase loss (cross-entropy against a distribution) and the hard branch
    (cross-entropy against the target's argmax) receive the same signal, and
    top_k stops mattering because fewer than k entries survive anyway.

    Confirmed on real log-probs, not just these synthetic ones. Measured with
    scripts/measure_eta_saturation.py on Qwen2.5-0.5B-Instruct over WMDP
    cyber forget-corpus text, std of (novice - expert) = 2.15:

        eta      mean max prob   mean entropy   mean support
        1             0.42           3.59          9823
        20            0.82           0.75           877
        100           0.94           0.15             6.8
        1000          0.99           0.02             1.1

    Still to confirm on zephyr-7b-beta, where the guidance magnitude may differ.
    """
    original, expert, novice = triple(n_positions=1, vocab=32000)

    soft = combine_logprobs(original, expert, novice, 1.0, 1.0, top_k=None)
    assert soft.max() < 0.05, "at eta=1 the target should still be diffuse"

    saturated = combine_logprobs(original, expert, novice, 1000.0, 1000.0, top_k=50)
    assert saturated.max() == pytest.approx(1.0, abs=1e-6)
    assert int((saturated > 0).sum()) == 1


def test_top_k_keeps_exactly_k_tokens():
    original, expert, novice = triple(vocab=64)
    target = combine_logprobs(original, expert, novice, 1.0, 10.0, top_k=7)
    assert ((target > 0).sum(dim=-1) == 7).all()


def test_top_k_larger_than_vocab_is_clamped():
    original, expert, novice = triple(vocab=16)
    target = combine_logprobs(original, expert, novice, 1.0, 10.0, top_k=999)
    assert ((target > 0).sum(dim=-1) == 16).all()


def test_identical_prefixes_leave_the_original_untouched():
    """If the expert and novice prefixes induce the same distribution, their
    difference is zero and no amount of eta should change anything."""
    original, expert, _ = triple()
    target = combine_logprobs(original, expert, expert, 1.0, 1000.0, top_k=None)
    assert torch.allclose(target, original.exp(), atol=1e-6)


def test_shape_mismatch_is_rejected():
    original, expert, _ = triple()
    novice = logprobs(5, 64, 2)
    with pytest.raises(ValueError, match="shape mismatch"):
        combine_logprobs(original, expert, novice, 1.0, 10.0)


def test_wrong_rank_is_rejected():
    bad = logprobs(6, 64, 0).unsqueeze(0)
    with pytest.raises(ValueError, match="expected"):
        combine_logprobs(bad, bad, bad, 1.0, 10.0)
