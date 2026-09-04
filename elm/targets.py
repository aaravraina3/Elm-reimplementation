"""Construction of the ELM target distribution.

This is the whole paper. Everything else in the repo is plumbing around it.

Derivation
----------
We want a distribution P' that behaves like the original model P, reweighted to
prefer text that reads as authored by a novice in the concept over text that
reads as authored by an expert:

    P'(x)  is proportional to  P(x) * [ P(c_p|x) / P(c_n|x) ] ** eta

with c_p = "novice in <concept>" (want more of) and c_n = "expert in <concept>"
(want less of). P(c|x) is not computable: a language model does not score
concept labels given text. Bayes' rule reverses the direction,

    P(c|x) = P(x|c) * P(c) / P(x)

and applying it to numerator and denominator cancels P(x) and absorbs the
x-independent priors into the proportionality:

    P(c_p|x) / P(c_n|x)  is proportional to  P(x|c_p) / P(x|c_n)

Both of those *are* computable: P(x|c) means "prepend c to the text as a prefix,
run the model, read off the probability of x". Taking a log turns the product
and the exponent into a sum, which also avoids underflow:

    log P'(x)  =  log P(x) + eta * [ log P(x|c_p) - log P(x|c_n) ]  + const

For an autoregressive model this holds per token position, so each of the three
terms is a single forward pass of the *frozen* model over the same text with a
different prefix. No labels, no annotation: the model supplies its own target.

Sign convention
---------------
Here `eta > 0` always means "erase harder", and the direction lives explicitly
in the subtraction order (novice - expert). The reference implementation names
the *expert* prompt `positive`, computes (expert - novice), and negates eta when
action == 'erase'. The two are algebraically identical; this one is harder to
misread, since in the reference the paper's `c_p` and the code's `positive` mean
opposite things.
"""

from __future__ import annotations

import torch
from torch import Tensor


def combine_logprobs(
    original: Tensor,
    expert: Tensor,
    novice: Tensor,
    eta_start: float,
    eta_end: float,
    top_k: int | None = None,
) -> Tensor:
    """Combine three aligned log-probability tensors into the ELM target.

    Pure tensor math: no model, no tokenizer, no device assumptions. This is
    where every property of the objective lives, which is what makes it
    testable in isolation.

    Args:
        original: (n_positions, vocab) log-probs, no concept prefix.
        expert:   (n_positions, vocab) log-probs, expert prefix. Aligned to
                  `original` position-for-position.
        novice:   (n_positions, vocab) log-probs, novice prefix. Likewise.
        eta_start: erasure strength at the first position.
        eta_end:   erasure strength at the last position. The reference ramps
                   1 -> 1000, so early tokens are barely edited and late ones
                   are pushed hard. This ramp is in the code but not in the
                   paper's equations.
        top_k: if set, all but the top k entries per position are driven to
               probability zero before normalizing.

    Returns:
        (n_positions, vocab) probabilities, each row summing to 1.
    """
    if not original.shape == expert.shape == novice.shape:
        raise ValueError(
            f"shape mismatch: original={tuple(original.shape)} "
            f"expert={tuple(expert.shape)} novice={tuple(novice.shape)}"
        )
    if original.ndim != 2:
        raise ValueError(f"expected (n_positions, vocab), got {tuple(original.shape)}")

    n_positions, vocab = original.shape

    eta = torch.linspace(
        eta_start, eta_end, n_positions,
        device=original.device, dtype=original.dtype,
    ).unsqueeze(1)

    edit = original + eta * (novice - expert)

    if top_k is not None:
        k = min(top_k, vocab)
        kth_best = edit.topk(k, dim=-1).values[:, -1:]
        edit = edit.masked_fill(edit < kth_best, float("-inf"))

    return torch.softmax(edit, dim=-1)


@torch.no_grad()
def build_target(
    model,
    tokenizer,
    text: str,
    expert_prefix: str,
    novice_prefix: str,
    eta_start: float,
    eta_end: float,
    top_k: int | None = 50,
    temperature: float | None = 1.2,
) -> Tensor:
    """Run the frozen model three times and build the ELM target for `text`.

    The caller is responsible for the model being in its unadapted state (for
    a peft model, inside `with model.disable_adapter():`). This function does
    not touch the adapter, so it cannot enforce that.

    Returns:
        (n_positions, vocab) probabilities, aligned to `tokenizer(text)`.
    """
    device = next(model.parameters()).device

    def logprobs(s: str) -> Tensor:
        ids = tokenizer(s, return_tensors="pt").to(device)
        logits = model(**ids).logits[0].float()
        if temperature is not None:
            logits = logits / temperature
        return torch.log_softmax(logits, dim=-1)

    original = logprobs(text)
    expert = logprobs(expert_prefix + text)
    novice = logprobs(novice_prefix + text)

    n = original.shape[0]
    if expert.shape[0] < n or novice.shape[0] < n:
        raise ValueError(
            "prefixed sequence is shorter than the bare text; check the prefixes"
        )

    # Alignment: for a causal LM, logits[i] is the prediction for token i+1.
    # A prefix shifts the text's positions right by the prefix's token count, so
    # the final n rows of a prefixed sequence predict exactly the same n tokens
    # as all n rows of the unprefixed one. Hence the tail slice.
    return combine_logprobs(
        original, expert[-n:], novice[-n:], eta_start, eta_end, top_k
    )
