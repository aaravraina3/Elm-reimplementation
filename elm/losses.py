"""The three ELM loss terms.

Each maps onto one of the paper's criteria:

    erase   -> innocence     (no trace of the concept)
    retain  -> specificity   (general capability untouched)
    fluency -> seamlessness  (coherent when prompted for the concept, rather
                              than gibberish)

All three call the model twice: once with the adapter active (the thing being
trained) and once with it disabled (the frozen reference). peft's
`disable_adapter()` context manager makes one set of weights serve as both, so
there is never a second copy of the base model in memory. On a 7B model that
is the difference between fitting and not.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from elm.targets import build_target

IGNORE = -100


def soft_cross_entropy(logits: Tensor, target_probs: Tensor) -> Tensor:
    """Cross-entropy against a full target distribution rather than a label.

    logits: (n_positions, vocab) from the adapted model.
    target_probs: (n_positions, vocab) probabilities, rows summing to 1.

    Equivalent to nn.CrossEntropyLoss with probability targets, written out
    because it is the centre of the method and worth being able to read.
    """
    return -(target_probs * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def erase_loss(model, tokenizer, text, expert_prefix, novice_prefix, cfg) -> Tensor:
    """Pull the adapted model's distribution toward the ELM target.

    Position alignment needs no shift: at every position both the adapted and
    frozen models are predicting the same next token, so the distributions are
    compared position-for-position.
    """
    device = next(model.parameters()).device
    ids = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=cfg.max_len).to(device)

    logits = model(**ids).logits[0]

    with model.disable_adapter():
        target = build_target(
            model, tokenizer, text, expert_prefix, novice_prefix,
            eta_start=cfg.eta_start, eta_end=cfg.eta_end,
            top_k=cfg.top_k, temperature=cfg.temperature,
        )

    n = min(logits.shape[0], target.shape[0])
    return soft_cross_entropy(logits[:n], target[:n].detach())


def retain_loss(model, tokenizer, text, cfg) -> Tensor:
    """Hold the adapted model's distribution at the frozen model's, on text that
    has nothing to do with the concept.

    Reported as KL(frozen || adapted) rather than cross-entropy. The two differ
    only by the frozen model's own entropy, which does not depend on the
    adapter, so the gradients are identical and training is unchanged. But the
    logged number becomes interpretable: KL starts at exactly 0 (the adapter is
    initialised to an identity, since lora_B is zeros) and measures pure drift.
    Cross-entropy instead starts at the frozen model's entropy, which varies by
    passage, so a rising retain term cannot be distinguished from having drawn a
    higher-entropy passage. The reference defaults to `--loss cross`, which is
    why its retain curve is hard to read.
    """
    device = next(model.parameters()).device
    ids = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=cfg.max_len).to(device)

    with torch.no_grad(), model.disable_adapter():
        reference_logits = model(**ids).logits[0].float()
    reference = reference_logits.softmax(dim=-1).detach()
    reference_entropy = -(reference * F.log_softmax(reference_logits, dim=-1)).sum(dim=-1).mean()

    logits = model(**ids).logits[0]
    return soft_cross_entropy(logits, reference) - reference_entropy.detach()


@torch.no_grad()
def generate_deflection(model, tokenizer, prompt, expert_prefix, novice_prefix,
                        cfg, max_new_tokens: int) -> str:
    """Sample a continuation from the *frozen* model with ELM guidance applied
    at every decode step.

    This is the target text for the fluency term. Applying the guidance during
    generation (rather than generating normally) is what makes the continuation
    both on-topic-adjacent and free of the erased concept, so the adapter learns
    to deflect fluently instead of emitting noise.

    Three KV caches are maintained in parallel, one per prefix condition. Cost
    is 3x a normal decode, which is why the reference offers
    prepare_consistency_data.py to precompute these offline.
    """
    device = next(model.parameters()).device
    gamma = cfg.fluency_guidance

    def start(s):
        ids = tokenizer(s, return_tensors="pt", truncation=True,
                        max_length=cfg.max_len).to(device)
        out = model(**ids, use_cache=True)
        return out.logits[:, -1, :].float(), out.past_key_values

    def step(token, cache):
        out = model(input_ids=token, past_key_values=cache, use_cache=True)
        return out.logits[:, -1, :].float(), out.past_key_values

    (l_bare, c_bare) = start(prompt)
    (l_exp, c_exp) = start(expert_prefix + prompt)
    (l_nov, c_nov) = start(novice_prefix + prompt)

    generated = []
    for _ in range(max_new_tokens):
        guided = (
            F.log_softmax(l_bare / cfg.temperature, dim=-1)
            + gamma * (
                F.log_softmax(l_nov / cfg.temperature, dim=-1)
                - F.log_softmax(l_exp / cfg.temperature, dim=-1)
            )
        )
        probs = guided.softmax(dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        if token.item() == tokenizer.eos_token_id:
            break
        generated.append(token.item())
        l_bare, c_bare = step(token, c_bare)
        l_exp, c_exp = step(token, c_exp)
        l_nov, c_nov = step(token, c_nov)

    return tokenizer.decode(generated, skip_special_tokens=True)


def fluency_loss(model, tokenizer, prompt, continuation, cfg) -> Tensor:
    """Plain next-token loss on the deflecting continuation only.

    The prompt is masked out with ignore_index so the adapter is supervised
    purely on producing the continuation, not on reproducing the concept text
    that preceded it.
    """
    device = next(model.parameters()).device
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=cfg.max_len).input_ids
    full = tokenizer(prompt + continuation, return_tensors="pt", truncation=True,
                     max_length=cfg.max_len).to(device)

    n_prompt = prompt_ids.shape[1]
    if full.input_ids.shape[1] <= n_prompt + 1:
        return torch.zeros((), device=device)

    labels = full.input_ids.clone()
    labels[:, :n_prompt] = IGNORE

    logits = model(**full).logits
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=IGNORE,
    )
