"""Measure how sharp the ELM erase target gets as a function of eta, on real
model log-probabilities over real WMDP cyber forget-corpus text.

Motivation: the reference trains with `--eta 1000` and ramps eta from 1 to
1000 across token positions. On synthetic log-probs the target collapses to
one-hot by eta ~= 50. If that also holds on real log-probs then for most token
positions the "soft" erase target carries no more information than its argmax,
which would make --use_erase_soft_loss and --topk inert over most of a sequence.

Usage:
    python scripts/measure_eta_saturation.py
"""

import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from elm.config import ELMConfig
from elm.data import CYBER_CONCEPT, load_cyber_corpora, sample_personas
from elm.targets import combine_logprobs

N_PASSAGES = 8
MAX_TOKENS = 192
ETAS = [0.5, 1, 2, 5, 10, 20, 50, 100, 1000]


def logprobs(model, tokenizer, text, device, temperature):
    ids = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=MAX_TOKENS + 64).to(device)
    with torch.no_grad():
        logits = model(**ids).logits[0].float()
    return torch.log_softmax(logits / temperature, dim=-1)


def main():
    cfg = ELMConfig()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"model {cfg.model_id} on {device}, temperature {cfg.temperature}\n")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=torch.float32).to(device)
    model.eval()

    forget, _ = load_cyber_corpora(cfg.min_len, cfg.max_len)
    rng = random.Random(cfg.seed)
    passages = rng.sample(forget, N_PASSAGES)

    rows, guidance_std = {e: [] for e in ETAS}, []

    for text in passages:
        expert_prefix, novice_prefix = sample_personas(CYBER_CONCEPT, rng)
        original = logprobs(model, tokenizer, text, device, cfg.temperature)
        expert = logprobs(model, tokenizer, expert_prefix + text, device, cfg.temperature)
        novice = logprobs(model, tokenizer, novice_prefix + text, device, cfg.temperature)

        n = min(original.shape[0], MAX_TOKENS)
        original, expert, novice = original[:n], expert[-n:], novice[-n:]
        guidance_std.append((novice - expert).std().item())

        for eta in ETAS:
            t = combine_logprobs(original, expert, novice, eta, eta, top_k=None)
            safe = t.clamp_min(1e-30)
            rows[eta].append((
                t.max(dim=-1).values.mean().item(),
                -(safe * safe.log()).sum(dim=-1).mean().item(),
                (t > 1e-6).sum(dim=-1).float().mean().item(),
            ))

    vocab = model.config.vocab_size
    print(f"vocab {vocab:,} | {N_PASSAGES} passages | "
          f"std of (novice - expert) = {sum(guidance_std)/len(guidance_std):.3f}\n")
    print(f"{'eta':>6} {'mean max prob':>14} {'mean entropy':>13} {'mean support':>13}")
    print("-" * 50)
    for eta in ETAS:
        mp = sum(r[0] for r in rows[eta]) / len(rows[eta])
        en = sum(r[1] for r in rows[eta]) / len(rows[eta])
        su = sum(r[2] for r in rows[eta]) / len(rows[eta])
        print(f"{eta:>6} {mp:>14.6f} {en:>13.4f} {su:>13.1f}")

    print(f"\nfor reference, eta=0 is the unmodified model: "
          f"entropy {-(original.exp()*original).sum(dim=-1).mean().item():.4f}")


if __name__ == "__main__":
    main()
