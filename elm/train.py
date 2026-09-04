"""ELM training loop.

One deliberate difference from the reference: the three losses are summed and
backpropagated once, rather than each calling .backward() separately. The
reference's version is numerically equivalent (gradients accumulate) but does
three backward passes per sample, and it leaves a commented-out
`# loss += consistency_loss` where the same consolidation was attempted.

A second difference: the reference reuses the name `retain_loss` for both the
"is this term enabled" boolean and the loss tensor, so after the first step
`if retain_loss:` is testing a tensor's truthiness. It works, but a retain loss
of exactly 0.0 would silently disable the term for the rest of the run. Here
the flags live in the config and are never rebound.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from elm.config import ELMConfig
from elm.data import (
    CYBER_CONCEPT,
    DEFLECTION_TEMPLATES,
    load_cyber_corpora,
    sample_personas,
)
from elm.losses import erase_loss, fluency_loss, generate_deflection, retain_loss


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def lora_config(cfg: ELMConfig) -> LoraConfig:
    """The peft config. Mirrors the released ELM WMDP adapter exactly;
    scripts/check_config_fidelity.py asserts that."""
    return LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        layers_to_transform=list(cfg.layers_to_transform),
        target_modules=list(cfg.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )


def build_model(cfg: ELMConfig, device: str):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=torch.float32)
    model = model.to(device)
    model.requires_grad_(False)
    model = get_peft_model(model, lora_config(cfg))
    return model, tokenizer


def train(cfg: ELMConfig | None = None, out_dir: str | Path = "runs/cyber"):
    cfg = cfg or ELMConfig()
    device = pick_device()
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)

    model, tokenizer = build_model(cfg, device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"device {device} | trainable params {sum(p.numel() for p in trainable):,}")

    forget, retain = load_cyber_corpora(cfg.min_len, cfg.max_len)
    print(f"forget {len(forget)} | retain {len(retain)} passages\n")

    optimizer = AdamW(trainable, lr=cfg.lr)
    history = []
    started = time.time()

    for step in range(cfg.num_steps):
        text = forget[step % len(forget)]
        retain_text = retain[step % len(retain)]
        expert_prefix, novice_prefix = sample_personas(CYBER_CONCEPT, rng)

        model.train()
        loss_erase = erase_loss(model, tokenizer, text, expert_prefix, novice_prefix, cfg)
        loss_retain = retain_loss(model, tokenizer, retain_text, cfg)

        # The fluency target is sampled from the frozen model, so the adapter
        # must be off while generating it.
        model.eval()
        deflection = rng.choice(DEFLECTION_TEMPLATES)
        with model.disable_adapter():
            generated = generate_deflection(
                model,
                tokenizer,
                f"{text}. {deflection}",
                expert_prefix,
                novice_prefix,
                cfg,
                cfg.fluency_max_new_tokens,
            )
        model.train()
        continuation = f". {deflection} {generated}"
        loss_fluency = fluency_loss(model, tokenizer, text, continuation, cfg)

        total = (
            cfg.erase_scale * loss_erase
            + cfg.retain_scale * loss_retain
            + cfg.fluency_scale * loss_fluency
        )
        total.backward()

        if (step + 1) % cfg.grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        history.append(
            {
                "step": step,
                "erase": loss_erase.item(),
                "retain": loss_retain.item(),
                "fluency": loss_fluency.item(),
                "total": total.item(),
            }
        )
        if step % 5 == 0 or step == cfg.num_steps - 1:
            h = history[-1]
            print(
                f"step {step:>4} | erase {h['erase']:>8.4f} | retain {h['retain']:>7.4f} "
                f"| fluency {h['fluency']:>7.4f} | total {h['total']:>8.4f} "
                f"| {time.time() - started:>5.0f}s"
            )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    print(f"\nsaved adapter to {out}")
    return history


if __name__ == "__main__":
    train()
