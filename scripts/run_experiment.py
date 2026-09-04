"""End-to-end ELM run: baseline eval, train, post eval, dump everything to JSON.

    python scripts/run_experiment.py --model Qwen/Qwen2.5-0.5B-Instruct --steps 400

Writes runs/<name>/{history.json,results.json} plus the trained adapter, so the
loss curves and the before/after table can be charted without re-running.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

# Run from anywhere, with or without an editable install.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from elm.config import ELMConfig
from elm.data import load_cyber_corpora, load_cyber_mcq
from elm.evaluate import CONCEPT_PROMPTS, measure
from elm.train import build_model, pick_device, train

QUESTIONS = os.path.expanduser(
    "~/Documents/projects/erasing-llm/data/wmdp/cyber-questions.json"
)
RUNS_ROOT = PROJECT_ROOT / "runs"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--fluency-tokens", type=int, default=48)
    p.add_argument("--mcq-limit", type=int, default=300)
    p.add_argument("--ppl-limit", type=int, default=30)
    p.add_argument("--name", default=None)
    # Loss weights. All 1.0 matches the reference, but the terms are ~20x apart
    # in magnitude (FINDINGS.md §4), so these are the dial that tests it.
    p.add_argument("--erase-scale", type=float, default=1.0)
    p.add_argument("--retain-scale", type=float, default=1.0)
    p.add_argument("--fluency-scale", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    name = args.name or args.model.split("/")[-1].lower()
    out = RUNS_ROOT / name
    out.mkdir(parents=True, exist_ok=True)

    cfg = dataclasses.replace(
        ELMConfig(),
        model_id=args.model,
        num_steps=args.steps,
        grad_accum=args.grad_accum,
        lr=args.lr,
        max_len=args.max_len,
        fluency_max_new_tokens=args.fluency_tokens,
        erase_scale=args.erase_scale,
        retain_scale=args.retain_scale,
        fluency_scale=args.fluency_scale,
    )

    device = pick_device()
    questions = load_cyber_mcq(QUESTIONS)
    forget, retain = load_cyber_corpora(cfg.min_len, cfg.max_len)
    print(f"=== {name} on {device} ===")
    print(f"steps {cfg.num_steps} | grad_accum {cfg.grad_accum} | lr {cfg.lr} "
          f"| max_len {cfg.max_len} | fluency tokens {cfg.fluency_max_new_tokens}")
    print(f"scales: erase {cfg.erase_scale} | retain {cfg.retain_scale} "
          f"| fluency {cfg.fluency_scale}\n")

    # Baseline. The adapter is an identity at init (lora_B is zeros), so this is
    # the frozen model's behaviour; asserted in the smoke tests.
    model, tokenizer = build_model(cfg, device)
    model.eval()
    started = time.time()
    with torch.no_grad():
        before = measure(model, tokenizer, questions, retain, forget,
                         args.mcq_limit, args.ppl_limit, CONCEPT_PROMPTS)
    print(f"before: {before}   ({time.time()-started:.0f}s)\n")
    del model
    if device == "mps":
        torch.mps.empty_cache()

    history = train(cfg, out_dir=str(out / "adapter"))
    (out / "history.json").write_text(json.dumps(history, indent=2))

    # Reload with the trained adapter attached.
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=torch.float32).to(device)
    model = PeftModel.from_pretrained(base, str(out / "adapter")).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)

    started = time.time()
    with torch.no_grad():
        after = measure(model, tokenizer, questions, retain, forget,
                        args.mcq_limit, args.ppl_limit, CONCEPT_PROMPTS)
    print(f"\nafter: {after}   ({time.time()-started:.0f}s)")

    results = {
        "model": cfg.model_id,
        "config": dataclasses.asdict(cfg),
        "mcq_limit": args.mcq_limit,
        "ppl_limit": args.ppl_limit,
        "before": before,
        "after": after,
        "delta": {k: after[k] - before[k] for k in before},
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))

    print(f"\n{'metric':<18} {'before':>10} {'after':>10} {'delta':>10}")
    print("-" * 52)
    for k in before:
        print(f"{k:<18} {before[k]:>10.4f} {after[k]:>10.4f} {after[k]-before[k]:>+10.4f}")
    print(f"\nwrote {out}/results.json and {out}/history.json")


if __name__ == "__main__":
    main()
