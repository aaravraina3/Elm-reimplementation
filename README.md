# ELM reimplementation

A from-scratch reimplementation of **Erasure of Language Memory** (Gandikota,
Feucht, Marks, Bau — NeurIPS 2025, [arXiv:2410.02760](https://arxiv.org/pdf/2410.02760)),
written to be read and walked through rather than to reproduce the paper's
headline numbers on a datacenter GPU.

Reference implementation: [rohitgandikota/erasing-llm](https://github.com/rohitgandikota/erasing-llm).

Everything here runs on a 16 GB M3 MacBook. The method is model-size agnostic,
so the code is identical whether the base model is 0.5B or 7B; only
`ELMConfig.model_id` changes.

---

## 1. What the method actually is

The goal is to remove a concept from a language model such that three things
hold at once:

| criterion | meaning |
|---|---|
| **innocence** | no trace of the concept remains |
| **seamlessness** | the model deflects fluently instead of producing gibberish |
| **specificity** | unrelated capabilities are untouched |

### The derivation

Start by writing down the distribution you want. `P'` should behave like the
original model `P`, reweighted to prefer text that reads as novice-authored
over text that reads as expert-authored:

```
P'(x)  ∝  P(x) · [ P(c_p|x) / P(c_n|x) ] ^ η
```

- `c_p` = "novice in {concept}", the persona we want more of
- `c_n` = "expert in {concept}", the persona we want less of
- the bracketed ratio is > 1 for novice-looking text, < 1 for expert-looking text
- `η` controls how hard you push

**Problem:** `P(c_p|x)` is not computable. A language model does not score
concept labels given text; it scores text given a prefix.

**Bayes reverses the direction.** With `P(c|x) = P(x|c)·P(c)/P(x)` applied to
numerator and denominator, `P(x)` cancels and the `x`-independent priors fold
into the proportionality:

```
P(c_p|x) / P(c_n|x)  ∝  P(x|c_p) / P(x|c_n)
```

Both sides are now computable, because `P(x|c)` just means "prepend `c` to the
text, run the model, read off the probability of `x`".

**A log turns it into an addition** (and avoids underflow from multiplying
thousands of token probabilities):

```
log P'(x)  ∝  log P(x)  +  η · [ log P(x|c_p) − log P(x|c_n) ]
```

For an autoregressive model this holds at every token position, so each of the
three terms is a single forward pass of the **frozen** model over the same text
with a different prefix. That's the contribution: the training target is
computed from the model itself. No labels, no annotation.

### The three losses

| loss | what it does | criterion |
|---|---|---|
| erase | pull the adapted model toward the target above | innocence |
| retain | hold the adapted model at the frozen model on unrelated text | specificity |
| fluency | train on a deflecting continuation sampled from the frozen model | seamlessness |

Only a small LoRA adapter is trained. The base model stays frozen throughout,
and peft's `disable_adapter()` lets one set of weights serve as both the
frozen reference and the model being trained, so there is never a second copy
in memory. On a 7B model that is the difference between fitting and not.

---

## 2. Layout

```
elm/
  config.py      47   every hyperparameter, one frozen dataclass
  targets.py    151   the objective + the derivation above, in the docstring
  data.py        89   WMDP cyber corpora, concept string, persona templates
  losses.py     169   erase / retain / fluency, plus guided decoding
  train.py      148   the loop
  evaluate.py   103   MCQ accuracy and perplexity
tests/                34 tests total
  conftest.py         tiny randomly-initialised Qwen2 + real tokenizer
  test_targets.py     13 property tests on the objective
  test_losses.py      13 tests on the three loss terms
  test_evaluate.py     8 tests on the metrics
scripts/
  run_experiment.py           baseline eval -> train -> post eval -> JSON
  measure_eta_saturation.py   the eta experiment (see FINDINGS.md §1)
  check_config_fidelity.py    asserts our LoRA config == the released one
notebooks/
  confirm_saturation_7b.ipynb Colab notebook repeating the eta experiment on
                              zephyr-7b-beta, which needs a CUDA GPU
reference/
  elm-zephyr-wmdp/            the released ELM WMDP adapter, weights + config
```

`FINDINGS.md` holds the observations on the reference implementation.

### The one function that matters

`elm/targets.py::combine_logprobs` is the paper. It takes three aligned
log-probability tensors and returns the target distribution:

```python
eta  = torch.linspace(eta_start, eta_end, n_positions).unsqueeze(1)
edit = original + eta * (novice - expert)
# optional top-k truncation, then softmax
```

`build_target` wraps it with tokenization, the three forward passes, and
position alignment.

---

## 3. What changed from the reference, and why

This is the part worth discussing. Every deviation is deliberate.

### Structural

**1. The pure math is separated from the model plumbing.**
The reference's `get_edit_vector` is one 75-line function doing tokenization,
three forward passes, attention-mask alignment, and the combination. Here
`combine_logprobs` is pure tensor math — no model, no tokenizer, no device —
and `build_target` does the plumbing around it.

This is not a style preference. It is what made the objective testable without
loading a model, and running those tests is what surfaced the eta saturation
result in `FINDINGS.md` §1.

**2. Position alignment is a documented tail slice, not mask arithmetic.**
The reference builds padded attention masks to line the prefixed sequences up
with the bare one. The underlying fact is simpler: for a causal LM `logits[i]`
predicts token `i+1`, so a prefix shifts the text's positions right by exactly
the prefix's token count, and the final `n` rows of a prefixed sequence predict
the same `n` tokens as all `n` rows of the unprefixed one. So: `expert[-n:]`.
Same result, and the reasoning is in a comment.

**3. peft only.** The reference ships `utils/lora.py`, a 205-line hand-rolled
LoRA carried over from the author's diffusion repos, which the training script
never imports. Omitted here.

### Correctness and readability

**4. The sign convention follows the paper.**
The reference names the *expert* prompt `positive_concept_prompt`, computes
`(expert − novice)`, and then negates eta when `action == 'erase'`. The paper's
`c_p` is the *novice*. So code-`positive` and paper-`c_p` mean opposite things,
reconciled by a hidden sign flip.

Here `eta > 0` always means "erase harder" and the direction lives in the
subtraction order, `(novice − expert)`. Algebraically identical, one fewer
thing to misread.

**5. One backward pass, not three.**
The reference calls `.backward()` separately on each loss term. Gradients
accumulate so it is numerically equivalent, but it is three backward passes per
sample, and there is a commented-out `# loss += consistency_loss` at
`erase.py:580` where the same consolidation was attempted. Here the three
losses are summed and backpropagated once.

**6. Loss-enable flags cannot be shadowed.**
`erase.py:371` sets `retain_loss = False`, `:374` sets it `True`, and `:552`
rebinds the same name to the loss tensor. From step 2 onward `if retain_loss:`
is testing a tensor's truthiness — so a retain loss of exactly 0.0 would
silently disable the term for the rest of the run. Here the scales live in a
frozen dataclass and are never rebound.

**7. Retain is KL, not cross-entropy.**
`KL(frozen ‖ adapted)` and cross-entropy differ only by the frozen model's own
entropy, which does not depend on the adapter — so **the gradients are
identical and training is unchanged**. But KL starts at exactly 0.0 and
measures pure drift, while cross-entropy starts at the passage's base entropy,
so a rising retain term can't be told apart from having drawn a
higher-entropy passage. Verified: step-0 retain loss here is `-2.4e-07`.

**8. Fluency masks the prompt with `ignore_index`.**
The reference slices logits and labels by hand
(`[:, prompt_len:][:, 1:]` against `[:, prompt_len:, :][:, :-1, :]`). Here the
prompt positions are set to `-100` and passed to `cross_entropy`. Same
supervision, much harder to get off by one.

**9. `soft_cross_entropy` is written out.** The reference relies on
`nn.CrossEntropyLoss` accepting probability targets. Since this is the centre
of the method, it is spelled out as
`-(target * log_softmax(logits)).sum(-1).mean()`.

**10. Guided decoding is an explicit 3-stream KV-cached loop.**
The reference implements fluency-target generation as a `LogitsProcessor`
subclass wired into `model.generate`. Here `generate_deflection` maintains three
KV caches (bare, expert-prefixed, novice-prefixed), combines the last-position
logits, and samples. Costs the same 3x, and you can read it top to bottom.

**11. Device-agnostic.** The reference hardcodes `cuda:0` and calls
`torch.cuda.empty_cache()`. `pick_device()` here resolves cuda → mps → cpu.

### New

**12. A test suite** (34 tests). The reference has none.

`test_targets.py` asserts things that follow directly from the derivation, so
they are checkable rather than eyeballed:

- at `eta=0` the target equals the original distribution exactly
- KL from the original increases monotonically in eta
- identical expert/novice prefixes leave the original untouched at any eta
- `top_k=k` yields exactly k tokens with mass
- the eta ramp lowers entropy monotonically down the sequence
- the target saturates to one-hot at the reference's default eta
- shape and rank mismatches raise

`test_losses.py` and `test_evaluate.py` run against a randomly initialised
tiny Qwen2 (`conftest.py`) with the real tokenizer, so they assert behaviour
without needing a capable model or a GPU. The load-bearing ones:

- retain loss is exactly 0 for an untrained adapter, which simultaneously
  proves `disable_adapter()` is wired correctly and that lora_B starts at zeros
- fluency loss is recomputed by hand and required to match exactly, which is
  what would catch an off-by-one in the label shift
- MCQ accuracy scores integer answers and letter answers identically, so a
  dataset format change cannot silently shift the metric
- perplexity is exactly `exp(pooled NLL / pooled tokens)`, pooled across
  passages rather than averaged per-passage
- guided decoding is deterministic under a fixed seed

The whole suite runs in about 30 seconds with no GPU.

**13. `scripts/check_config_fidelity.py`.** Builds our `LoraConfig` and diffs
it against the released adapter's `adapter_config.json`, key by key. Runs in a
second, needs no GPU, and verifies exactly the part of a reimplementation that
*can* be verified exactly. Currently 11/11 keys match.

**14. `scripts/measure_eta_saturation.py`.** An experiment the reference does
not contain, measuring how sharp the erase target gets as a function of eta on
real model log-probabilities. Result in `FINDINGS.md` §1.

---

## 4. Running it

```bash
uv venv --python python3.12 .venv
uv pip install --python .venv/bin/python torch transformers peft datasets accelerate safetensors pytest
uv pip install --python .venv/bin/python -e .
```

Tests, about a second, no model download:

```bash
.venv/bin/python -m pytest tests/ -q
```

Config fidelity against the released adapter:

```bash
.venv/bin/python scripts/check_config_fidelity.py
```

The eta experiment:

```bash
.venv/bin/python scripts/measure_eta_saturation.py
```

A full experiment (baseline eval, train, post eval, JSON dump to `runs/<name>/`):

```bash
.venv/bin/python scripts/run_experiment.py --model Qwen/Qwen2.5-0.5B-Instruct --steps 400
```

Training alone:

```bash
.venv/bin/python -m elm.train
```

The 7B confirmation of the eta result needs CUDA, so it lives in
`notebooks/confirm_saturation_7b.ipynb`. Open it in Colab, set the runtime to a
T4 GPU, and run top to bottom. It is self-contained and needs no checkout.

> **Setup gotcha.** `uv pip install -e .` writes an editable-install finder
> whose `.pth` hook did not fire on this machine, so `import elm` worked from
> the project directory (pytest puts the rootdir on `sys.path`) but failed
> anywhere else, including from background jobs. Fixed by adding a plain path
> `.pth` to site-packages. `scripts/run_experiment.py` also prepends the project
> root to `sys.path` itself, so it runs regardless of install state.

### Data

WMDP **cyber** is used throughout because both its corpora load from
`cais/wmdp-corpora` with no authentication. WMDP **bio** is not usable here:
the reference reads its forget set from a local `data/bio-remove-dataset.jsonl`
that ships with neither the repo nor the dataset and requires the WMDP team's
access form.

The 2,225 WMDP cyber multiple-choice questions come from the reference repo at
`data/wmdp/cyber-questions.json`.

---

## 5. What has actually been verified

| claim | evidence |
|---|---|
| 12 property tests pass | `pytest tests/ -q`, 1.2s |
| LoRA config matches the released adapter | `check_config_fidelity.py`, 11/11 keys |
| training runs end to end on MPS | smoke run, all losses finite, adapter saved |
| adapter is a true identity at init | untrained adapter's eval numbers are **identical** to frozen |
| retain loss starts at zero | step-0 value `-2.4e-07` |
| the erase target saturates | measured on real log-probs, `FINDINGS.md` §1 |

Baseline, Qwen2.5-0.5B-Instruct: WMDP cyber accuracy **0.317** (chance 0.25),
retain perplexity **12.00**, forget perplexity **13.72**.

## 6. What is not done

- **No full training run yet.** Everything above is a 3–4 step smoke test. There
  is no trained adapter, no loss curves, and no before/after numbers.
- **Tests cover `targets.py` only.** `losses.py` and `evaluate.py` are
  exercised by smoke runs, not asserted.
- **No batching**, same limitation as the reference: batch size is effectively 1.
- **0.5B has only ~6.7 points of headroom** over chance on WMDP cyber, so the
  erasure signal will be weak. Final numbers should come from
  Qwen2.5-1.5B-Instruct.
- **The saturation result is measured at 0.5B**, not on zephyr-7b-beta.
