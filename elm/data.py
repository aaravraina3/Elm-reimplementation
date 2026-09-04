"""Corpora, concept strings, and persona prompts for WMDP cyber.

Both cyber corpora come straight from `cais/wmdp-corpora` and need no
authentication. Only the *bio* forget set is gated: the reference reads it from
a local `data/bio-remove-dataset.jsonl` that ships with neither the repo nor
the dataset, and which requires the WMDP team's access form.

Prompt templates are reproduced verbatim from trainscripts/erase.py in
rohitgandikota/erasing-llm so the objective is faithful. Only a
representative subset is kept; the reference has ten of each.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import load_dataset

# The reference builds the concept string by joining RMU keywords with ", ".
CYBER_KEYWORDS = (
    "exploit development",
    "malware analysis",
    "reverse engineering",
    "penetration testing",
    "vulnerability research",
)
CYBER_CONCEPT = ", ".join(CYBER_KEYWORDS)

# c_n in the paper: the persona we want *less* of.
EXPERT_TEMPLATES = (
    "Here is a text written by an expert in the field of {concept}, with detailed technical information and all the knowledge:\n",
    "The following passage is authored by a specialist in {concept}, providing in-depth technical details and comprehensive knowledge:\n",
    "An authority on {concept} has written this text, offering precise technical information and extensive expertise:\n",
    "Below is a detailed explanation from a {concept} expert, containing thorough technical data and professional insights:\n",
)

# c_p in the paper: the persona we want *more* of. Note these bundle two
# distinct behaviours, ignorance of the concept AND deflecting to another
# topic. That is a design choice worth questioning: it means the target is not
# purely "a novice", and it likely does some of the work the fluency term is
# credited with.
NOVICE_TEMPLATES = (
    "The text is written by a novice, with no knowledge about {concept} and steering the conversation to random fun topics:\n",
    "This passage is from a beginner who knows nothing about {concept} and diverts the discussion to unrelated entertaining subjects:\n",
    "An amateur with zero expertise in {concept} wrote this, changing the topic to various amusing matters:\n",
    "The following text comes from someone unfamiliar with {concept}, who shifts the focus to random enjoyable themes:\n",
)

# Used by the fluency term: appended to a concept prompt before the frozen
# model generates the continuation the adapter is then trained to reproduce.
DEFLECTION_TEMPLATES = (
    "This is a dangerous topic to discuss, shifting the focus to a totally different and fun topic,",
    "Let's avoid this risky subject and talk about something more enjoyable instead,",
    "Moving away from this sensitive issue to discuss a lighter, more entertaining matter,",
    "Changing the subject from this hazardous topic to a more pleasant conversation,",
)


def _clean(texts, min_len: int, max_len: int) -> list[str]:
    """Drop short passages and truncate long ones, as the reference does."""
    return [str(t[:max_len]) for t in texts if t is not None and len(t) > min_len]


def load_cyber_corpora(min_len: int = 50, max_len: int = 700) -> tuple[list[str], list[str]]:
    """Return (forget, retain) passages for WMDP cyber.

    Neither split requires an HF token.
    """
    forget = load_dataset("cais/wmdp-corpora", "cyber-forget-corpus", split="train")["text"]
    retain = load_dataset("cais/wmdp-corpora", "cyber-retain-corpus", split="train")["text"]
    return _clean(forget, min_len, max_len), _clean(retain, min_len, max_len)


def load_cyber_mcq(path: str | Path) -> list[dict]:
    """Load the WMDP cyber multiple-choice questions bundled with the reference
    repo (data/wmdp/cyber-questions.json)."""
    with open(path) as fp:
        return json.load(fp)


def sample_personas(concept: str, rng: random.Random) -> tuple[str, str]:
    """One (expert, novice) prefix pair. The reference resamples every step, so
    the objective is averaged over paraphrases rather than tied to one wording."""
    return (
        rng.choice(EXPERT_TEMPLATES).format(concept=concept),
        rng.choice(NOVICE_TEMPLATES).format(concept=concept),
    )
