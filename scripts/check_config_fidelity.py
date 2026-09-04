"""Assert our LoRA config matches the released ELM WMDP adapter, key for key.

Needs no GPU and no model download. Configuration is the part of a
reimplementation that can be verified exactly rather than approximately, so it
is worth verifying exactly.
"""

import json
from pathlib import Path

from elm.config import ELMConfig
from elm.train import lora_config

REFERENCE = Path(__file__).parent.parent / "reference/elm-zephyr-wmdp/adapter_config.json"
COMPARED = [
    "r",
    "lora_alpha",
    "lora_dropout",
    "layers_to_transform",
    "target_modules",
    "bias",
    "task_type",
    "peft_type",
    "use_dora",
    "use_rslora",
    "init_lora_weights",
]


def normalize(value):
    """peft returns target_modules as a set, the released JSON stores a list, and
    peft_type is a str-enum. Compare by value, not by container type."""
    if isinstance(value, (list, tuple, set, frozenset)):
        return sorted(str(v) for v in value)
    return value.value if hasattr(value, "value") else value


def main() -> int:
    released = json.loads(REFERENCE.read_text())
    ours = lora_config(ELMConfig()).to_dict()

    print(f"{'key':<22} {'ours':<32} {'released':<32} ok")
    print("-" * 92)
    failures = 0
    for key in COMPARED:
        a, b = normalize(ours.get(key)), normalize(released.get(key))
        ok = a == b
        failures += not ok
        print(f"{key:<22} {str(a)[:31]:<32} {str(b)[:31]:<32} {'yes' if ok else 'NO'}")

    print()
    if failures:
        print(f"{failures} key(s) differ from the released adapter")
    else:
        print(f"all {len(COMPARED)} compared keys match the released adapter")
    print(
        "note: base_model_name_or_path differs by design "
        "(small model locally vs zephyr-7b-beta)"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
