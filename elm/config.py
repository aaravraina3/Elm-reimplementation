from dataclasses import dataclass


@dataclass(frozen=True)
class ELMConfig:
    """Every hyperparameter in one place.

    Defaults reproduce the released ELM WMDP adapter's LoRA setup exactly
    (see reference/elm-zephyr-wmdp/adapter_config.json), on a base model small
    enough to train locally. scripts/check_config_fidelity.py asserts the match.
    """

    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # LoRA. These mirror the released adapter.
    rank: int = 4
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    layers_to_transform: tuple[int, ...] = (4, 5, 6, 7)
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    # ELM objective. eta ramps across token positions; the reference
    # implementation hardcodes eta_start=1 and exposes only eta_end (--eta 1000).
    eta_start: float = 1.0
    eta_end: float = 1000.0
    top_k: int | None = 50
    temperature: float | None = 1.2
    # Guidance scale used only when sampling the fluency target
    # (the reference passes gamma=3 to its generate()).
    fluency_guidance: float = 3.0
    fluency_max_new_tokens: int = 96

    # Loss weights.
    erase_scale: float = 1.0
    retain_scale: float = 1.0
    fluency_scale: float = 1.0

    # Optimization.
    lr: float = 5e-5
    grad_accum: int = 4
    num_steps: int = 300
    min_len: int = 50
    max_len: int = 700
    seed: int = 0
