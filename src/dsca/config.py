from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ImageConfig:
    height: int = 256
    width: int = 128


@dataclass
class ModelConfig:
    pose_channels: int = 17
    style_dim: int = 64
    condition_dim: int = 256
    latent_channels: int = 128
    base_channels: int = 64
    diffusion_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    ddim_steps: int = 10
    ddim_eta: float = 0.0


@dataclass
class USCCConfig:
    temperature_max: float = 2.0
    temperature_min: float = 0.7
    temperature_end: int = 10000
    probability_floor: float = 1e-3
    lambda_boundary: float = 1.0
    lambda_skeleton: float = 1.0
    sharpness: float = 5.0
    mask_scales: list[int] = field(default_factory=lambda: [1, 2, 4, 8])


@dataclass
class LossConfig:
    lambda_id_transfer: float = 1.0
    lambda_visual: float = 1.5
    lambda_diffusion: float = 0.5
    visual_l2_weight: float = 1.0
    visual_ssim_weight: float = 1.0
    visual_perceptual_weight: float = 0.1
    lambda_gate: float = 0.0


@dataclass
class SurrogateConfig:
    embed_dim: int = 256
    num_cnn: int = 2
    num_transformer: int = 1


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-5
    num_workers: int = 4
    log_every: int = 50
    checkpoint_dir: str = "checkpoints"
    grad_accum_steps: int = 1
    use_ema: bool = True
    ema_decay: float = 0.999
    train_extractor: bool = False
    resume: str = ""


@dataclass
class DataConfig:
    train_dir: str = ""
    gallery_dir: str = ""


@dataclass
class DSCAConfig:
    seed: int = 42
    device: str = "cuda"
    image: ImageConfig = field(default_factory=ImageConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    uscc: USCCConfig = field(default_factory=USCCConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    surrogate: SurrogateConfig = field(default_factory=SurrogateConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)


def _update_dataclass(obj: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(obj, key):
            raise KeyError(f"Unknown config key: {key}")
        current = getattr(obj, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _update_dataclass(current, value)
        else:
            setattr(obj, key, value)
    return obj


def load_config(path: str | Path) -> DSCAConfig:
    cfg = DSCAConfig()
    with Path(path).open("r", encoding="utf-8") as f:
        values = yaml.safe_load(f) or {}
    return _update_dataclass(cfg, values)
