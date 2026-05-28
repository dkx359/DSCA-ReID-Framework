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


def _require_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _require_non_negative(name: str, value: int | float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def validate_config(cfg: DSCAConfig, require_train_dir: bool = False) -> DSCAConfig:
    """Validate a DSCA config and return it for convenient chaining.

    The checks intentionally cover only generic framework constraints, not
    paper-specific experimental choices. This keeps lightweight examples easy to
    run while still catching common configuration mistakes early.
    """
    _require_positive("image.height", cfg.image.height)
    _require_positive("image.width", cfg.image.width)
    if cfg.image.height % 4 != 0 or cfg.image.width % 4 != 0:
        raise ValueError("image.height and image.width must be divisible by 4 for the latent encoder")

    _require_positive("model.pose_channels", cfg.model.pose_channels)
    _require_positive("model.style_dim", cfg.model.style_dim)
    _require_positive("model.condition_dim", cfg.model.condition_dim)
    _require_positive("model.latent_channels", cfg.model.latent_channels)
    _require_positive("model.base_channels", cfg.model.base_channels)
    _require_positive("model.diffusion_steps", cfg.model.diffusion_steps)
    _require_positive("model.ddim_steps", cfg.model.ddim_steps)
    if cfg.model.ddim_steps > cfg.model.diffusion_steps:
        raise ValueError("model.ddim_steps must be <= model.diffusion_steps")
    if not 0.0 < cfg.model.beta_start < cfg.model.beta_end < 1.0:
        raise ValueError("expected 0 < model.beta_start < model.beta_end < 1")
    _require_non_negative("model.ddim_eta", cfg.model.ddim_eta)

    if cfg.uscc.temperature_min <= 0 or cfg.uscc.temperature_max <= 0:
        raise ValueError("USCC temperatures must be positive")
    if cfg.uscc.temperature_min > cfg.uscc.temperature_max:
        raise ValueError("uscc.temperature_min must be <= uscc.temperature_max")
    _require_positive("uscc.temperature_end", cfg.uscc.temperature_end)
    if not 0.0 < cfg.uscc.probability_floor < 1.0:
        raise ValueError("uscc.probability_floor must be in (0, 1)")
    _require_positive("uscc.sharpness", cfg.uscc.sharpness)
    if not cfg.uscc.mask_scales or any(s <= 0 for s in cfg.uscc.mask_scales):
        raise ValueError("uscc.mask_scales must contain positive integers")

    _require_positive("surrogate.embed_dim", cfg.surrogate.embed_dim)
    _require_non_negative("surrogate.num_cnn", cfg.surrogate.num_cnn)
    _require_non_negative("surrogate.num_transformer", cfg.surrogate.num_transformer)
    if cfg.surrogate.num_cnn + cfg.surrogate.num_transformer <= 0:
        raise ValueError("at least one surrogate model is required")

    _require_positive("train.batch_size", cfg.train.batch_size)
    _require_positive("train.epochs", cfg.train.epochs)
    _require_positive("train.lr", cfg.train.lr)
    _require_non_negative("train.weight_decay", cfg.train.weight_decay)
    _require_non_negative("train.num_workers", cfg.train.num_workers)
    _require_positive("train.log_every", cfg.train.log_every)
    _require_positive("train.grad_accum_steps", cfg.train.grad_accum_steps)
    if not 0.0 < cfg.train.ema_decay < 1.0:
        raise ValueError("train.ema_decay must be in (0, 1)")

    if require_train_dir:
        if not cfg.data.train_dir:
            raise ValueError("Please set data.train_dir in the config file.")
        if not Path(cfg.data.train_dir).exists():
            raise FileNotFoundError(f"data.train_dir does not exist: {cfg.data.train_dir}")
    return cfg


def load_config(path: str | Path, validate: bool = True) -> DSCAConfig:
    cfg = DSCAConfig()
    with Path(path).open("r", encoding="utf-8") as f:
        values = yaml.safe_load(f) or {}
    _update_dataclass(cfg, values)
    if validate:
        validate_config(cfg)
    return cfg
