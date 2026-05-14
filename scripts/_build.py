from __future__ import annotations

from dsca.config import DSCAConfig
from dsca.models.diffusion_generator import DSCAGenerator
from dsca.models.extractors import LightweightConditionExtractor
from dsca.models.surrogates import SurrogateEnsemble, build_surrogate_ensemble


def build_generator(cfg: DSCAConfig) -> DSCAGenerator:
    return DSCAGenerator(
        pose_channels=cfg.model.pose_channels,
        style_dim=cfg.model.style_dim,
        condition_dim=cfg.model.condition_dim,
        latent_channels=cfg.model.latent_channels,
        base_channels=cfg.model.base_channels,
        diffusion_steps=cfg.model.diffusion_steps,
        beta_start=cfg.model.beta_start,
        beta_end=cfg.model.beta_end,
        mask_scales=cfg.uscc.mask_scales,
        probability_floor=cfg.uscc.probability_floor,
        lambda_boundary=cfg.uscc.lambda_boundary,
        lambda_skeleton=cfg.uscc.lambda_skeleton,
        sharpness=cfg.uscc.sharpness,
    )


def build_extractor(cfg: DSCAConfig) -> LightweightConditionExtractor:
    return LightweightConditionExtractor(
        pose_channels=cfg.model.pose_channels, style_dim=cfg.model.style_dim
    )


def build_surrogates(cfg: DSCAConfig | None = None) -> SurrogateEnsemble:
    if cfg is None:
        return build_surrogate_ensemble()
    return build_surrogate_ensemble(
        embed_dim=cfg.surrogate.embed_dim,
        num_cnn=cfg.surrogate.num_cnn,
        num_transformer=cfg.surrogate.num_transformer,
    )
