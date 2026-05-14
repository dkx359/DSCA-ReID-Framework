from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from dsca.models.extractors import DSCAConditions
from dsca.models.masks import gaussian_blur, spatial_gradient


@dataclass
class USCCOutput:
    """Bundle of everything the dual-pathway injection needs.

    condition          : fused semantic condition c_t (paper Eq. 11)
    pose/mask/style_token : per-modality condition tokens
    gates              : reliability-aware fusion weights (omega_p, omega_m, omega_s)
    skeleton           : normalised pose skeleton heatmap (full resolution)
    edge_masks         : multi-scale edge/skeleton attention masks (paper Eq. 13)
    foreground_priors  : multi-scale foreground attention priors (paper Eq. 16)
    """

    condition: Tensor
    pose_token: Tensor
    mask_token: Tensor
    style_token: Tensor
    gates: Tensor
    skeleton: Tensor
    edge_masks: dict[str, Tensor]
    foreground_priors: dict[str, Tensor]


class DepthwiseTokenAdapter(nn.Module):
    """Lightweight modality adapter for pose / mask conditions (paper Sec. III-B)."""

    def __init__(self, in_channels: int, hidden: int, condition_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.GroupNorm(1, in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, hidden, 1),
            nn.GroupNorm(max(1, min(8, hidden // 4)), hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GroupNorm(max(1, min(8, hidden // 4)), hidden),
            nn.SiLU(inplace=True),
        )
        self.proj = nn.Linear(hidden, condition_dim)
        self.norm = nn.LayerNorm(condition_dim)

    def forward(self, x: Tensor) -> Tensor:
        feat = self.net(x).mean(dim=(-2, -1))
        return self.norm(self.proj(feat))


class StyleTokenAdapter(nn.Module):
    """Two-layer MLP with LayerNorm + residual for the style vector (paper Sec. III-B)."""

    def __init__(self, style_dim: int, condition_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(style_dim, condition_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(condition_dim),
            nn.Linear(condition_dim, condition_dim * 2),
            nn.GELU(),
            nn.Linear(condition_dim * 2, condition_dim),
        )
        self.norm = nn.LayerNorm(condition_dim)

    def forward(self, style: Tensor) -> Tensor:
        base = self.proj(style)
        return self.norm(base + self.net(base))


class UnifiedSemanticConditionComposer(nn.Module):
    """USCC: fuse pose, mask and style semantics into a single condition c_t.

    Implements the reliability-aware, temperature-annealed gating with a
    probability floor (paper Eq. 17), the composed condition (Eq. 11), and the
    multi-scale edge/skeleton masks and foreground priors (Eq. 13, 16).
    """

    def __init__(
        self,
        pose_channels: int,
        style_dim: int,
        condition_dim: int,
        mask_scales: list[int],
        probability_floor: float = 1e-3,
        lambda_boundary: float = 1.0,
        lambda_skeleton: float = 1.0,
        sharpness: float = 5.0,
    ) -> None:
        super().__init__()
        hidden = max(32, condition_dim // 2)
        self.pose_adapter = DepthwiseTokenAdapter(pose_channels, hidden, condition_dim)
        self.mask_adapter = DepthwiseTokenAdapter(1, hidden, condition_dim)
        self.style_adapter = StyleTokenAdapter(style_dim, condition_dim)
        self.pose_proj = nn.Linear(condition_dim, condition_dim)
        self.mask_proj = nn.Linear(condition_dim, condition_dim)
        self.style_proj = nn.Linear(condition_dim, condition_dim)
        self.gate = nn.Sequential(
            nn.LayerNorm(condition_dim * 3 + 3),
            nn.Linear(condition_dim * 3 + 3, condition_dim),
            nn.GELU(),
            nn.Linear(condition_dim, 3),
        )
        self.mask_scales = list(dict.fromkeys(int(s) for s in mask_scales if s >= 1)) or [1]
        self.probability_floor = probability_floor
        self.lambda_boundary = lambda_boundary
        self.lambda_skeleton = lambda_skeleton
        self.sharpness = sharpness

    def forward(self, conditions: DSCAConditions, temperature: float = 1.0) -> USCCOutput:
        tp = self.pose_adapter(conditions.pose)
        tm = self.mask_adapter(conditions.mask)
        ts = self.style_adapter(conditions.style)
        stats = self._reliability_stats(conditions)
        logits = self.gate(torch.cat([tp, tm, ts, stats], dim=1))
        gates = self._temperature_softmax(logits, temperature)
        condition = (
            gates[:, 0:1] * self.pose_proj(tp)
            + gates[:, 1:2] * self.mask_proj(tm)
            + gates[:, 2:3] * self.style_proj(ts)
        )
        skeleton, edge_masks, foreground_priors = self._build_masks(conditions.pose, conditions.mask)
        return USCCOutput(
            condition=condition,
            pose_token=tp,
            mask_token=tm,
            style_token=ts,
            gates=gates,
            skeleton=skeleton,
            edge_masks=edge_masks,
            foreground_priors=foreground_priors,
        )

    @staticmethod
    def gate_entropy(gates: Tensor) -> Tensor:
        """Negative entropy of the fusion weights; minimising -H discourages
        single-modality dominance (paper gate-diversity intention)."""
        g = gates.clamp_min(1e-8)
        return (g * g.log()).sum(dim=1).mean()

    def _temperature_softmax(self, logits: Tensor, temperature: float) -> Tensor:
        t = max(float(temperature), 1e-6)
        # numerically stable: subtract row max before exponentiating
        scaled = logits / t
        scaled = torch.exp(scaled - scaled.amax(dim=1, keepdim=True))
        scaled = torch.clamp(scaled, min=self.probability_floor)
        return scaled / scaled.sum(dim=1, keepdim=True).clamp_min(1e-8)

    @staticmethod
    def _reliability_stats(conditions: DSCAConditions) -> Tensor:
        # pose: mean peak keypoint confidence
        pose_conf = conditions.pose.amax(dim=1, keepdim=True).mean(dim=(-2, -1))
        # mask: 1 - mean binary entropy (sharper masks => higher reliability)
        mask = conditions.mask.clamp(1e-6, 1 - 1e-6)
        entropy = -(mask * mask.log() + (1 - mask) * (1 - mask).log()).mean(dim=(-2, -1))
        mask_conf = (1.0 - entropy / 0.6931472).clamp(0.0, 1.0)  # normalise by ln2
        # style: self-similarity proxy via normalised L2 norm
        style_conf = conditions.style.norm(dim=1, keepdim=True) / (conditions.style.shape[1] ** 0.5)
        return torch.cat([pose_conf, mask_conf, style_conf], dim=1)

    def _build_masks(
        self, pose: Tensor, mask: Tensor
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
        boundary = spatial_gradient(mask)
        boundary = boundary / boundary.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        skeleton = pose.amax(dim=1, keepdim=True)
        skeleton = skeleton / skeleton.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        base = self.lambda_boundary * boundary + self.lambda_skeleton * skeleton
        edge_masks: dict[str, Tensor] = {}
        fg_priors: dict[str, Tensor] = {}
        for scale in self.mask_scales:
            size = (max(1, mask.shape[-2] // scale), max(1, mask.shape[-1] // scale))
            down = F.interpolate(base, size=size, mode="bilinear", align_corners=False)
            down = gaussian_blur(down, kernel_size=5, sigma=max(1.0, scale / 2))
            edge_masks[str(scale)] = torch.sigmoid(self.sharpness * down)
            fg = F.interpolate(mask, size=size, mode="bilinear", align_corners=False)
            fg_priors[str(scale)] = fg.clamp_min(self.probability_floor)
        return skeleton, edge_masks, fg_priors
