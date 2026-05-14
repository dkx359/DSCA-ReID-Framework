from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class DSCAConditions:
    pose: Tensor
    mask: Tensor
    style: Tensor


class ConditionExtractorBase(nn.Module):
    def forward(self, target: Tensor) -> DSCAConditions:  # pragma: no cover
        raise NotImplementedError


class LightweightConditionExtractor(ConditionExtractorBase):
    """Immediate-use condition extractor.

    This lightweight module is intended for immediate code execution and interface validation.
    For paper-level experiments, replace it with OpenPose, DeepLabV3, and CLIP while
    preserving the same output interface.
    """

    def __init__(self, pose_channels: int = 17, style_dim: int = 64) -> None:
        super().__init__()
        self.pose_channels = pose_channels
        self.style_dim = style_dim

    def train(self, mode: bool = True):  # type: ignore[override]
        # The lightweight extractor is parameter-free and deterministic; keep it
        # in eval mode so swapping in real OpenPose/DeepLabV3/CLIP does not change
        # behaviour unexpectedly. Override in production extractors if needed.
        return super().train(False)

    def forward(self, target: Tensor) -> DSCAConditions:
        b, _, h, w = target.shape
        mask = self._foreground_prior(target)
        pose = self._pose_heatmaps(mask, self.pose_channels)
        style = self._style_vector(target, mask, self.style_dim)
        return DSCAConditions(pose=pose, mask=mask, style=style)

    @staticmethod
    def _foreground_prior(x: Tensor) -> Tensor:
        gray = x.mean(dim=1, keepdim=True)
        center_y = torch.linspace(-1.0, 1.0, x.shape[-2], device=x.device, dtype=x.dtype).view(1, 1, -1, 1)
        center_x = torch.linspace(-1.0, 1.0, x.shape[-1], device=x.device, dtype=x.dtype).view(1, 1, 1, -1)
        center = torch.exp(-1.6 * (center_x.square() + 0.35 * center_y.square()))
        local = torch.sigmoid(8.0 * (gray - gray.mean(dim=(-2, -1), keepdim=True)))
        mask = (0.55 * center + 0.45 * local).clamp(0, 1)
        return mask

    @staticmethod
    def _pose_heatmaps(mask: Tensor, channels: int) -> Tensor:
        b, _, h, w = mask.shape
        yy = torch.linspace(0, 1, h, device=mask.device, dtype=mask.dtype).view(1, 1, h, 1)
        xx = torch.linspace(0, 1, w, device=mask.device, dtype=mask.dtype).view(1, 1, 1, w)
        points = torch.linspace(0.08, 0.92, channels, device=mask.device, dtype=mask.dtype)
        heatmaps = []
        for i, y0 in enumerate(points):
            x0 = 0.5 + 0.12 * torch.sin(torch.tensor(float(i), device=mask.device, dtype=mask.dtype))
            sigma_y = 0.035 + 0.01 * (i % 3)
            sigma_x = 0.055
            heat = torch.exp(-((yy - y0).square() / (2 * sigma_y**2) + (xx - x0).square() / (2 * sigma_x**2)))
            heatmaps.append(heat.expand(b, 1, h, w) * mask)
        return torch.cat(heatmaps, dim=1)

    @staticmethod
    def _style_vector(x: Tensor, mask: Tensor, style_dim: int) -> Tensor:
        masked = x * mask
        denom = mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        mean = masked.sum(dim=(-2, -1), keepdim=True) / denom
        var = ((x - mean).square() * mask).sum(dim=(-2, -1), keepdim=True) / denom
        pooled = F.adaptive_avg_pool2d(masked, output_size=(4, 4)).flatten(1)
        stats = torch.cat([mean.flatten(1), var.sqrt().flatten(1), pooled], dim=1)
        if stats.shape[1] < style_dim:
            stats = F.pad(stats, (0, style_dim - stats.shape[1]))
        return stats[:, :style_dim]
