from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def spatial_gradient(x: Tensor) -> Tensor:
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return torch.sqrt(dx.square() + dy.square() + 1e-8)


def gaussian_blur(x: Tensor, kernel_size: int = 5, sigma: float = 1.0) -> Tensor:
    if kernel_size <= 1:
        return x
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_x = kernel_1d.view(1, 1, 1, -1).expand(x.shape[1], 1, 1, -1)
    kernel_y = kernel_1d.view(1, 1, -1, 1).expand(x.shape[1], 1, -1, 1)
    x = F.conv2d(x, kernel_x, padding=(0, radius), groups=x.shape[1])
    x = F.conv2d(x, kernel_y, padding=(radius, 0), groups=x.shape[1])
    return x


def resize_like(x: Tensor, reference: Tensor) -> Tensor:
    if x.shape[-2:] == reference.shape[-2:]:
        return x
    return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)
