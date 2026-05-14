from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from dsca.models.masks import resize_like
from dsca.models.uscc import USCCOutput


class FiLMController(nn.Module):
    """Per-layer FiLM controller implementing paper Eq. 12.

    Predicts raw modulation parameters (gamma_hat, beta_hat) from the current
    features and the selected condition token, then applies a learnable fusion
    strength ``alpha`` so that ``alpha = 0`` degenerates to an identity mapping
    (which stabilises early training).
    """

    def __init__(self, condition_dim: int, channels: int) -> None:
        super().__init__()
        self.feat_summary = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, condition_dim),
        )
        self.to_gamma_beta = nn.Sequential(
            nn.LayerNorm(condition_dim * 2),
            nn.Linear(condition_dim * 2, condition_dim),
            nn.SiLU(inplace=True),
            nn.Linear(condition_dim, channels * 2),
        )
        # alpha initialised to 1.0 and learned end-to-end (paper Sec. III-B).
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, h: Tensor, token: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        ctx = torch.cat([self.feat_summary(h), token], dim=1)
        gamma, beta = self.to_gamma_beta(ctx).chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return gamma, beta, F.softplus(self.alpha)


class ForegroundMaskedCrossAttention(nn.Module):
    """Foreground-biased cross-attention for the appearance pathway (paper Eq. 16).

    Spatial features attend to the style token. The attention logits are biased
    by the (log of the) downsampled foreground mask so that appearance transfer
    is confined to human regions and background leakage is suppressed.
    """

    def __init__(self, channels: int, condition_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = max(1, min(num_heads, channels))
        while channels % self.num_heads != 0:
            self.num_heads -= 1
        self.head_dim = channels // self.num_heads
        self.norm = nn.GroupNorm(max(1, min(8, channels // 4)), channels)
        self.to_q = nn.Conv2d(channels, channels, 1)
        self.to_kv = nn.Linear(condition_dim, channels * 2)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, h: Tensor, token: Tensor, fg_prior: Tensor) -> Tensor:
        b, c, hh, ww = h.shape
        x = self.norm(h)
        q = self.to_q(x).view(b, self.num_heads, self.head_dim, hh * ww)
        k, v = self.to_kv(token).view(b, 2, self.num_heads, self.head_dim, 1).unbind(dim=1)
        # logits: [b, heads, hw, 1]
        logits = torch.einsum("bhdn,bhdm->bhnm", q, k) / (self.head_dim**0.5)
        fg = resize_like(fg_prior, h).view(b, 1, hh * ww, 1).clamp_min(1e-6)
        logits = logits + torch.log(fg)
        attn = torch.softmax(logits, dim=2)
        out = torch.einsum("bhnm,bhdm->bhdn", attn, v).reshape(b, c, hh, ww)
        return self.proj(out)


class DualPathInjectionBlock(nn.Module):
    """One U-Net block applying the structure and/or appearance pathway.

    Structure pathway (Eq. 23): ControlNet-style residual gated by the
    multi-scale edge/skeleton mask, with a learnable nonnegative strength.

    Appearance pathway (Eq. 24-25): foreground-masked cross-attention followed
    by a residual FiLM update, again with a learnable strength.
    """

    def __init__(
        self,
        channels: int,
        condition_dim: int,
        use_structure: bool,
        use_style: bool,
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.use_structure = use_structure
        self.use_style = use_style
        self.conv = nn.Sequential(
            nn.GroupNorm(max(1, min(norm_groups, channels // 4)), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        if use_structure:
            # input: [pose skeleton, foreground mask] -> structural residual
            self.structure = nn.Sequential(
                nn.Conv2d(2, channels, 3, padding=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(channels, channels, 3, padding=1),
            )
            self.alpha_structure = nn.Parameter(torch.tensor(1.0))
        if use_style:
            self.cross_attn = ForegroundMaskedCrossAttention(channels, condition_dim)
            self.film = FiLMController(condition_dim, channels)

    def forward(self, h: Tensor, uscc: USCCOutput) -> Tensor:
        h = h + self.conv(h)
        if self.use_structure:
            edge = resize_like(self._select(uscc.edge_masks, h), h)
            skeleton = resize_like(uscc.skeleton, h)
            fg = resize_like(self._select(uscc.foreground_priors, h), h)
            structural_map = torch.cat([skeleton, fg], dim=1)
            residual = self.structure(structural_map)
            # Eq. 23-25: masked structural residual, gated by learnable strength.
            h = h + F.softplus(self.alpha_structure) * edge * residual
        if self.use_style:
            fg = self._select(uscc.foreground_priors, h)
            attended = self.cross_attn(h, uscc.style_token, fg)
            gamma, beta, alpha = self.film(attended, uscc.style_token)
            fg_h = resize_like(fg, h)
            # Eq. 12 / 24: residual FiLM, confined to foreground regions.
            h = h + alpha * fg_h * (gamma * h + beta)
        return h

    @staticmethod
    def _select(masks: dict[str, Tensor], h: Tensor) -> Tensor:
        best_key = min(masks, key=lambda k: abs(masks[k].shape[-2] - h.shape[-2]))
        return masks[best_key]
