from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class _ConvReIDExtractor(nn.Module):
    """Compact CNN ReID feature extractor (ResNet-style inductive bias)."""

    def __init__(self, embed_dim: int = 256, width: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(width * 4, embed_dim)
        self._freeze()

    def _freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, x: Tensor) -> Tensor:
        feat = self.net(x).flatten(1)
        return F.normalize(self.proj(feat), dim=1)


class _TransformerReIDExtractor(nn.Module):
    """Compact ViT-style ReID feature extractor (attention inductive bias).

    Provides architectural diversity in the surrogate ensemble so that the
    transferability bound (paper Eq. 7) can be studied: a heterogeneous
    CNN + Transformer set compresses the surrogate-victim discrepancy.
    """

    def __init__(self, embed_dim: int = 256, patch: int = 8, dim: int = 128, depth: int = 2) -> None:
        super().__init__()
        self.patch = patch
        self.dim = dim
        self.to_patch = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=4, dim_feedforward=dim * 2, batch_first=True, dropout=0.0
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, embed_dim)
        self._freeze()

    def _freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, x: Tensor) -> Tensor:
        b = x.shape[0]
        tokens = self.to_patch(x).flatten(2).transpose(1, 2)  # [b, n, dim]
        cls = self.cls.expand(b, -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        seq = self.encoder(seq)
        feat = self.norm(seq[:, 0])
        return F.normalize(self.proj(feat), dim=1)


# Backwards-compatible alias (old code imported TinyReIDFeatureExtractor).
TinyReIDFeatureExtractor = _ConvReIDExtractor


class SurrogateEnsemble(nn.Module):
    """Frozen surrogate ensemble F = {f_1, ..., f_K} used only for supervision."""

    def __init__(self, models: list[nn.Module]) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)
        for model in self.models:
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)

    def __len__(self) -> int:
        return len(self.models)

    def forward(self, x: Tensor) -> list[Tensor]:
        return [model(x) for model in self.models]

    def train(self, mode: bool = True):  # type: ignore[override]
        # Surrogates stay frozen regardless of the parent module's train/eval state.
        super().train(False)
        return self


def build_surrogate_ensemble(
    embed_dim: int = 256,
    num_cnn: int = 2,
    num_transformer: int = 1,
) -> SurrogateEnsemble:
    """Build a heterogeneous CNN + Transformer surrogate ensemble.

    The default (2 CNNs + 1 ViT) mirrors the paper's lightweight heterogeneous
    configuration that already achieves near-saturated transferability.
    """
    models: list[nn.Module] = []
    for i in range(max(0, num_cnn)):
        models.append(_ConvReIDExtractor(embed_dim=embed_dim, width=32 + 16 * i))
    for _ in range(max(0, num_transformer)):
        models.append(_TransformerReIDExtractor(embed_dim=embed_dim))
    if not models:
        raise ValueError("Surrogate ensemble must contain at least one model.")
    return SurrogateEnsemble(models)


def build_tiny_surrogate_ensemble(num_models: int = 3, embed_dim: int = 256) -> SurrogateEnsemble:
    """Backwards-compatible builder. Now heterogeneous by default."""
    num_cnn = max(1, num_models - 1)
    num_transformer = 1 if num_models >= 2 else 0
    return build_surrogate_ensemble(embed_dim=embed_dim, num_cnn=num_cnn, num_transformer=num_transformer)
