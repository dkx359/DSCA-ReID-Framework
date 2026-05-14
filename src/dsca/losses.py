from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from dsca.models.surrogates import SurrogateEnsemble
from dsca.models.uscc import UnifiedSemanticConditionComposer


class IdentityTransferLoss(nn.Module):
    """Transferable Identity Alignment loss L_id-transfer (paper Eq. 27).

    Averages the squared Euclidean distance between the generated camouflage
    query and the target reference across the frozen surrogate ensemble.
    """

    def __init__(self, surrogate_ensemble: SurrogateEnsemble) -> None:
        super().__init__()
        self.surrogates = surrogate_ensemble

    def forward(self, generated: Tensor, target: Tensor) -> Tensor:
        gen_feats = self.surrogates(generated)
        with torch.no_grad():
            target_feats = self.surrogates(target)
        total = generated.new_tensor(0.0)
        for fg, ft in zip(gen_feats, target_feats):
            # squared Euclidean per sample, then batch mean
            total = total + (fg - ft).square().sum(dim=1).mean()
        return total / max(1, len(gen_feats))


def ssim_loss(x: Tensor, y: Tensor, window_size: int = 7) -> Tensor:
    pad = window_size // 2
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=pad)
    mu_y = F.avg_pool2d(y, window_size, stride=1, padding=pad)
    sigma_x = F.avg_pool2d(x * x, window_size, stride=1, padding=pad) - mu_x.square()
    sigma_y = F.avg_pool2d(y * y, window_size, stride=1, padding=pad) - mu_y.square()
    sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=pad) - mu_x * mu_y
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    ssim_map = numerator / denominator.clamp_min(1e-8)
    return 1.0 - ssim_map.mean()


class TinyPerceptualLoss(nn.Module):
    """Lightweight LPIPS-style perceptual proxy (frozen feature extractor).

    Replace with a real LPIPS network for paper-faithful evaluation; the
    interface is kept identical.
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(False)
        return self

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return F.l1_loss(self.features(x), self.features(y))


class VisualConsistencyLoss(nn.Module):
    """Composite visual consistency loss L_vis (paper Eq. 28):
    LPIPS-style perceptual term + L2 pixel fidelity + (1 - SSIM) structural term.
    """

    def __init__(self, l2_weight: float, ssim_weight: float, perceptual_weight: float) -> None:
        super().__init__()
        self.l2_weight = l2_weight
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight
        self.perceptual = TinyPerceptualLoss() if perceptual_weight > 0 else None

    def forward(self, generated: Tensor, source: Tensor) -> Tensor:
        loss = generated.new_tensor(0.0)
        if self.l2_weight:
            loss = loss + self.l2_weight * F.mse_loss(generated, source)
        if self.ssim_weight:
            loss = loss + self.ssim_weight * ssim_loss(generated, source)
        if self.perceptual_weight and self.perceptual is not None:
            loss = loss + self.perceptual_weight * self.perceptual(generated, source)
        return loss


class DiffusionDenoisingLoss(nn.Module):
    """Standard noise-prediction objective L_diff (paper Eq. 29)."""

    def forward(self, pred_noise: Tensor, true_noise: Tensor) -> Tensor:
        return F.mse_loss(pred_noise, true_noise)


class DSCALoss(nn.Module):
    """Composite DSCA objective L_total (paper Eq. 26).

    L_total = lambda_id * L_id-transfer + lambda_vis * L_vis + lambda_diff * L_diff
              ( + optional lambda_gate * gate-diversity regulariser )

    The gate-diversity term is an optional addition that operationalises the
    paper's "discourage single-modality dominance" intention as an explicit,
    differentiable penalty; set ``lambda_gate=0`` to recover the exact Eq. 26.
    """

    def __init__(
        self,
        surrogate_ensemble: SurrogateEnsemble,
        lambda_id_transfer: float = 1.0,
        lambda_visual: float = 1.5,
        lambda_diffusion: float = 0.5,
        visual_l2_weight: float = 1.0,
        visual_ssim_weight: float = 1.0,
        visual_perceptual_weight: float = 0.1,
        lambda_gate: float = 0.0,
    ) -> None:
        super().__init__()
        self.identity = IdentityTransferLoss(surrogate_ensemble)
        self.visual = VisualConsistencyLoss(
            visual_l2_weight, visual_ssim_weight, visual_perceptual_weight
        )
        self.diffusion = DiffusionDenoisingLoss()
        self.lambda_id_transfer = lambda_id_transfer
        self.lambda_visual = lambda_visual
        self.lambda_diffusion = lambda_diffusion
        self.lambda_gate = lambda_gate

    def forward(
        self, outputs: dict[str, Tensor], source: Tensor, target: Tensor
    ) -> dict[str, Tensor]:
        l_id = self.identity(outputs["generated"], target)
        l_vis = self.visual(outputs["generated"], source)
        l_diff = self.diffusion(outputs["pred_noise"], outputs["true_noise"])
        total = (
            self.lambda_id_transfer * l_id
            + self.lambda_visual * l_vis
            + self.lambda_diffusion * l_diff
        )
        result = {"total": total, "id_transfer": l_id, "visual": l_vis, "diffusion": l_diff}
        if self.lambda_gate and "gates" in outputs:
            l_gate = UnifiedSemanticConditionComposer.gate_entropy(outputs["gates"])
            total = total + self.lambda_gate * l_gate
            result["total"] = total
            result["gate"] = l_gate
        return result
