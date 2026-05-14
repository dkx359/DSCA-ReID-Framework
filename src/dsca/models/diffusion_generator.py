from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from dsca.models.extractors import DSCAConditions
from dsca.models.uscc import UnifiedSemanticConditionComposer, USCCOutput
from dsca.models.dual_path_injection import DualPathInjectionBlock
from dsca.samplers.ddim import DDIMScheduler


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, timesteps: Tensor) -> Tensor:
        half = self.dim // 2
        freq = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = timesteps.float()[:, None] * freq[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return self.mlp(emb)


class LatentEncoder(nn.Module):
    """Frozen-style VAE-encoder stand-in: maps RGB to a 4x-downsampled latent."""

    def __init__(self, latent_channels: int, base_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, latent_channels, 4, stride=2, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class LatentDecoder(nn.Module):
    def __init__(self, latent_channels: int, base_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, base_channels, 4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(base_channels, base_channels, 4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


class ConditionalDenoisingUNet(nn.Module):
    """U-Net with explicit depth scheduling for dual-pathway injection (paper Eq. 25).

    Layer subsets follow the paper:
      * S_str (structure): encoder + early upsampling blocks  -> enc1, enc2, dec2
      * S_sty (appearance): bottleneck + decoder blocks        -> mid, dec1
    Some decoder layers belong to both; this is the only place the pathways
    overlap, mirroring the paper's "mainly covering" wording.
    """

    def __init__(self, latent_channels: int, condition_dim: int) -> None:
        super().__init__()
        self.time = TimeEmbedding(condition_dim)
        self.time_to_channels = nn.Linear(condition_dim, latent_channels)
        self.input = nn.Conv2d(latent_channels * 2, latent_channels, 3, padding=1)

        # Encoder (S_str)
        self.enc1 = DualPathInjectionBlock(latent_channels, condition_dim, use_structure=True, use_style=False)
        self.down1 = nn.Conv2d(latent_channels, latent_channels, 4, stride=2, padding=1)
        self.enc2 = DualPathInjectionBlock(latent_channels, condition_dim, use_structure=True, use_style=False)
        self.down2 = nn.Conv2d(latent_channels, latent_channels, 4, stride=2, padding=1)

        # Bottleneck (S_sty)
        self.mid = DualPathInjectionBlock(latent_channels, condition_dim, use_structure=False, use_style=True)

        # Decoder
        self.up1 = nn.ConvTranspose2d(latent_channels, latent_channels, 4, stride=2, padding=1)
        self.dec1 = DualPathInjectionBlock(latent_channels, condition_dim, use_structure=False, use_style=True)
        self.up2 = nn.ConvTranspose2d(latent_channels, latent_channels, 4, stride=2, padding=1)
        self.dec2 = DualPathInjectionBlock(latent_channels, condition_dim, use_structure=True, use_style=False)

        self.out = nn.Sequential(
            nn.GroupNorm(max(1, min(8, latent_channels // 4)), latent_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(latent_channels, latent_channels, 3, padding=1),
        )

    @staticmethod
    def _match(x: Tensor, ref: Tensor) -> Tensor:
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, zt: Tensor, source_latent: Tensor, timesteps: Tensor, uscc: USCCOutput) -> Tensor:
        temb = self.time(timesteps)
        t_bias = self.time_to_channels(temb + uscc.condition)[:, :, None, None]
        h = self.input(torch.cat([zt, source_latent], dim=1)) + t_bias

        # ---- Encoder: structure anchoring ----
        e1 = self.enc1(h, uscc)
        e2 = self.enc2(self.down1(e1), uscc)
        b = self.down2(e2)

        # ---- Bottleneck: appearance rendering ----
        b = self.mid(b, uscc)

        # ---- Decoder: appearance + late structure refinement ----
        d1 = self.up1(b)
        d1 = self.dec1(self._match(d1, e2) + e2, uscc)
        d2 = self.up2(d1)
        d2 = self.dec2(self._match(d2, e1) + e1, uscc)
        return self.out(d2)


class DSCAGenerator(nn.Module):
    """Conditional Diffusion Camouflage Generator G_theta (paper Sec. III-B/C)."""

    def __init__(
        self,
        pose_channels: int,
        style_dim: int,
        condition_dim: int,
        latent_channels: int,
        base_channels: int,
        diffusion_steps: int,
        beta_start: float,
        beta_end: float,
        mask_scales: list[int],
        probability_floor: float,
        lambda_boundary: float,
        lambda_skeleton: float,
        sharpness: float,
    ) -> None:
        super().__init__()
        self.encoder = LatentEncoder(latent_channels, base_channels)
        self.decoder = LatentDecoder(latent_channels, base_channels)
        self.uscc = UnifiedSemanticConditionComposer(
            pose_channels=pose_channels,
            style_dim=style_dim,
            condition_dim=condition_dim,
            mask_scales=mask_scales,
            probability_floor=probability_floor,
            lambda_boundary=lambda_boundary,
            lambda_skeleton=lambda_skeleton,
            sharpness=sharpness,
        )
        self.denoiser = ConditionalDenoisingUNet(latent_channels, condition_dim)
        self.scheduler = DDIMScheduler(diffusion_steps, beta_start, beta_end)

    def _apply(self, fn):  # type: ignore[override]
        # Ensures the (non-Parameter) scheduler buffers follow .to()/.cuda()/.cpu().
        module = super()._apply(fn)
        try:
            probe = fn(torch.zeros(1))
            self.scheduler.to(probe.device)
        except Exception:
            pass
        return module

    def compose_conditions(self, conditions: DSCAConditions, temperature: float) -> USCCOutput:
        return self.uscc(conditions, temperature=temperature)

    def forward(
        self,
        source: Tensor,
        conditions: DSCAConditions,
        temperature: float = 1.0,
        timesteps: Tensor | None = None,
    ) -> dict[str, Tensor]:
        source_latent = self.encoder(source)
        uscc = self.uscc(conditions, temperature=temperature)
        if timesteps is None:
            timesteps = self.scheduler.sample_timesteps(source.shape[0], source.device)
        noise = torch.randn_like(source_latent)
        zt = self.scheduler.add_noise(source_latent, noise, timesteps)
        pred_noise = self.denoiser(zt, source_latent, timesteps, uscc)
        z0_pred = self.scheduler.predict_x0(zt, pred_noise, timesteps)
        generated = self.decoder(z0_pred)
        return {
            "generated": generated,
            "pred_noise": pred_noise,
            "true_noise": noise,
            "source_latent": source_latent,
            "z0_pred": z0_pred,
            "gates": uscc.gates,
        }

    @torch.no_grad()
    def generate(
        self,
        source: Tensor,
        conditions: DSCAConditions,
        ddim_steps: int = 10,
        temperature: float = 1.0,
        eta: float = 0.0,
    ) -> Tensor:
        source_latent = self.encoder(source)
        uscc = self.uscc(conditions, temperature=temperature)
        indices = self.scheduler.ddim_indices(ddim_steps, source.device)
        start = torch.full((source.shape[0],), indices[0], device=source.device, dtype=torch.long)
        z = self.scheduler.add_noise(source_latent, torch.randn_like(source_latent), start)
        for i in range(len(indices) - 1):
            t, next_t = indices[i], indices[i + 1]
            step_t = torch.full((source.shape[0],), t, device=source.device, dtype=torch.long)
            pred_noise = self.denoiser(z, source_latent, step_t, uscc)
            z = self.scheduler.step(z, pred_noise, t, next_t, eta=eta)
        return self.decoder(z)
