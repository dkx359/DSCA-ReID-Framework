from __future__ import annotations

import torch
from torch import Tensor, nn


class DDIMScheduler(nn.Module):
    """DDPM forward process for training and DDIM sampling for inference.

    The scheduler is an ``nn.Module`` so that diffusion buffers follow normal
    ``.to()``, ``.cuda()``, ``.cpu()``, and dtype conversions without custom
    device hooks in the generator. The buffers are non-persistent because they
    are fully determined by ``steps``, ``beta_start``, and ``beta_end``.
    """

    def __init__(self, steps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> None:
        super().__init__()
        if steps < 1:
            raise ValueError(f"diffusion steps must be >= 1, got {steps}")
        if not 0.0 < beta_start < beta_end < 1.0:
            raise ValueError(
                "expected 0 < beta_start < beta_end < 1, "
                f"got beta_start={beta_start}, beta_end={beta_end}"
            )
        self.steps = steps
        betas = torch.linspace(beta_start, beta_end, steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas", alphas, persistent=False)
        self.register_buffer("alpha_bars", alpha_bars, persistent=False)

    @property
    def device(self) -> torch.device:
        return self.alpha_bars.device

    def sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.randint(0, self.steps, (batch_size,), device=device, dtype=torch.long)

    def _gather(self, values: Tensor, timesteps: Tensor, ref: Tensor) -> Tensor:
        values = values.to(device=ref.device, dtype=ref.dtype)
        out = values[timesteps.to(ref.device)]
        return out.view(-1, *([1] * (ref.dim() - 1)))

    def add_noise(self, x0: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        """Forward diffusion q(z_t | z_0), matching the paper's Eq. 19."""
        a = self._gather(self.alpha_bars, timesteps, x0)
        return torch.sqrt(a) * x0 + torch.sqrt(1.0 - a) * noise

    def predict_x0(self, zt: Tensor, pred_noise: Tensor, timesteps: Tensor) -> Tensor:
        """Recover the clean latent estimate x_hat_0, matching the paper's Eq. 21."""
        a = self._gather(self.alpha_bars, timesteps, zt)
        return (zt - torch.sqrt(1.0 - a) * pred_noise) / torch.sqrt(a).clamp_min(1e-8)

    def ddim_indices(self, num_steps: int, device: torch.device) -> list[int]:
        """Evenly spaced, strictly decreasing, de-duplicated sampling indices ending at 0."""
        num_steps = max(1, min(num_steps, self.steps))
        values = torch.linspace(self.steps - 1, 0, num_steps + 1, device=device)
        idx = [int(round(v.item())) for v in values]
        dedup: list[int] = []
        for v in idx:
            if not dedup or v < dedup[-1]:
                dedup.append(v)
        if dedup[-1] != 0:
            dedup.append(0)
        return dedup

    def step(self, zt: Tensor, pred_noise: Tensor, t: int, next_t: int, eta: float = 0.0) -> Tensor:
        """One DDIM update from step ``t`` to ``next_t``.

        ``eta=0`` gives the deterministic DDIM sampler used by default at inference.
        """
        a_t = self.alpha_bars[t].to(device=zt.device, dtype=zt.dtype)
        a_next = self.alpha_bars[max(next_t, 0)].to(device=zt.device, dtype=zt.dtype)
        x0 = (zt - torch.sqrt(1.0 - a_t) * pred_noise) / torch.sqrt(a_t).clamp_min(1e-8)
        x0 = x0.clamp(-3.0, 3.0)  # mild stabilisation against latent blow-up
        if eta > 0.0 and next_t > 0:
            sigma = eta * torch.sqrt(
                (1 - a_next) / (1 - a_t).clamp_min(1e-8) * (1 - a_t / a_next.clamp_min(1e-8))
            )
            dir_zt = torch.sqrt((1.0 - a_next - sigma**2).clamp_min(0.0)) * pred_noise
            return torch.sqrt(a_next) * x0 + dir_zt + sigma * torch.randn_like(zt)
        dir_zt = torch.sqrt((1.0 - a_next).clamp_min(0.0)) * pred_noise
        return torch.sqrt(a_next) * x0 + dir_zt
