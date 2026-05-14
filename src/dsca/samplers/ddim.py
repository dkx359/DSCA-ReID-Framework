from __future__ import annotations

import torch
from torch import Tensor


class DDIMScheduler:
    """DDPM forward process for training and deterministic DDIM sampling for inference.

    Training uses the standard noise-prediction objective (paper Eq. 20).
    Inference uses the deterministic DDIM update (paper Eq. 21-22) with optional
    stochasticity controlled by ``eta`` (eta=0.0 is fully deterministic).
    """

    def __init__(self, steps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> None:
        if steps < 1:
            raise ValueError(f"diffusion steps must be >= 1, got {steps}")
        self.steps = steps
        self.betas = torch.linspace(beta_start, beta_end, steps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self._device = torch.device("cpu")

    def to(self, device: torch.device | str) -> "DDIMScheduler":
        device = torch.device(device)
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        self._device = device
        return self

    @property
    def device(self) -> torch.device:
        return self._device

    def sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.randint(0, self.steps, (batch_size,), device=device, dtype=torch.long)

    def _gather(self, values: Tensor, timesteps: Tensor, ref: Tensor) -> Tensor:
        out = values.to(ref.device)[timesteps.to(values.device)].to(ref.device)
        return out.view(-1, *([1] * (ref.dim() - 1)))

    def add_noise(self, x0: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        """Forward diffusion: q(z_t | z_0) (paper Eq. 19)."""
        a = self._gather(self.alpha_bars, timesteps, x0)
        return torch.sqrt(a) * x0 + torch.sqrt(1.0 - a) * noise

    def predict_x0(self, zt: Tensor, pred_noise: Tensor, timesteps: Tensor) -> Tensor:
        """Recover the clean latent estimate x_hat_0 (paper Eq. 21)."""
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
        """One DDIM update from step ``t`` to ``next_t`` (paper Eq. 22).

        ``eta=0`` gives the deterministic DDIM sampler used at inference.
        """
        a_t = self.alpha_bars[t].to(zt.device)
        a_next = self.alpha_bars[max(next_t, 0)].to(zt.device)
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
