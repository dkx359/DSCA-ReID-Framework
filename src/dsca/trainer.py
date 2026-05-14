from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dsca.config import DSCAConfig
from dsca.losses import DSCALoss
from dsca.models.extractors import ConditionExtractorBase
from dsca.models.diffusion_generator import DSCAGenerator
from dsca.models.surrogates import SurrogateEnsemble


class EMA:
    """Exponential moving average of generator weights (improves sample quality)."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)

    def state_dict(self) -> dict:
        return self.shadow.state_dict()


class DSCATrainer:
    """Offline generator training (paper Sec. III-E, Algorithm 2 Part 1).

    Adds practical training machinery from the paper's setup: mixed precision,
    cosine-annealed learning rate, gradient accumulation, EMA, and resumable
    checkpoints. Surrogates and (optionally) the extractor are frozen.
    """

    def __init__(
        self,
        cfg: DSCAConfig,
        generator: DSCAGenerator,
        extractor: ConditionExtractorBase,
        surrogates: SurrogateEnsemble,
        device: torch.device,
        train_extractor: bool = False,
        grad_accum_steps: int = 1,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        lambda_gate: float = 0.0,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.generator = generator.to(device)
        self.extractor = extractor.to(device)
        self.surrogates = surrogates.to(device)
        self.train_extractor = train_extractor
        self.grad_accum_steps = max(1, grad_accum_steps)

        if not train_extractor:
            self.extractor.eval()
            for p in self.extractor.parameters():
                p.requires_grad_(False)

        self.loss_fn = DSCALoss(
            self.surrogates,
            lambda_id_transfer=cfg.loss.lambda_id_transfer,
            lambda_visual=cfg.loss.lambda_visual,
            lambda_diffusion=cfg.loss.lambda_diffusion,
            visual_l2_weight=cfg.loss.visual_l2_weight,
            visual_ssim_weight=cfg.loss.visual_ssim_weight,
            visual_perceptual_weight=cfg.loss.visual_perceptual_weight,
            lambda_gate=lambda_gate,
        ).to(device)

        params = [p for p in self.generator.parameters() if p.requires_grad]
        if train_extractor:
            params += [p for p in self.extractor.parameters() if p.requires_grad]
        self.optim = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

        total_steps = max(1, cfg.train.epochs)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=total_steps)

        self.use_amp = device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.ema = EMA(self.generator, decay=ema_decay) if use_ema else None

    def temperature(self, step: int) -> float:
        """Linear temperature anneal (paper Eq. 17 schedule)."""
        end = max(1, self.cfg.uscc.temperature_end)
        u = min(1.0, step / end)
        t_max, t_min = self.cfg.uscc.temperature_max, self.cfg.uscc.temperature_min
        return t_max - (t_max - t_min) * u

    def train_one_epoch(self, loader: DataLoader, epoch: int, global_step: int = 0) -> int:
        self.generator.train()
        if self.train_extractor:
            self.extractor.train()
        bar = tqdm(loader, desc=f"epoch {epoch}")
        self.optim.zero_grad(set_to_none=True)
        for i, batch in enumerate(bar):
            source = batch["source"].to(self.device, non_blocking=True)
            target = batch["target"].to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                if self.train_extractor:
                    conditions = self.extractor(target)
                else:
                    with torch.no_grad():
                        conditions = self.extractor(target)
                outputs = self.generator(
                    source, conditions, temperature=self.temperature(global_step)
                )
                losses = self.loss_fn(outputs, source, target)
                loss = losses["total"] / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (i + 1) % self.grad_accum_steps == 0:
                self.scaler.unscale_(self.optim)
                nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim.zero_grad(set_to_none=True)
                if self.ema is not None:
                    self.ema.update(self.generator)

            if global_step % self.cfg.train.log_every == 0:
                bar.set_postfix({k: f"{v.item():.4f}" for k, v in losses.items()})
            global_step += 1

        self.scheduler.step()
        return global_step

    def save_checkpoint(self, path: str | Path, step: int, epoch: int = 0) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "step": step,
            "epoch": epoch,
            "generator": self.generator.state_dict(),
            "extractor": self.extractor.state_dict(),
            "optimizer": self.optim.state_dict(),
            "lr_scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
        }
        if self.ema is not None:
            state["generator_ema"] = self.ema.state_dict()
        torch.save(state, path)

    def load_checkpoint(self, path: str | Path) -> tuple[int, int]:
        """Resume training. Returns (global_step, epoch)."""
        state = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(state["generator"])
        if "extractor" in state:
            self.extractor.load_state_dict(state["extractor"], strict=False)
        if "optimizer" in state:
            self.optim.load_state_dict(state["optimizer"])
        if "lr_scheduler" in state:
            self.scheduler.load_state_dict(state["lr_scheduler"])
        if "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])
        if self.ema is not None and "generator_ema" in state:
            self.ema.shadow.load_state_dict(state["generator_ema"])
        return int(state.get("step", 0)), int(state.get("epoch", 0))
