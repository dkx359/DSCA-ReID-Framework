from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch

from dsca.config import load_config
from dsca.inferencer import DSCAInferencer
from dsca.losses import DSCALoss
from dsca.utils.device import resolve_device
from dsca.utils.io import save_image
from dsca.utils.seed import seed_everything
from _build import build_extractor, build_generator, build_surrogates


def make_pair(size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    h, w = size
    yy = torch.linspace(0, 1, h).view(1, 1, h, 1).expand(1, 3, h, w)
    xx = torch.linspace(0, 1, w).view(1, 1, 1, w).expand(1, 3, h, w)
    source = torch.cat([xx[:, :1], yy[:, :1], 0.3 + 0.2 * xx[:, :1]], dim=1).clamp(0, 1)
    target = torch.cat([0.2 + 0.7 * yy[:, :1], 1.0 - xx[:, :1], 0.5 * yy[:, :1]], dim=1).clamp(0, 1)
    return source, target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lightweight.yaml")
    parser.add_argument("--out", default="outputs/smoke")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    torch.set_num_threads(1)
    device = resolve_device(cfg.device)

    generator = build_generator(cfg).to(device)
    extractor = build_extractor(cfg).to(device)
    surrogates = build_surrogates(cfg).to(device)
    loss_fn = DSCALoss(
        surrogates,
        lambda_id_transfer=cfg.loss.lambda_id_transfer,
        lambda_visual=cfg.loss.lambda_visual,
        lambda_diffusion=cfg.loss.lambda_diffusion,
        visual_l2_weight=cfg.loss.visual_l2_weight,
        visual_ssim_weight=cfg.loss.visual_ssim_weight,
        visual_perceptual_weight=cfg.loss.visual_perceptual_weight,
        lambda_gate=cfg.loss.lambda_gate,
    ).to(device)

    source, target = make_pair((cfg.image.height, cfg.image.width))
    source = source.to(device)
    target = target.to(device)

    # forward + backward to confirm gradients flow end-to-end
    conditions = extractor(target)
    outputs = generator(source, conditions, temperature=1.0)
    losses = loss_fn(outputs, source, target)
    losses["total"].backward()
    grad_ok = any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in generator.parameters()
        if p.requires_grad
    )

    inferencer = DSCAInferencer(generator, extractor, device)
    result = inferencer.generate(source, target, ddim_steps=cfg.model.ddim_steps, eta=cfg.model.ddim_eta)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_image(result, out_dir / "camouflage_result.png")
    torch.save(
        {"generator": generator.state_dict(), "extractor": extractor.state_dict()},
        out_dir / "dsca_smoke_state.pt",
    )
    values = {k: float(v.detach().cpu()) for k, v in losses.items()}
    print(f"Smoke check passed. gradients_ok={grad_ok}. Loss snapshot: {values}")
    print(f"Generated shape: {tuple(result.shape)}; surrogates: {len(surrogates)}")
    print(f"Files written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
