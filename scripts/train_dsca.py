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

from torch.utils.data import DataLoader

from dsca.config import load_config
from dsca.data.reid_dataset import ReIDImageDataset
from dsca.data.pair_sampler import SourceTargetPairDataset
from dsca.trainer import DSCATrainer
from dsca.utils.device import resolve_device
from dsca.utils.seed import seed_everything
from _build import build_extractor, build_generator, build_surrogates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default="", help="checkpoint path to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)

    if not cfg.data.train_dir:
        raise ValueError("Please set data.train_dir in the config file.")
    base = ReIDImageDataset(cfg.data.train_dir, image_size=(cfg.image.height, cfg.image.width))
    pairs = SourceTargetPairDataset(base)
    loader = DataLoader(
        pairs,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    trainer = DSCATrainer(
        cfg,
        build_generator(cfg),
        build_extractor(cfg),
        build_surrogates(cfg),
        device,
        train_extractor=cfg.train.train_extractor,
        grad_accum_steps=cfg.train.grad_accum_steps,
        use_ema=cfg.train.use_ema,
        ema_decay=cfg.train.ema_decay,
        lambda_gate=cfg.loss.lambda_gate,
    )

    step, start_epoch = 0, 0
    resume_path = args.resume or cfg.train.resume
    if resume_path and Path(resume_path).exists():
        step, start_epoch = trainer.load_checkpoint(resume_path)
        print(f"Resumed from {resume_path} at epoch {start_epoch}, step {step}")

    ckpt_dir = Path(cfg.train.checkpoint_dir)
    for epoch in range(start_epoch + 1, cfg.train.epochs + 1):
        step = trainer.train_one_epoch(loader, epoch=epoch, global_step=step)
        trainer.save_checkpoint(ckpt_dir / f"dsca_epoch_{epoch:03d}.pt", step, epoch)
        trainer.save_checkpoint(ckpt_dir / "dsca_last.pt", step, epoch)


if __name__ == "__main__":
    main()
