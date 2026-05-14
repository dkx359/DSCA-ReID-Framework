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
from torch.utils.data import DataLoader

from dsca.config import load_config
from dsca.data.pair_sampler import SourceTargetPairDataset
from dsca.data.reid_dataset import ReIDImageDataset
from dsca.inferencer import DSCAInferencer
from dsca.metrics import (
    attack_success_rate,
    cmc_and_map,
    psnr,
    ssim,
    surrogate_victim_discrepancy,
)
from dsca.models.surrogates import build_surrogate_ensemble
from dsca.utils.device import resolve_device
from dsca.utils.seed import seed_everything
from _build import build_extractor, build_generator, build_surrogates


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate DSCA: targeted ASR, mAP/Rank-10 degradation, perceptual "
        "fidelity, and the surrogate-victim transferability bound (paper Sec. IV)."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=512)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)

    if not cfg.data.train_dir:
        raise ValueError("Set data.train_dir (used as the evaluation image pool).")
    base = ReIDImageDataset(cfg.data.train_dir, image_size=(cfg.image.height, cfg.image.width))
    pairs = SourceTargetPairDataset(base)
    loader = DataLoader(pairs, batch_size=min(32, cfg.train.batch_size), shuffle=False, num_workers=cfg.train.num_workers)

    generator = build_generator(cfg)
    extractor = build_extractor(cfg)
    state = torch.load(args.checkpoint, map_location="cpu")
    gen_key = "generator_ema" if (args.use_ema and "generator_ema" in state) else "generator"
    generator.load_state_dict(state[gen_key], strict=False)
    if "extractor" in state:
        extractor.load_state_dict(state["extractor"], strict=False)
    inferencer = DSCAInferencer(generator, extractor, device)

    # Surrogate ensemble (used for training) and an *unseen* victim model.
    surrogates = build_surrogates(cfg).to(device).eval()
    victim = build_surrogate_ensemble(
        embed_dim=cfg.surrogate.embed_dim, num_cnn=0, num_transformer=1
    ).models[0].to(device).eval()

    cam_feats, src_feats, tgt_feats = [], [], []
    target_pids, source_pids = [], []
    psnr_vals, ssim_vals = [], []
    disc_mean, disc_count = 0.0, 0

    seen = 0
    for batch in loader:
        if seen >= args.max_pairs:
            break
        source = batch["source"].to(device)
        target = batch["target"].to(device)
        camouflage = inferencer.generate(
            source, target, ddim_steps=cfg.model.ddim_steps, eta=cfg.model.ddim_eta
        )

        cam_feats.append(victim(camouflage).cpu())
        src_feats.append(victim(source).cpu())
        tgt_feats.append(victim(target).cpu())
        target_pids.append(batch["target_pid"])
        source_pids.append(batch["source_pid"])

        psnr_vals.append(psnr(camouflage, source).item())
        ssim_vals.append(ssim(camouflage, source).item())

        d = surrogate_victim_discrepancy(surrogates, victim, source)
        disc_mean += d["delta_mean"]
        disc_count += 1
        seen += source.shape[0]

    cam_feats = torch.cat(cam_feats)
    src_feats = torch.cat(src_feats)
    tgt_feats = torch.cat(tgt_feats)
    target_pids = torch.cat(target_pids)
    source_pids = torch.cat(source_pids)

    # gallery = target references; ASR = camouflage queries matching target identity
    asr = attack_success_rate(cam_feats, target_pids, tgt_feats, target_pids)
    # destructiveness: how much the camouflage degrades retrieval of the *source* id
    cmc_clean, map_clean = cmc_and_map(src_feats, source_pids, src_feats, source_pids)
    cmc_attack, map_attack = cmc_and_map(cam_feats, source_pids, src_feats, source_pids)

    print("=" * 60)
    print(f"Evaluated pairs              : {seen}")
    print(f"Targeted ASR (Rank-1)        : {asr * 100:.2f}%")
    print(f"Rank-10 (clean -> attack)    : {cmc_clean[-1] * 100:.2f}% -> {cmc_attack[-1] * 100:.2f}%")
    print(f"mAP     (clean -> attack)    : {map_clean * 100:.2f}% -> {map_attack * 100:.2f}%")
    print(f"PSNR / SSIM (camouflage,src) : {sum(psnr_vals) / len(psnr_vals):.2f} dB / "
          f"{sum(ssim_vals) / len(ssim_vals):.4f}")
    print(f"Surrogate-victim delta_mean  : {disc_mean / max(1, disc_count):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
