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
from dsca.utils.device import resolve_device
from dsca.utils.io import load_image, save_image
from _build import build_extractor, build_generator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lightweight.yaml")
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--use-ema", action="store_true", help="load EMA generator weights if present")
    parser.add_argument("--out", default="outputs/infer/camouflage.png")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.device)
    generator = build_generator(cfg)
    extractor = build_extractor(cfg)

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        gen_key = "generator_ema" if (args.use_ema and "generator_ema" in state) else "generator"
        generator.load_state_dict(state[gen_key], strict=False)
        if "extractor" in state:
            extractor.load_state_dict(state["extractor"], strict=False)
        print(f"Loaded weights from {args.checkpoint} (key='{gen_key}')")

    source = load_image(args.source, size=(cfg.image.height, cfg.image.width))
    target = load_image(args.target, size=(cfg.image.height, cfg.image.width))
    inferencer = DSCAInferencer(generator, extractor, device)
    result = inferencer.generate(
        source, target, ddim_steps=cfg.model.ddim_steps, eta=cfg.model.ddim_eta
    )
    save_image(result, args.out)
    print(f"Generated image saved to: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
