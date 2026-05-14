from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_PID_RE = re.compile(r"^(-?\d+)_")


@dataclass(frozen=True)
class ReIDItem:
    path: Path
    pid: int


def parse_market_pid(path: Path) -> int:
    match = _PID_RE.match(path.name)
    if match is None:
        raise ValueError(f"Cannot parse person id from file name: {path.name}")
    return int(match.group(1))


def pil_to_tensor(image: Image.Image, image_size: tuple[int, int]) -> torch.Tensor:
    image = image.resize((image_size[1], image_size[0]), Image.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class ReIDImageDataset(Dataset):
    def __init__(self, root: str | Path, image_size: tuple[int, int]) -> None:
        self.root = Path(root)
        self.image_size = image_size
        if not self.root.exists():
            raise FileNotFoundError(f"Training image directory does not exist: {self.root}")
        self.items = [
            ReIDItem(p, parse_market_pid(p))
            for p in sorted(self.root.rglob("*"))
            if p.suffix.lower() in _IMAGE_EXTS and not p.name.startswith("-1")
        ]
        if not self.items:
            raise RuntimeError(f"No images found under {self.root}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        item = self.items[index]
        image = Image.open(item.path).convert("RGB")
        return {"image": pil_to_tensor(image, self.image_size), "pid": item.pid, "path": str(item.path)}
