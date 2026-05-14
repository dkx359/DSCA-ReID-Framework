from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    x = tensor.detach().cpu().clamp(0, 1)
    if x.dim() == 4:
        x = x[0]
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def load_image(path: str | Path, size: tuple[int, int]) -> torch.Tensor:
    """Load an RGB image as Tensor[1, 3, H, W] in [0, 1]."""
    image = Image.open(path).convert("RGB")
    image = image.resize((size[1], size[0]), Image.BICUBIC)
    return _pil_to_tensor(image).unsqueeze(0)


def save_image(tensor: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _tensor_to_pil(tensor).save(path)
