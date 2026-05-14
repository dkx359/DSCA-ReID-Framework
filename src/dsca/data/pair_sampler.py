from __future__ import annotations

import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset

from dsca.data.reid_dataset import ReIDImageDataset


class SourceTargetPairDataset(Dataset):
    """Create source-target pairs with different identities for DSCA training."""

    def __init__(self, base: ReIDImageDataset) -> None:
        self.base = base
        self.pid_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, item in enumerate(base.items):
            self.pid_to_indices[item.pid].append(idx)
        self.pids = sorted(self.pid_to_indices)
        if len(self.pids) < 2:
            raise RuntimeError("At least two identities are required for source-target pairing.")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        source = self.base[index]
        source_pid = int(source["pid"])
        target_pid = random.choice([pid for pid in self.pids if pid != source_pid])
        target_idx = random.choice(self.pid_to_indices[target_pid])
        target = self.base[target_idx]
        return {
            "source": source["image"],
            "target": target["image"],
            "source_pid": source_pid,
            "target_pid": target_pid,
            "source_path": source["path"],
            "target_path": target["path"],
        }
