"""Range-only NPZ dataset utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class MeasurementDataset(Dataset):
    """Load range-only measurements from NPZ produced by data_generator."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        payload = np.load(path, allow_pickle=True)
        self.observations = torch.as_tensor(payload["Y"], dtype=torch.float32)  # (N, T, obs_dim)
        self.anchors = torch.as_tensor(payload["anchors"], dtype=torch.float32)  # (2, M)
        self.dt = float(payload["dt"])
        self.qc = float(payload["qc"])

    def __len__(self) -> int:
        return self.observations.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.observations[idx]


@dataclass
class WindowConfig:
    length: int
    stride: int | None = None


def windowed_dataset(dataset: MeasurementDataset, window_length: int, stride: int | None = None) -> List[torch.Tensor]:
    """Split each trajectory into non-overlapping windows (or with given stride)."""
    stride = stride if stride is not None else window_length
    windows: List[torch.Tensor] = []
    for seq in dataset:
        T = seq.size(0)
        for start in range(0, T - window_length + 1, stride):
            windows.append(seq[start : start + window_length])
    return windows if windows else [seq for seq in dataset]


def collate_padded_observations(batch: Sequence[torch.Tensor], obs_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length observation sequences and return mask."""
    lengths = [b.size(0) for b in batch]
    max_len = max(lengths)
    B = len(batch)
    padded = torch.zeros(B, max_len, obs_dim, dtype=batch[0].dtype)
    mask = torch.ones(B, max_len, dtype=torch.bool)
    for i, seq in enumerate(batch):
        L = seq.size(0)
        padded[i, :L] = seq
        mask[i, :L] = False  # False means valid
    return padded, mask


__all__ = ["MeasurementDataset", "collate_padded_observations", "windowed_dataset"]
