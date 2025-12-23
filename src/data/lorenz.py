"""Lorenz dataset loading helpers."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


class MeasurementDataset(Dataset):
    """Load measurement-only sequences from a pickle dataset."""

    def __init__(self, path: Path) -> None:
        with path.open("rb") as handle:
            data = pickle.load(handle)
        self.obs = [torch.tensor(item[1], dtype=torch.float32) for item in data["data"]]

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.obs[idx]


class StateObservationDataset(Dataset):
    """Load state and observation sequences (for evaluation)."""

    def __init__(self, path: Path) -> None:
        with path.open("rb") as handle:
            data = pickle.load(handle)
        self.pairs = [
            (torch.tensor(item[0], dtype=torch.float32), torch.tensor(item[1], dtype=torch.float32))
            for item in data["data"]
        ]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.pairs[idx]


class WindowedDataset(Dataset):
    """Return random fixed-length windows for TBPTT-style training."""

    def __init__(self, base: Dataset, window_length: int) -> None:
        self.base = base
        self.window_length = max(int(window_length), 0)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        seq = self.base[idx]
        seq_len = seq[0].size(0) if isinstance(seq, tuple) else seq.size(0)
        if self.window_length <= 0 or seq_len <= self.window_length:
            return seq
        start = torch.randint(0, seq_len - self.window_length + 1, ()).item()
        if isinstance(seq, tuple):
            state, obs = seq
            return state[start : start + self.window_length], obs[start : start + self.window_length]
        return seq[start : start + self.window_length]


def windowed_dataset(base: Dataset, window_length: int) -> Dataset:
    if window_length and window_length > 0:
        return WindowedDataset(base, window_length)
    return base


def collate_padded_observations(
    batch: Sequence[torch.Tensor],
    obs_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = [item.size(0) for item in batch]
    max_len = max(lengths)
    obs_batch = torch.zeros(len(batch), max_len, obs_dim)
    mask_batch = torch.ones(len(batch), max_len, dtype=torch.bool)
    for i, item in enumerate(batch):
        obs_batch[i, : item.size(0)] = item
        mask_batch[i, : item.size(0)] = False
    return obs_batch, mask_batch


def collate_padded_state_obs(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    state_dim: int,
    obs_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = [item[0].size(0) for item in batch]
    max_len = max(lengths)
    state_batch = torch.zeros(len(batch), max_len, state_dim)
    obs_batch = torch.zeros(len(batch), max_len, obs_dim)
    mask_batch = torch.ones(len(batch), max_len, dtype=torch.bool)
    for i, (state, obs) in enumerate(batch):
        state_batch[i, : state.size(0)] = state
        obs_batch[i, : obs.size(0)] = obs
        mask_batch[i, : obs.size(0)] = False
    return state_batch, obs_batch, mask_batch


def parse_noise_from_name(path: Path, key: str) -> float | None:
    """Extract a linear variance value encoded as ...{key}_{XX}dB... in the filename."""
    stem = path.stem
    token = f"{key}_"
    if token in stem:
        try:
            tag = stem.split(token)[1]
            db = float(tag.split("dB")[0])
            return 10.0 ** (-db / 10.0)
        except Exception:
            return None
    return None


__all__ = [
    "MeasurementDataset",
    "StateObservationDataset",
    "WindowedDataset",
    "windowed_dataset",
    "collate_padded_observations",
    "collate_padded_state_obs",
    "parse_noise_from_name",
]
