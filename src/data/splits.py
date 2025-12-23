"""Train/val/test split utilities."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch


def obtain_tr_val_test_idx(
    dataset: Sequence[Iterable],
    tr_to_test_split: float,
    tr_to_val_split: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    generator = torch.Generator().manual_seed(seed)
    total_len = len(dataset)
    permutation = torch.randperm(total_len, generator=generator).tolist()

    cutoff_tt = int(tr_to_test_split * total_len)
    train_val_idx = permutation[:cutoff_tt]
    test_idx = permutation[cutoff_tt:]

    cutoff_tv = int(tr_to_val_split * len(train_val_idx))
    train_idx = train_val_idx[:cutoff_tv]
    val_idx = train_val_idx[cutoff_tv:]
    return train_idx, val_idx, test_idx


def create_splits_file_name(train_path: Path, splits_name: str) -> Path:
    if splits_name:
        candidate = Path(splits_name)
        return candidate if candidate.is_absolute() else train_path.parent / candidate
    return train_path.parent / f"{train_path.stem}_split.pkl"


def load_splits_file(path: Path) -> dict:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def save_splits_file(path: Path, splits: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(splits, handle, protocol=pickle.HIGHEST_PROTOCOL)


__all__ = [
    "obtain_tr_val_test_idx",
    "create_splits_file_name",
    "load_splits_file",
    "save_splits_file",
]
