"""Synthetic random-tensor dataset for compute-ceiling baselines (no storage).

Ported from ``arc_toolkit.data.SyntheticDataset``. Useful to measure the pure
training throughput an accelerator can sustain with the data pipeline removed —
compare against a real backend to quantify IO stall.
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

__all__ = ["SyntheticDataset"]


class SyntheticDataset(Dataset):
    """Random tensors generated deterministically per index (no storage)."""

    def __init__(
        self,
        num_samples: int = 10000,
        *,
        sample_shape: tuple[int, ...] = (16, 21, 60, 104),
        feature_dim: int = 4096,
        feature_len: int = 256,
        sample_key: str = "sample",
        feature_key: str = "feature",
    ):
        self.num_samples = num_samples
        self.sample_shape = sample_shape
        self.feature_dim = feature_dim
        self.feature_len = feature_len
        self.sample_key = sample_key
        self.feature_key = feature_key

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        gen = torch.Generator().manual_seed(idx)
        return {
            self.sample_key: torch.randn(*self.sample_shape, generator=gen),
            self.feature_key: torch.randn(self.feature_len, self.feature_dim, generator=gen),
        }
