"""Synthetic backend for :func:`arcstore.open_dataset` (``format="synthetic"``).

A storage-free compute-ceiling baseline. The ``path`` argument is ignored; pass
``format="synthetic"`` explicitly. ``length`` (or ``num_samples``) sets the
sample count; an optional unified ``decode`` maps each ``{sample, feature}``
dict to the final training sample.
"""
from __future__ import annotations

from typing import Any, Callable

from torch.utils.data import Dataset

from ...data.registry import register_backend
from ..synthetic import SyntheticDataset


class _DecodedMap(Dataset):
    def __init__(self, base: Dataset, decode: Callable[[dict], Any]):
        self._base = base
        self._decode = decode

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> Any:
        return self._decode(self._base[idx])


def open_synthetic_dataset(
    path: str = "",
    *,
    decode: Callable[[dict], Any] | None = None,
    length: int | None = None,
    num_samples: int = 10000,
    sample_shape: tuple[int, ...] = (16, 21, 60, 104),
    feature_dim: int = 4096,
    feature_len: int = 256,
    sample_key: str = "sample",
    feature_key: str = "feature",
    **_ignored: Any,
):
    """Build a :class:`SyntheticDataset` normalized for ``open_dataset``."""
    ds = SyntheticDataset(
        num_samples=length if length is not None else num_samples,
        sample_shape=sample_shape,
        feature_dim=feature_dim,
        feature_len=feature_len,
        sample_key=sample_key,
        feature_key=feature_key,
    )
    return ds if decode is None else _DecodedMap(ds, decode)


register_backend("synthetic", open_synthetic_dataset)

__all__ = ["open_synthetic_dataset"]
