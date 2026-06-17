"""``build_dataloader``: a DataLoader over :func:`open_dataset` with training defaults.

Ported from ``arc_toolkit.data.build_dataloader`` and rewired onto arcstore's
unified :func:`arcstore.open_dataset`. Convenience only — callers who need full
control can construct the ``DataLoader`` themselves around ``open_dataset``.

torch is imported lazily inside the function so importing this module stays
torch-free (the core ``arcstore`` namespace must not require the torch extra).
"""
from __future__ import annotations

from typing import Any, Callable

from .registry import open_dataset

__all__ = ["build_dataloader"]


def build_dataloader(
    path: str,
    *,
    format: str | None = None,
    decode: Callable[[dict], Any] | None = None,
    batch_size: int = 1,
    num_workers: int = 4,
    collate_fn: Callable | None = None,
    prefetch_factor: int = 2,
    read_policy: str | None = None,
    shuffle_buffer: int = 1000,
    length: int | None = None,
    region: str | None = None,
    **backend_kwargs: Any,
):
    """Build a ``torch.utils.data.DataLoader`` over :func:`open_dataset`.

    Iterable datasets (scatter / wds / mds / S3 streams) shard internally, so
    the loader uses ``shuffle=False`` / ``drop_last=False`` for them and
    ``shuffle=True`` / ``drop_last=True`` for map-style datasets (synthetic).
    ``pin_memory`` follows CUDA availability; ``persistent_workers`` is enabled
    whenever ``num_workers > 0``.
    """
    import torch
    from torch.utils.data import DataLoader, IterableDataset

    dataset = open_dataset(
        path,
        format=format,
        decode=decode,
        read_policy=read_policy,
        shuffle_buffer=shuffle_buffer,
        length=length,
        region=region,
        **backend_kwargs,
    )
    is_iterable = isinstance(dataset, IterableDataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False if is_iterable else True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False if is_iterable else True,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
