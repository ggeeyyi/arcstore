"""Unified dataset reading layer.

``arcstore.open_dataset(path, ...)`` is the single entry point: it classifies
a dataset path (local dir / FUSE-mounted S3 / direct ``s3://``), dispatches to
the right backend, and returns a ``torch.utils.data.IterableDataset`` of
decoded samples.

The path classification lives in :func:`resolve_dataset_access`; the backend
dispatch + registry live in :mod:`arcstore.data.registry`. Concrete torch
backends (scatter, wds) register themselves from
:mod:`arcstore.torch.backends`.
"""
from __future__ import annotations

from .access import (
    DIRECT_S3,
    LOCAL,
    MOUNT,
    DatasetAccess,
    resolve_dataset_access,
)
from .loader import build_dataloader
from .registry import (
    available_backends,
    open_dataset,
    register_backend,
)

__all__ = [
    "DIRECT_S3",
    "LOCAL",
    "MOUNT",
    "DatasetAccess",
    "available_backends",
    "build_dataloader",
    "open_dataset",
    "register_backend",
    "resolve_dataset_access",
]
