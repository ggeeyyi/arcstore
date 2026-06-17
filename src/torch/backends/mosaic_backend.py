"""Mosaic StreamingDataset (MDS) backend for :func:`arcstore.open_dataset`.

Ported from ``arc_toolkit.data`` mosaic backend. Registering this module
replaces the ``mds`` not-implemented placeholder with a real loader backed by
``mosaicml-streaming`` (install the ``arcstore[mosaic]`` extra). Use via
``open_dataset(uri, format="mds", ...)``.

The remote ``uri`` (S3 prefix or local MDS dir) is streamed by Mosaic itself;
shards are cached under ``mds_local`` (default: arcstore's cache dir). A unified
``decode`` maps each Mosaic sample dict to the final training sample.
"""
from __future__ import annotations

import os
from typing import Any, Callable

from ..._env import cache_dir
from ...data.registry import register_backend


def open_mds_dataset(
    path: str,
    *,
    decode: Callable[[dict], Any] | None = None,
    read_policy: str | None = None,  # noqa: ARG001 — Mosaic owns its own S3 reads
    shuffle_buffer: int = 1000,
    length: int | None = None,  # noqa: ARG001 — StreamingDataset defines its own length
    region: str | None = None,  # noqa: ARG001
    mds_local: str | None = None,
    batch_size: int | None = None,
    seed: int = 0,
    **_ignored: Any,
):
    """Build a Mosaic ``StreamingDataset`` normalized for ``open_dataset``."""
    try:
        from streaming import StreamingDataset
    except (ImportError, RuntimeError, OSError) as e:
        raise ImportError(
            "[arcstore] open_dataset(format='mds') needs mosaicml-streaming; "
            "install the arcstore[mosaic] extra."
        ) from e

    import hashlib

    seed_mixed = int.from_bytes(
        hashlib.blake2b(f"{seed},mds".encode(), digest_size=8).digest(), "little"
    ) >> 1

    class _ArcMosaicDataset(StreamingDataset):
        def __getitem__(self, at):
            obj = super().__getitem__(at)
            if not isinstance(obj, dict):
                obj = {"sample": obj}
            obj.setdefault("__key__", str(at))
            return obj if decode is None else decode(obj)

    local = mds_local or str(cache_dir("mds-cache", create=False))
    os.makedirs(local, exist_ok=True)
    return _ArcMosaicDataset(
        remote=path,
        local=local,
        shuffle=shuffle_buffer > 0,
        shuffle_seed=seed_mixed,
        batch_size=batch_size,
        predownload=max(8, (batch_size or 1) * 4),
    )


register_backend("mds", open_mds_dataset)

__all__ = ["open_mds_dataset"]
