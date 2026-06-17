"""WebDataset backend for :func:`arcstore.open_dataset`.

Reuses :func:`arcstore.torch.wds.build_wds_dataset` (which already handles
shard expansion, ``pipe:s5cmd cat`` vs mount/local paths, and
``split_by_node`` / ``split_by_worker`` sharding). The unified ``decode`` is
wired in as the pipeline's ``sample_map`` so it runs on each
``tarfile_to_samples`` dict; the unified ``shuffle_buffer`` maps to
sample-level shuffle. WebDataset pipelines are already ``IterableDataset``s,
so this backend does not use ``_DecodedView``.
"""
from __future__ import annotations

from typing import Any, Callable

from ...data.registry import register_backend
from ..wds import build_wds_dataset


def open_wds_dataset(
    path: str,
    *,
    decode: Callable[[dict], Any] | None = None,
    read_policy: str | None = None,
    shuffle_buffer: int = 1000,
    length: int | None = None,
    region: str | None = None,  # noqa: ARG001 — wds reads shards by URL, no region
    shuffle_shards: bool = True,
    shard_shuffle: int = 100,
    sample_shuffle_initial: int | None = None,
    **_ignored: Any,
):
    """Build a WebDataset ``DataPipeline`` normalized for ``open_dataset``."""
    pipeline = build_wds_dataset(
        path,
        read_policy=read_policy,
        shuffle_shards=shuffle_shards,
        shard_shuffle=shard_shuffle,
        sample_shuffle=shuffle_buffer or 0,
        sample_shuffle_initial=sample_shuffle_initial,
        sample_map=decode,
    )
    if length is not None and hasattr(pipeline, "with_length"):
        return pipeline.with_length(length)
    return pipeline


register_backend("wds", open_wds_dataset)

__all__ = ["open_wds_dataset"]
