"""Scatter ``.pt`` backend for :func:`arcstore.open_dataset`.

Reuses :class:`arcstore.torch.scatter.ScatterPtDataset` for the actual
local/mount/direct-S3 reading and rank/worker sharding, wrapping each raw
object's bytes into a WebDataset-style ``{"pt": bytes}`` sample. A unified
``decode`` (plus sample-level shuffle and optional length) is applied by
:class:`arcstore.data.view._DecodedView`.
"""
from __future__ import annotations

from typing import Any, Callable

from ...data.registry import register_backend
from ...data.view import _DecodedView
from ..scatter import ScatterPtDataset


def _wrap_bytes(raw: bytes) -> dict:
    return {"pt": raw}


def open_scatter_dataset(
    path: str,
    *,
    decode: Callable[[dict], Any] | None = None,
    read_policy: str | None = None,
    shuffle_buffer: int = 1000,
    length: int | None = None,
    region: str | None = None,
    transform: Callable[[bytes], Any] | None = None,
    use_mount: bool | None = None,
    **_ignored: Any,
):
    """Build a scatter ``.pt`` dataset normalized for ``open_dataset``.

    ``transform`` is accepted for backward compatibility with the old
    ``ScatterPtDataset(transform=...)`` single-arg API: when given (and no
    ``decode``), it is bridged as ``decode = lambda s: transform(s["pt"])``.
    """
    if decode is None and transform is not None:
        decode = lambda sample: transform(sample["pt"])  # noqa: E731

    # Disable ScatterPtDataset's internal reservoir shuffle (buffer<=1 is a
    # passthrough); the unified shuffle happens in _DecodedView so it composes
    # identically across backends.
    raw_ds = ScatterPtDataset(
        path,
        transform=_wrap_bytes,
        region=region,
        shuffle_buffer=1,
        use_mount=use_mount,
        read_policy=read_policy,
    )
    return _DecodedView(
        raw_ds,
        decode=decode,
        shuffle_buffer=shuffle_buffer,
        length=length,
    )


register_backend("scatter", open_scatter_dataset)

__all__ = ["open_scatter_dataset"]
