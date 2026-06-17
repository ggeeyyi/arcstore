"""Accelerate full-state backend for the unified checkpoint interface.

Adapts :func:`arcstore.torch.accelerate_ckpt.save_accelerate_state` /
``load_accelerate_state`` onto ``(path, **kwargs)``. ``accelerator`` is
required. ``path`` is the durable destination/source:

* save with an ``s3://`` ``path`` -> upload from ``local_dir`` (required) to
  ``s3_uri=path``;
* save with a local ``path`` -> ``save_state(path)`` only (``s3_uri=None``);
* load -> ``load_accelerate_state(accelerator, path, ...)``.

Registered for the ``accelerate`` kind. Raw DeepSpeed engines go through
:class:`arcstore.torch.CheckpointManager` instead.
"""
from __future__ import annotations

from typing import Any

from ...checkpoint.registry import register_checkpoint_backend
from ...location import is_s3
from ..accelerate_ckpt import load_accelerate_state, save_accelerate_state


def _save(
    path: str,
    *,
    accelerator: Any,
    local_dir: str | None = None,
    ema: Any = None,
    async_upload: bool = False,
    keep_local: bool = False,
    upload_workers: int | None = None,
) -> None:
    if is_s3(path):
        if local_dir is None:
            raise ValueError(
                "[arcstore] save_checkpoint(kind='accelerate') to an s3:// path "
                "requires local_dir= (the node-local shard dir to upload from)."
            )
        return save_accelerate_state(
            accelerator,
            local_dir,
            path,
            ema=ema,
            async_upload=async_upload,
            keep_local=keep_local,
            upload_workers=upload_workers,
        )
    return save_accelerate_state(
        accelerator,
        path,
        None,
        ema=ema,
        async_upload=async_upload,
        keep_local=keep_local,
        upload_workers=upload_workers,
    )


def _load(
    path: str,
    *,
    accelerator: Any,
    local_dir: str | None = None,
    ema: Any = None,
    download_workers: int | None = None,
    required_files: Any = None,
) -> int:
    return load_accelerate_state(
        accelerator,
        path,
        local_dir=local_dir,
        ema=ema,
        download_workers=download_workers,
        required_files=required_files,
    )


register_checkpoint_backend("accelerate", save=_save, load=_load)

__all__ = ["_load", "_save"]
