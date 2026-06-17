"""Single ``.pt`` blob backend for the unified checkpoint interface.

* load -> :func:`arcstore.torch.load.load_ckpt` (stage to local NVMe + the
  mmap / weights_only fallback chain).
* save -> a thin new implementation: for an ``s3://`` destination, ``torch.save``
  to a temp local file then :func:`arcstore.upload_file` (avoids mountpoint-s3
  overwrite restrictions); for a local destination, ``torch.save`` directly.
"""
from __future__ import annotations

from typing import Any

import torch

from ...checkpoint.registry import register_checkpoint_backend
from ...uploads import open_write
from ..load import load_ckpt


def _save(path: str, *, obj: Any, **torch_save_kwargs: Any) -> None:
    # open_write handles both destinations uniformly: an s3:// path writes to
    # a temp file and uploads on clean exit; a local path is written directly
    # (parent dirs created). Either way the write never touches a mount.
    with open_write(path) as f:
        torch.save(obj, f, **torch_save_kwargs)


def _load(
    path: str,
    *,
    siblings: Any = (),
    map_location: str = "cpu",
    label: str = "ckpt",
    logger: Any = None,
) -> dict:
    return load_ckpt(
        path,
        siblings=siblings,
        map_location=map_location,
        label=label,
        logger=logger,
    )


register_checkpoint_backend("blob", save=_save, load=_load)

__all__ = ["_load", "_save"]
