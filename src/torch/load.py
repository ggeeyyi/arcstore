"""Staged ``torch.load`` with an mmap/weights_only fallback chain.

Lifted from CausalVideoDiffusion ``src/utils/ckpt_cache.py::load_ckpt``;
the checkpoint sibling set is now a parameter passed through to
:func:`arcstore.staging.stage_to_local`.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional, Sequence

import torch

from ..staging import stage_to_local

_module_logger = logging.getLogger(__name__)


def load_ckpt(
    path: str,
    *,
    siblings: Sequence[str] = (),
    map_location: str = "cpu",
    label: str = "ckpt",
    logger=None,
) -> dict:
    """Stage to local NVMe (if remote) and ``torch.load`` with mmap fallback.

    - ``mmap=True`` so tensor storages are demand-paged from the file
      rather than copied into Python memory up-front.
    - ``weights_only=True`` skips arbitrary pickle execution; falls back to
      ``weights_only=False`` for checkpoints with non-tensor objects, then
      to a non-mmap full read.

    Callers that need their own ``torch.load`` flags should call
    :func:`arcstore.stage_to_local` directly.
    """
    log = logger if logger is not None else _module_logger

    path = stage_to_local(path, siblings=siblings, logger=log)

    file_size_mb: Optional[float] = None
    try:
        file_size_mb = os.path.getsize(path) / (1024 * 1024)
    except OSError:
        pass

    size_str = f" ({file_size_mb:.1f} MiB)" if file_size_mb is not None else ""
    log.info(f"[arcstore] loading {label} from {path}{size_str}")
    t0 = time.perf_counter()

    try:
        blob = torch.load(path, map_location=map_location, mmap=True, weights_only=True)
        mode = "mmap+weights_only"
    except Exception as e_safe:  # noqa: BLE001
        log.warning(
            f"[arcstore] weights_only=True load failed ({type(e_safe).__name__}: "
            f"{e_safe}); retrying without it."
        )
        try:
            blob = torch.load(
                path, map_location=map_location, mmap=True, weights_only=False
            )
            mode = "mmap"
        except Exception as e_mmap:  # noqa: BLE001
            log.warning(
                f"[arcstore] mmap=True load failed ({type(e_mmap).__name__}: "
                f"{e_mmap}); retrying with full read."
            )
            blob = torch.load(path, map_location=map_location, weights_only=False)
            mode = "full-read"

    dt = time.perf_counter() - t0
    rate = (
        f", {file_size_mb / dt:.1f} MiB/s"
        if file_size_mb is not None and dt > 0
        else ""
    )
    log.info(f"[arcstore] {label} loaded in {dt:.2f}s [{mode}]{rate}")
    return blob
