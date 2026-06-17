"""Full training-state checkpointing via PyTorch Distributed Checkpoint (DCP).

Lifted from CausalVideoDiffusion ``src/utils/dcp_ckpt.py`` (itself ported
from ``arc_training_example``), rewired onto the arcstore core primitives.

* sharded **model + optimizer** tensors saved in parallel (each rank
  streams its own shard), so a resume restores Adam momentum for bit-exact
  continuation;
* path-transparent I/O — an ``s3://`` destination streams straight to S3
  via ``s3torchconnector.dcp.S3StorageWriter`` (load stages shards to local
  NVMe with s5cmd, then reads with the default ``FileSystemReader``); a
  local path uses DCP's plain ``FileSystem`` writer/reader. Saves never
  touch a FUSE mount.

``save_full_state`` / ``load_full_state`` are thin wrappers over the shared
checkpoint core in :mod:`arcstore.torch._ckpt_common`, the same core the
high-level :class:`arcstore.torch.CheckpointManager` uses. The on-disk layout
is therefore identical to the manager's: the global ``step`` plus any
``scheduler`` / ``ema`` / ``extra_state`` ride in a per-rank
``extras_rank{R}.pt`` sidecar, and completeness is signalled by the
``_ARC_COMPLETE`` marker (or DCP's own ``.metadata``).

Staging root (env, read at call time):

* ``ARCSTORE_DCP_STAGE_DIR`` — S3-load staging (default
  ``/local-ssd/arcstore/dcp_load``)
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Iterable, Mapping, Union

import torch

from .._env import env_str
from ..io import exists
from ..location import is_s3
from ..uploads import download_dir, track_future
from ._ckpt_common import (
    dcp_load_app,
    dcp_save_app,
    make_app_state,
    move_optim_state_to_param_devices,
    prime_optim_state,
    read_extras,
    read_extras_payload,
    stage_dir_for_s3,
    write_extras,
    write_marker,
)
from .runtime import barrier

logger = logging.getLogger(__name__)


def _load_stage_root() -> str:
    return env_str("ARCSTORE_DCP_STAGE_DIR", "/local-ssd/arcstore/dcp_load")


def _stage_dir_for_s3(root: str, uri: str) -> str:
    return stage_dir_for_s3(root, uri, default_basename="dcp")


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


# ---------------------------------------------------------------------------
# Python 3.13 DCP error-unmasking patch
# ---------------------------------------------------------------------------
def patch_dcp_wrap_exception_py313() -> None:
    """Unmask the real error from a multi-rank ``dcp.load`` on Python 3.13+.

    PyTorch DCP stores a live traceback in the wrapped exception it
    ``gather_object``'s across ranks; on 3.13+ each ``FrameSummary`` keeps a
    ``_code`` bytecode object that pickle refuses, so the gather dies with a
    misleading ``TypeError: cannot pickle code objects`` that hides the true
    error. We rebuild the ``StackSummary`` from plain strings so the gather
    succeeds. No-op on <3.13 and once torch carries the upstream fix.
    """
    import sys

    if sys.version_info < (3, 13):
        return
    try:
        import traceback

        from torch.distributed.checkpoint import utils as _dcp_utils
    except Exception:  # pragma: no cover - DCP layout changed
        return

    if getattr(_dcp_utils, "_arcstore_wrap_patched", False):
        return

    orig = getattr(_dcp_utils, "_wrap_exception", None)
    if orig is None:
        return

    def _safe_wrap(exc):  # pragma: no cover - only hit on 3.13 multi-rank fault
        try:
            tb = traceback.extract_tb(exc.__traceback__)
            for fs in tb:
                if hasattr(fs, "_code"):
                    fs._code = None
        except Exception:
            pass
        return orig(exc)

    _dcp_utils._wrap_exception = _safe_wrap
    _dcp_utils._arcstore_wrap_patched = True


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------
def save_full_state(
    dest: str,
    models: Union[torch.nn.Module, Iterable[torch.nn.Module]],
    optimizers,
    *,
    step: int = 0,
    scheduler=None,
    ema=None,
    extra_state: Mapping[str, Any] | None = None,
    side_files: Mapping[str, Any] | None = None,
    async_save: bool = False,
    thread_count: int = 8,
) -> None:
    """Save the complete FSDP training state (model + optimizer) to ``dest``.

    ``dest`` is an ``s3://`` prefix (streamed via ``S3StorageWriter``) or a
    local directory (DCP ``FileSystem`` writer). The global ``step`` plus the
    optional ``scheduler`` / ``ema`` / ``extra_state`` / ``side_files`` are
    written into a per-rank ``extras_rank{R}.pt`` sidecar (the same layout the
    :class:`arcstore.torch.CheckpointManager` uses). With ``async_save`` the
    tensor write runs off the critical path and its future is registered with
    :func:`arcstore.wait_for_uploads` for a shutdown flush; completeness is
    then signalled by DCP's ``.metadata`` (the synchronous path also drops the
    ``_ARC_COMPLETE`` marker).
    """
    app = {"app": make_app_state(models, optimizers)}
    t0 = time.perf_counter()

    extras: dict[str, Any] = {}
    if scheduler is not None:
        extras["scheduler"] = scheduler
    if ema is not None:
        extras["ema"] = ema
    extras.update(extra_state or {})
    extras.update(side_files or {})
    write_extras(dest, step, extras)

    fut = dcp_save_app(app, dest, async_save=async_save, thread_count=thread_count)
    if fut is not None:
        track_future(fut)
        logger.info("[arcstore-dcp] async save started -> %s", dest)
        return
    write_marker(dest)
    logger.info(
        "[arcstore-dcp] saved -> %s in %.2fs", dest, time.perf_counter() - t0
    )


def load_full_state(
    src: str,
    models: Union[torch.nn.Module, Iterable[torch.nn.Module]],
    optimizers,
    *,
    scheduler=None,
    ema=None,
    return_meta: bool = False,
) -> int | dict[str, Any]:
    """Restore the complete FSDP training state from ``src``. Returns global step.

    ``src`` may be an ``s3://`` prefix (shards staged to local NVMe with
    s5cmd, once per node, then read via the default ``FileSystemReader``) or
    a local directory. DCP does small random reads, so even for a mounted
    bucket the NVMe staging path is kept — it is strictly faster than FUSE
    and already barrier-coordinated per node.
    """
    prime_optim_state(models, optimizers)

    app = {"app": make_app_state(models, optimizers)}
    t0 = time.perf_counter()

    if not is_s3(src):
        dcp_load_app(app, src)
        load_dir = src
    else:
        load_dir = _stage_dir_for_s3(_load_stage_root(), src)
        if _local_rank() == 0:
            download_dir(src, load_dir, workers=256, required_files=(".metadata",))
        barrier()
        dcp_load_app(app, load_dir)

    move_optim_state_to_param_devices(optimizers)

    restore: dict[str, Any] = {}
    if scheduler is not None:
        restore["scheduler"] = scheduler
    if ema is not None:
        restore["ema"] = ema
    step = read_extras(load_dir, restore)
    logger.info(
        "[arcstore-dcp] loaded <- %s in %.2fs (step=%d)",
        src,
        time.perf_counter() - t0,
        step,
    )
    if return_meta:
        saved = read_extras_payload(load_dir).get("extras", {})
        return {
            "step": int(step),
            "scheduler": saved.get("scheduler"),
            "extra_state": {k: v for k, v in saved.items() if k not in ("scheduler", "ema")},
        }
    return int(step)


def dcp_dir_exists(path: str) -> bool:
    """True if ``path`` looks like a populated DCP checkpoint dir (local or s3)."""
    if is_s3(path):
        return exists(path.rstrip("/") + "/.metadata")
    # Local: DCP writes a ``.metadata`` file at the root.
    return os.path.isdir(path) and os.path.exists(os.path.join(path, ".metadata"))


__all__ = [
    "dcp_dir_exists",
    "load_full_state",
    "patch_dcp_wrap_exception_py313",
    "prime_optim_state",
    "save_full_state",
]
