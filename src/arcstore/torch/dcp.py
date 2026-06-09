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

Non-tensor state (the global ``step``) is kept OUT of the DCP state dict
and persisted as a tiny rank-0 ``train_meta.pt`` side-file, because folding
non-tensor payloads into the gathered multi-rank Save/Load plan can make it
unpicklable.

Staging roots (env, read at call time):

* ``ARCSTORE_DCP_STAGE_DIR`` — S3-load staging (default
  ``/local-ssd/arcstore/dcp_load``)
* save fallback (no s3torchconnector): ``/tmp/arcstore/dcp_save``
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Iterable, List, Union

import torch
import torch.distributed as dist

from .._env import aws_region, env_str
from ..location import is_s3, split_s3
from ..uploads import download_dir, track_future, upload_dir, upload_file

logger = logging.getLogger(__name__)

_DCP_SAVE_STAGE = "/tmp/arcstore/dcp_save"


def _load_stage_root() -> str:
    return env_str("ARCSTORE_DCP_STAGE_DIR", "/local-ssd/arcstore/dcp_load")


def _as_list(x) -> List:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


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
# App state (sharded model + optimizer tensors only)
# ---------------------------------------------------------------------------
def _make_app_state(models, optimizers):
    """A DCP ``Stateful`` bundling ONLY sharded model + optimizer *tensors*.

    ``get_state_dict`` / ``set_state_dict`` normalize FSDP sharding and FQNs
    so the checkpoint reshards across topologies. ``models`` / ``optimizers``
    may be a single object or a list.
    """
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
    from torch.distributed.checkpoint.stateful import Stateful

    ms = _as_list(models)
    os_ = _as_list(optimizers)

    if len(ms) != len(os_):
        raise ValueError(
            f"DCP full-state checkpoint expects one optimizer per model; "
            f"got {len(ms)} model(s) and {len(os_)} optimizer(s)."
        )

    class _AppState(Stateful):
        def state_dict(self):
            if len(ms) == 1:
                msd, osd = get_state_dict(ms[0], os_[0])
                return {"model": msd, "optim": osd}

            model_state, optim_state = {}, {}
            for idx, (model, optimizer) in enumerate(zip(ms, os_)):
                msd, osd = get_state_dict(model, optimizer)
                key = str(idx)
                model_state[key] = msd
                optim_state[key] = osd
            return {"model": model_state, "optim": optim_state}

        def load_state_dict(self, state_dict):
            if len(ms) == 1:
                set_state_dict(
                    ms[0],
                    os_[0],
                    model_state_dict=state_dict["model"],
                    optim_state_dict=state_dict["optim"],
                )
                return

            for idx, (model, optimizer) in enumerate(zip(ms, os_)):
                key = str(idx)
                set_state_dict(
                    model,
                    optimizer,
                    model_state_dict=state_dict["model"][key],
                    optim_state_dict=state_dict["optim"][key],
                )

    return _AppState()


def _optim_has_tensor_state(optimizer) -> bool:
    for group in optimizer.param_groups:
        for p in group["params"]:
            st = optimizer.state.get(p)
            if st and any(isinstance(v, torch.Tensor) for v in st.values()):
                return True
    return False


def prime_optim_state(models, optimizers) -> None:
    """Allocate per-param optimizer buffers before a multi-rank ``dcp.load``.

    A never-stepped optimizer only exposes ``param_groups`` (BYTES) in the
    flatten plan; a checkpoint saved after training holds sharded momentum
    tensors instead, so the strict planner would raise ``Missing key:
    app.optim.param_groups`` on non-zero ranks. One dummy zero-loss backward
    + ``step()`` materializes Adam ``exp_avg`` / ``exp_avg_sq`` so the plan
    structure matches; the subsequent load overwrites the values.
    """
    ms = _as_list(models)
    os_ = _as_list(optimizers)
    if os_ and all(_optim_has_tensor_state(o) for o in os_):
        return

    for m in ms:
        m.train()
    loss = None
    for m in ms:
        for p in m.parameters():
            if p.requires_grad:
                term = p.sum() * 0.0
                loss = term if loss is None else loss + term
    if loss is None:
        return
    loss.backward()
    for o in os_:
        o.step()
        o.zero_grad(set_to_none=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if not all(_optim_has_tensor_state(o) for o in os_):
        raise RuntimeError(
            "[arcstore-dcp] failed to materialize optimizer state before DCP load"
        )


# ---------------------------------------------------------------------------
# step side-file
# ---------------------------------------------------------------------------
def _save_train_meta(dest: str, step: int) -> None:
    meta = {"step": int(step)}
    if is_s3(dest):
        local = tempfile.mktemp(suffix=".pt")
        torch.save(meta, local)
        upload_file(local, dest.rstrip("/") + "/train_meta.pt")
        os.remove(local)
    else:
        Path(dest).mkdir(parents=True, exist_ok=True)
        torch.save(meta, str(Path(dest) / "train_meta.pt"))


def _load_train_meta(local_dir: str) -> int:
    p = Path(local_dir) / "train_meta.pt"
    if not p.exists():
        logger.warning("[arcstore-dcp] no train_meta.pt at %s; step=0", local_dir)
        return 0
    meta = torch.load(str(p), map_location="cpu", weights_only=False)
    return int(meta.get("step", 0))


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------
def save_full_state(
    dest: str,
    models: Union[torch.nn.Module, Iterable[torch.nn.Module]],
    optimizers,
    *,
    step: int = 0,
    async_save: bool = False,
    thread_count: int = 8,
) -> None:
    """Save the complete FSDP training state (model + optimizer) to ``dest``.

    ``dest`` is an ``s3://`` prefix (streamed via ``S3StorageWriter``) or a
    local directory (DCP ``FileSystem`` writer). The global ``step`` is
    written as a rank-0 ``train_meta.pt`` side-file. With ``async_save`` the
    tensor write runs off the critical path and its future is registered
    with :func:`arcstore.wait_for_uploads` for a shutdown flush.
    """
    import torch.distributed.checkpoint as dcp

    app = {"app": _make_app_state(models, optimizers)}
    t0 = time.perf_counter()

    if _rank() == 0:
        _save_train_meta(dest, step)

    # Local destination: plain DCP FileSystem write.
    if not is_s3(dest):
        Path(dest).mkdir(parents=True, exist_ok=True)
        if async_save:
            try:
                track_future(dcp.async_save(app, checkpoint_id=dest))
                logger.info("[arcstore-dcp] async save started -> %s (local)", dest)
                return
            except (AssertionError, RuntimeError) as e:
                logger.warning(
                    "[arcstore-dcp] async_save unavailable (%s); sync fallback.",
                    str(e)[:80],
                )
        dcp.save(app, checkpoint_id=dest)
        logger.info(
            "[arcstore-dcp] saved -> %s (local) in %.2fs",
            dest,
            time.perf_counter() - t0,
        )
        return

    # S3 destination: stream straight to S3 via s3torchconnector.
    try:
        from s3torchconnector.dcp import S3StorageWriter

        writer = S3StorageWriter(
            region=aws_region(), path=dest, thread_count=thread_count
        )
        if async_save:
            try:
                track_future(dcp.async_save(app, storage_writer=writer))
                logger.info("[arcstore-dcp] async save started -> %s", dest)
                return
            except (AssertionError, RuntimeError) as e:
                logger.warning(
                    "[arcstore-dcp] async_save unavailable (%s); sync fallback.",
                    str(e)[:80],
                )
        dcp.save(app, storage_writer=writer)
        logger.info(
            "[arcstore-dcp] saved -> %s in %.2fs", dest, time.perf_counter() - t0
        )
    except ImportError:
        # Fallback: DCP to local FileSystem, then parallel s5cmd upload.
        local = os.path.join(_DCP_SAVE_STAGE, dest.rstrip("/").split("/")[-1])
        Path(local).mkdir(parents=True, exist_ok=True)
        dcp.save(app, checkpoint_id=local)
        if _rank() == 0:
            upload_dir(local, dest, workers=256)
        logger.info(
            "[arcstore-dcp] saved (local+s5cmd) -> %s in %.2fs",
            dest,
            time.perf_counter() - t0,
        )


def load_full_state(
    src: str,
    models: Union[torch.nn.Module, Iterable[torch.nn.Module]],
    optimizers,
) -> int:
    """Restore the complete FSDP training state from ``src``. Returns global step.

    ``src`` may be an ``s3://`` prefix (shards staged to local NVMe with
    s5cmd, once per node, then read via the default ``FileSystemReader``) or
    a local directory. DCP does small random reads, so even for a mounted
    bucket the NVMe staging path is kept — it is strictly faster than FUSE
    and already barrier-coordinated per node.
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import DefaultLoadPlanner

    prime_optim_state(models, optimizers)

    app = {"app": _make_app_state(models, optimizers)}
    t0 = time.perf_counter()
    # Saved checkpoints hold sharded Adam tensors; a freshly primed
    # optimizer's plan structure should match, but allow_partial_load
    # tolerates benign mismatches (e.g. frozen params absent).
    planner = DefaultLoadPlanner(allow_partial_load=True)

    if not is_s3(src):
        dcp.load(app, checkpoint_id=src, planner=planner)
        load_dir = src
    else:
        load_dir = os.path.join(_load_stage_root(), src.rstrip("/").split("/")[-1])
        if _local_rank() == 0:
            download_dir(src, load_dir, workers=256)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        dcp.load(app, checkpoint_id=load_dir, planner=planner)

    os_ = _as_list(optimizers)
    for optimizer in os_:
        for group in optimizer.param_groups:
            for p in group["params"]:
                st = optimizer.state.get(p)
                if not st:
                    continue
                for k, v in st.items():
                    if isinstance(v, torch.Tensor) and v.device != p.device:
                        st[k] = v.to(p.device)

    step = _load_train_meta(load_dir)
    logger.info(
        "[arcstore-dcp] loaded <- %s in %.2fs (step=%d)",
        src,
        time.perf_counter() - t0,
        step,
    )
    return int(step)


def dcp_dir_exists(path: str) -> bool:
    """True if ``path`` looks like a populated DCP checkpoint dir (local or s3)."""
    if is_s3(path):
        import shutil
        import subprocess

        listing = path.rstrip("/") + "/"
        if shutil.which("s5cmd"):
            proc = subprocess.run(
                ["s5cmd", "ls", listing],
                capture_output=True,
                text=True,
                check=False,
            )
            return proc.returncode == 0 and bool(proc.stdout.strip())
        try:
            import boto3

            bucket, key = split_s3(listing)
            resp = boto3.client("s3").list_objects_v2(
                Bucket=bucket, Prefix=key, MaxKeys=1
            )
            return resp.get("KeyCount", 0) > 0
        except Exception:
            return False
    # Local: DCP writes a ``.metadata`` file at the root.
    return os.path.isdir(path) and os.path.exists(os.path.join(path, ".metadata"))


__all__ = [
    "dcp_dir_exists",
    "load_full_state",
    "patch_dcp_wrap_exception_py313",
    "prime_optim_state",
    "save_full_state",
]
