"""Shared core for the torch checkpoint modules (``dcp`` / ``manager`` /
``accelerate_ckpt``).

Single source of truth for the DCP training-state checkpoint mechanics:

* :func:`make_app_state` — the DCP ``Stateful`` bundling sharded model
  (+ optimizer) tensors; accepts a single object or equal-length lists, and a
  single model with ``optimizers=None`` saves model-only;
* :func:`prime_optim_state` / :func:`move_optim_state_to_param_devices` —
  materialize Adam buffers before a strict load and place them on the param
  devices afterwards;
* the per-rank **extras sidecar** (``extras_rank{R}.pt``) + the
  ``_ARC_COMPLETE`` **completeness marker**, written last so a torn
  (preempted-mid-save) checkpoint is never resumed from;
* :func:`dcp_save_app` / :func:`dcp_load_app` — the low-level
  ``dcp.save``/``dcp.load`` calls, S3 (``s3torchconnector``) or local.

Both the high-level :class:`arcstore.torch.CheckpointManager` and the
standalone :func:`arcstore.torch.save_full_state` / ``load_full_state`` build
on these, so the on-disk layout is identical across the two entry points
(step lives in the extras sidecar; completeness is signalled by the marker or
DCP's own ``.metadata``).
"""
from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, List, Mapping

import torch

from .._env import aws_region
from ..location import is_s3
from ..uploads import upload_file
from .runtime import barrier, get_rank, is_main

logger = logging.getLogger(__name__)

COMPLETE_MARKER = "_ARC_COMPLETE"


# ---------------------------------------------------------------------------
# stage dir + ema side-file (used by dcp load staging + the accelerate path)
# ---------------------------------------------------------------------------
def stage_dir_for_s3(root: str, uri: str, *, default_basename: str = "ckpt") -> str:
    """Stable, collision-resistant local stage dir for one S3 prefix.

    The directory name is ``<sha1(uri)[:16]>__<last-segment>`` under ``root``,
    so every artifact of one checkpoint shares a single staged dir.
    """
    norm = uri.rstrip("/")
    basename = norm.rsplit("/", 1)[-1] or default_basename
    key = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]
    return os.path.join(root, f"{key}__{basename}")


def load_ema(local_dir: str, ema, *, label: str = "arcstore-ckpt") -> bool:
    """Restore an optional ``ema.pt`` side-file from ``local_dir``.

    Returns True when the EMA state was loaded, False when ``ema`` is None or
    no ``ema.pt`` is present (a missing file logs a warning and is skipped).
    """
    if ema is None:
        return False
    path = Path(local_dir) / "ema.pt"
    if not path.exists():
        logger.warning("[%s] no ema.pt at %s; skipping EMA restore", label, local_dir)
        return False
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    ema.load_state_dict(state)
    return True


# ---------------------------------------------------------------------------
# value <-> state_dict helpers
# ---------------------------------------------------------------------------
def state_dict_or_value(obj):
    """``obj.state_dict()`` when stateful, else ``obj`` unchanged (None stays None)."""
    if obj is None:
        return None
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    return obj


def load_state_dict_or_return(target, state):
    """Restore ``state`` into ``target`` when stateful; always return ``state``."""
    if target is not None and state is not None and hasattr(target, "load_state_dict"):
        target.load_state_dict(state)
    return state


def _as_list(x) -> List:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


# ---------------------------------------------------------------------------
# DCP app state (sharded model + optimizer tensors only)
# ---------------------------------------------------------------------------
def make_app_state(models, optimizers):
    """A DCP ``Stateful`` bundling ONLY sharded model (+ optimizer) *tensors*.

    ``get_state_dict`` / ``set_state_dict`` normalize FSDP sharding and FQNs so
    the checkpoint reshards across topologies. ``models`` / ``optimizers`` may
    be a single object or equal-length lists; passing ``optimizers=None`` (or
    an empty list) saves model weights only.
    """
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict,
        get_state_dict,
        set_model_state_dict,
        set_state_dict,
    )
    from torch.distributed.checkpoint.stateful import Stateful

    ms = _as_list(models)
    os_ = _as_list(optimizers)
    if os_ and len(ms) != len(os_):
        raise ValueError(
            f"DCP full-state checkpoint expects one optimizer per model; "
            f"got {len(ms)} model(s) and {len(os_)} optimizer(s)."
        )
    has_optim = len(os_) > 0
    single = len(ms) == 1

    class _AppState(Stateful):
        def state_dict(self):
            if not has_optim:
                if single:
                    return {"model": get_model_state_dict(ms[0])}
                return {"model": {str(i): get_model_state_dict(m) for i, m in enumerate(ms)}}
            if single:
                msd, osd = get_state_dict(ms[0], os_[0])
                return {"model": msd, "optim": osd}
            model_state, optim_state = {}, {}
            for i, (m, o) in enumerate(zip(ms, os_)):
                msd, osd = get_state_dict(m, o)
                model_state[str(i)] = msd
                optim_state[str(i)] = osd
            return {"model": model_state, "optim": optim_state}

        def load_state_dict(self, state_dict):
            if not has_optim:
                if single:
                    set_model_state_dict(ms[0], state_dict["model"])
                else:
                    for i, m in enumerate(ms):
                        set_model_state_dict(m, state_dict["model"][str(i)])
                return
            if single:
                set_state_dict(
                    ms[0],
                    os_[0],
                    model_state_dict=state_dict["model"],
                    optim_state_dict=state_dict["optim"],
                )
                return
            for i, (m, o) in enumerate(zip(ms, os_)):
                set_state_dict(
                    m,
                    o,
                    model_state_dict=state_dict["model"][str(i)],
                    optim_state_dict=state_dict["optim"][str(i)],
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
    if not os_ or all(_optim_has_tensor_state(o) for o in os_):
        return

    was_training = [m.training for m in ms]
    for m in ms:
        m.train()
    loss = None
    for m in ms:
        for p in m.parameters():
            if p.requires_grad:
                term = p.sum() * 0.0
                loss = term if loss is None else loss + term
    if loss is None:
        for m, w in zip(ms, was_training):
            m.train(w)
        return
    loss.backward()
    for o in os_:
        o.step()
        o.zero_grad(set_to_none=True)
    for m, w in zip(ms, was_training):
        m.train(w)
    barrier()

    if not all(_optim_has_tensor_state(o) for o in os_):
        raise RuntimeError(
            "[arcstore-ckpt] failed to materialize optimizer state before DCP load"
        )


def move_optim_state_to_param_devices(optimizers) -> None:
    for optimizer in _as_list(optimizers):
        for group in optimizer.param_groups:
            for p in group["params"]:
                st = optimizer.state.get(p)
                if not st:
                    continue
                for k, v in st.items():
                    if isinstance(v, torch.Tensor) and v.device != p.device:
                        st[k] = v.to(p.device)


# ---------------------------------------------------------------------------
# extras sidecar (step + per-rank stateful payloads) + completeness marker
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _tmp_path(suffix: str):
    """Yield a closed temp-file path, removed on exit (for S3 staging)."""
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        yield tmp
    finally:
        with contextlib.suppress(OSError):
            os.remove(tmp)


def _extras_filename(rank: int) -> str:
    return f"extras_rank{rank}.pt"


def write_extras(dest: str, step: int, extras: Mapping[str, Any] | None) -> None:
    """Write this rank's ``{step, extras}`` sidecar next to the checkpoint.

    Stateful values are stored via ``state_dict()``; plain values are stored
    as-is, so callers can mix LR schedulers / EMA (stateful) and arbitrary
    metadata.
    """
    payload = {
        "step": int(step),
        "extras": {name: state_dict_or_value(comp) for name, comp in (extras or {}).items()},
    }
    filename = _extras_filename(get_rank())
    if is_s3(dest):
        with _tmp_path(".pt") as tmp:
            torch.save(payload, tmp)
            upload_file(tmp, dest.rstrip("/") + "/" + filename)
    else:
        Path(dest).mkdir(parents=True, exist_ok=True)
        torch.save(payload, str(Path(dest) / filename))


def read_extras_payload(local_dir: str | Path) -> dict:
    """Load this rank's extras sidecar (falling back to rank 0's, then to the
    step parsed from the dir name) without restoring anything."""
    local_dir = Path(local_dir)
    path = local_dir / _extras_filename(get_rank())
    if not path.exists():
        fallback = local_dir / _extras_filename(0)
        if fallback.exists():
            logger.warning(
                "[arcstore-ckpt] no extras file for rank %d; falling back to rank 0's "
                "(expected after a world-size change)",
                get_rank(),
            )
            path = fallback
        else:
            try:
                step = int(local_dir.name.rsplit("-", 1)[-1])
            except ValueError:
                step = 0
            return {"step": step, "extras": {}}
    return torch.load(str(path), map_location="cpu", weights_only=False)


def read_extras(local_dir: str | Path, extras: Mapping[str, Any] | None) -> int:
    """Restore each requested extra from the sidecar; return the saved step."""
    payload = read_extras_payload(local_dir)
    saved = payload.get("extras", {})
    for name, comp in (extras or {}).items():
        if name in saved:
            load_state_dict_or_return(comp, saved[name])
        else:
            logger.warning(
                "[arcstore-ckpt] checkpoint has no saved state for extra '%s'; skipping", name
            )
    return int(payload.get("step", 0))


def write_marker(dest: str) -> None:
    """Write the completeness marker as the final object (rank 0 only)."""
    if not is_main():
        return
    if is_s3(dest):
        with _tmp_path(".marker") as tmp:
            Path(tmp).write_text("ok")
            upload_file(tmp, dest.rstrip("/") + "/" + COMPLETE_MARKER)
    else:
        (Path(dest) / COMPLETE_MARKER).write_text("ok")


# ---------------------------------------------------------------------------
# low-level DCP save / load
# ---------------------------------------------------------------------------
def dcp_save_app(app: dict, dest: str, *, async_save: bool = False, thread_count: int = 16):
    """Run ``dcp.save`` / ``dcp.async_save`` to a local dir or ``s3://`` prefix.

    Returns the async future when ``async_save`` is requested and supported
    (the caller is responsible for the completeness marker once it lands);
    otherwise the save completes synchronously and ``None`` is returned.
    """
    import torch.distributed.checkpoint as dcp

    if is_s3(dest):
        try:
            from s3torchconnector.dcp import S3StorageWriter
        except ImportError as e:
            raise ImportError(
                "S3 DCP checkpoints require s3torchconnector (arcstore[torch])."
            ) from e
        save_kw = {
            "storage_writer": S3StorageWriter(
                region=aws_region(), path=dest, thread_count=thread_count
            )
        }
    else:
        Path(dest).mkdir(parents=True, exist_ok=True)
        save_kw = {"checkpoint_id": str(dest)}

    if async_save:
        try:
            return dcp.async_save(app, **save_kw)
        except (AssertionError, RuntimeError):
            logger.warning("[arcstore-ckpt] dcp.async_save unavailable; falling back to sync")
    dcp.save(app, **save_kw)
    return None


def dcp_load_app(app: dict, load_dir: str, *, strict: bool = False) -> None:
    """Run ``dcp.load`` from a local directory (S3 sources are staged first)."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import DefaultLoadPlanner

    dcp.load(
        app,
        checkpoint_id=str(load_dir),
        planner=DefaultLoadPlanner(allow_partial_load=not strict),
    )


__all__ = [
    "COMPLETE_MARKER",
    "dcp_load_app",
    "dcp_save_app",
    "load_ema",
    "load_state_dict_or_return",
    "make_app_state",
    "move_optim_state_to_param_devices",
    "prime_optim_state",
    "read_extras",
    "read_extras_payload",
    "stage_dir_for_s3",
    "state_dict_or_value",
    "write_extras",
    "write_marker",
]
