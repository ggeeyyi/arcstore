"""High-level training-state checkpoint manager with auto-resume.

Ported from ``arc_toolkit.checkpoint.CheckpointManager`` and rewired onto the
arcstore core primitives (:mod:`arcstore.uploads`, :mod:`arcstore.s3cli`,
:mod:`arcstore.torch.runtime`). This is the orchestration layer; the DCP
mechanics (app-state, optimizer priming, the extras sidecar + completeness
marker, the low-level ``dcp.save``/``dcp.load``) live in the shared core
:mod:`arcstore.torch._ckpt_common`, so the standalone
:func:`arcstore.torch.save_full_state` writes the exact same on-disk layout.

- dispatches on the *model object*: a DeepSpeed engine uses its native
  collective checkpoint + an s5cmd S3 upload; anything else (FSDP / DDP /
  plain ``nn.Module``) goes through torch Distributed Checkpoint (DCP);
- writes ``{local_dir|s3_prefix}/checkpoint-{step}`` with a completeness
  marker so a torn (preempted mid-save) checkpoint is never resumed from;
- ``keep_last`` garbage-collects older checkpoints;
- :meth:`latest_checkpoint` / :meth:`load_latest` find and resume from the
  highest-step *complete* checkpoint — the preemption-safe training start;
- arbitrary per-rank ``extras={name: stateful}`` (LR scheduler, EMA,
  :class:`arcstore.torch.RNGState`, GradScaler, ...) are saved per rank
  (``extras_rank{R}.pt``) so per-rank streams (RNG) stay per-rank.
"""
from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from .._env import cache_dir
from ..location import is_s3
from ..s3cli import head_object as _head_object
from ..s3cli import ls_prefix as _ls_prefix
from ..s3cli import remove_prefix as _s3_rmtree
from ..uploads import _pool, download_dir, track_future, upload_dir, wait_for_uploads
from ._ckpt_common import (
    COMPLETE_MARKER,
    dcp_load_app,
    dcp_save_app,
    make_app_state,
    move_optim_state_to_param_devices,
    prime_optim_state,
    read_extras,
    write_extras,
    write_marker,
)
from .dcp import patch_dcp_wrap_exception_py313
from .runtime import barrier, is_local_main, is_main

logger = logging.getLogger(__name__)

__all__ = ["CheckpointManager", "Stateful"]

patch_dcp_wrap_exception_py313()


class Stateful(Protocol):
    """Anything checkpointable: optimizers, schedulers, EMA, RNGState, ..."""

    def state_dict(self) -> dict: ...

    def load_state_dict(self, state_dict: dict) -> None: ...


def _unwrap_compiled(model):
    return getattr(model, "_orig_mod", model)


def _is_deepspeed_engine(obj) -> bool:
    import sys

    ds = sys.modules.get("deepspeed")
    if ds is not None and isinstance(obj, getattr(ds, "DeepSpeedEngine", ())):
        return True
    return callable(getattr(obj, "save_checkpoint", None)) and callable(
        getattr(obj, "load_checkpoint", None)
    )


# ----------------------------------------------------------- S3 helpers
def _s3_object_exists(uri: str) -> bool:
    try:
        return _head_object(uri) is not None
    except Exception:  # noqa: BLE001
        return False


def _s3_list_dirs(s3_prefix: str) -> list[str]:
    """Immediate child 'directory' names under an S3 prefix (no trailing slash)."""
    return [e.name.rstrip("/") for e in _ls_prefix(s3_prefix) if e.is_dir]


# ------------------------------------------------------------ completeness
def _member_exists(ckpt_uri: str, name: str) -> bool:
    if is_s3(ckpt_uri):
        return _s3_object_exists(ckpt_uri.rstrip("/") + "/" + name)
    return (Path(ckpt_uri) / name).exists()


def _is_complete(ckpt_uri: str) -> bool:
    """Resumable when our marker is present, or — for checkpoints written before
    markers existed — DCP's ``.metadata`` (written last) is."""
    return _member_exists(ckpt_uri, COMPLETE_MARKER) or _member_exists(ckpt_uri, ".metadata")


# --------------------------------------------------------------- the manager
class CheckpointManager:
    """Save/load full training state for DeepSpeed engines and torch modules.

    ``save(step, model, optimizer, extras={...})`` writes
    ``{local_dir|s3_prefix}/checkpoint-{step}``. With ``async_save=True`` the
    transfer overlaps training (DCP async or background s5cmd upload); a new
    ``save()`` first drains the previous one, and :meth:`wait` blocks until
    everything in flight has landed.

    ``keep_last`` (default 3) garbage-collects older checkpoints after each save
    so S3 / disk does not grow unbounded over a long run; set to ``None`` to keep
    every checkpoint.
    """

    def __init__(
        self,
        *,
        local_dir: str | Path | None = None,
        s3_prefix: str | None = None,
        async_save: bool = False,
        keep_local: bool = False,
        keep_last: int | None = 3,
        io_workers: int = 256,
        dcp_thread_count: int = 16,
    ):
        self.local_dir = Path(local_dir) if local_dir else Path("./checkpoints")
        self.s3_prefix = s3_prefix.rstrip("/") if s3_prefix else None
        self.async_save = async_save
        self.keep_local = keep_local
        self.keep_last = keep_last
        self.io_workers = io_workers
        self.dcp_thread_count = dcp_thread_count
        self._inflight: list = []

    # ------------------------------------------------------------- plumbing
    def _paths(self, step: int) -> tuple[str, Path, str | None]:
        name = f"checkpoint-{step}"
        s3_uri = f"{self.s3_prefix}/{name}" if self.s3_prefix else None
        return name, self.local_dir / name, s3_uri

    def _track(self, fut, dest: str | None) -> None:
        track_future(fut)
        self._inflight.append((fut, dest))

    def _drain_inflight(self) -> None:
        """Wait for every in-flight async save; mark each that lands; never orphan
        the list (a failed future must not permanently block future saves)."""
        if not self._inflight:
            return
        inflight, self._inflight = self._inflight, []
        errors: list[BaseException] = []
        for fut, dest in inflight:
            waiter = getattr(fut, "result", None) or fut.wait
            try:
                waiter()
            except Exception as e:  # keep draining the rest; don't poison later saves
                errors.append(e)
            else:
                if dest is not None:
                    write_marker(dest)
        if errors:
            raise RuntimeError(
                f"[arcstore-ckpt] {len(errors)} async checkpoint save(s) failed"
            ) from errors[0]

    def wait(self) -> None:
        """Block until all of this manager's async saves and uploads finish."""
        self._drain_inflight()
        wait_for_uploads()

    # ----------------------------------------------------------------- save
    def save(
        self,
        step: int,
        model,
        optimizer=None,
        *,
        extras: Mapping[str, Stateful] | None = None,
    ) -> None:
        """Save full training state; dispatches on the model object (see class doc)."""
        model = _unwrap_compiled(model)
        self._drain_inflight()
        name, local, s3_uri = self._paths(step)
        t0 = time.perf_counter()
        if _is_deepspeed_engine(model):
            self._save_deepspeed(model, step, name, local, s3_uri, extras)
        else:
            self._save_dcp(model, optimizer, step, local, s3_uri, extras)
        logger.info(
            "[arcstore-ckpt] save step %d -> %s%s in %.2fs",
            step,
            s3_uri or local,
            " (async)" if self.async_save else "",
            time.perf_counter() - t0,
        )
        self._prune_old(step)

    def _prune_old(self, current_step: int) -> None:
        """Keep only the most recent ``keep_last`` checkpoints (rank 0; current kept)."""
        if not self.keep_last or self.keep_last <= 0 or not is_main():
            return
        roots: list[tuple[str, bool]] = []
        if self.s3_prefix:
            roots.append((self.s3_prefix, True))
        if self.local_dir.is_dir():
            roots.append((str(self.local_dir), False))
        for base, remote in roots:
            names = (
                _s3_list_dirs(base)
                if remote
                else [p.name for p in self.local_dir.glob("checkpoint-*")]
            )
            complete_steps: list[int] = []
            for n in names:
                if not n.startswith("checkpoint-") or not n.rsplit("-", 1)[-1].isdigit():
                    continue
                step = int(n.rsplit("-", 1)[-1])
                uri = f"{base.rstrip('/')}/checkpoint-{step}"
                if _is_complete(uri):
                    complete_steps.append(step)
            complete_steps.sort()
            protected = set(complete_steps[-self.keep_last :])
            protected.add(current_step)  # never prune the checkpoint we just wrote
            for step in complete_steps:
                if step in protected:
                    continue
                uri = f"{base.rstrip('/')}/checkpoint-{step}"
                if remote:
                    _s3_rmtree(uri)
                else:
                    shutil.rmtree(uri, ignore_errors=True)
                logger.info(
                    "[arcstore-ckpt] pruned old checkpoint %s (keep_last=%d)",
                    uri,
                    self.keep_last,
                )

    def _save_deepspeed(self, engine, step, name, local, s3_uri, extras) -> None:
        engine.save_checkpoint(str(self.local_dir), tag=name)
        write_extras(str(local), step, extras)
        barrier()
        if s3_uri is None:
            write_marker(str(local))
            return
        if not is_local_main():
            return

        def _upload_and_mark() -> None:
            upload_dir(str(local), s3_uri, workers=self.io_workers)
            write_marker(s3_uri)  # marker last -> "present" iff upload finished
            if not self.keep_local:
                shutil.rmtree(local, ignore_errors=True)

        if self.async_save:
            self._track(_pool().submit(_upload_and_mark), None)  # job self-marks
        else:
            _upload_and_mark()

    def _save_dcp(self, model, optimizer, step, local, s3_uri, extras) -> None:
        app = {"app": make_app_state(model, optimizer)}
        dest = s3_uri or str(local)
        write_extras(dest, step, extras)
        fut = dcp_save_app(
            app, dest, async_save=self.async_save, thread_count=self.dcp_thread_count
        )
        if fut is not None:
            self._track(fut, dest)
            return
        write_marker(dest)

    # ----------------------------------------------------------------- load
    def load(
        self,
        source: str,
        model,
        optimizer=None,
        *,
        extras: Mapping[str, Stateful] | None = None,
        strict: bool = False,
    ) -> int:
        """Load full training state from ``source`` (local dir or ``s3://``); return the step."""
        model = _unwrap_compiled(model)
        t0 = time.perf_counter()
        if _is_deepspeed_engine(model):
            step = self._load_deepspeed(model, source, extras)
        else:
            step = self._load_dcp(model, optimizer, source, extras, strict)
        logger.info(
            "[arcstore-ckpt] loaded <- %s in %.2fs (step=%d)",
            source,
            time.perf_counter() - t0,
            step,
        )
        return step

    # ---------------------------------------------------------- auto-resume
    def latest_checkpoint(self) -> str | None:
        """URI of the highest-step *complete* checkpoint, or None if there is none.

        Prefers ``s3_prefix`` when set (the durable copy), else ``local_dir``.
        Torn checkpoints from mid-save preemption (no completeness marker) are
        skipped. Numeric step ordering, not lexicographic.
        """
        best_step, best_uri = -1, None
        if self.s3_prefix:
            base, names = self.s3_prefix, _s3_list_dirs(self.s3_prefix)
        elif self.local_dir.is_dir():
            base = str(self.local_dir)
            names = [p.name for p in self.local_dir.glob("checkpoint-*")]
        else:
            return None
        for name in names:
            if not name.startswith("checkpoint-"):
                continue
            try:
                step = int(name.split("-")[-1])
            except ValueError:
                continue
            uri = base.rstrip("/") + "/" + name
            if step > best_step and _is_complete(uri):
                best_step, best_uri = step, uri
        return best_uri

    def load_latest(
        self,
        model,
        optimizer=None,
        *,
        extras: Mapping[str, Stateful] | None = None,
        strict: bool = False,
    ) -> int:
        """Resume from the latest complete checkpoint; return its step (0 if none).

        The idiomatic preemption-safe start::

            start_step = ckpt.load_latest(model, optimizer, extras=extras)
            for step in range(start_step, max_steps): ...
        """
        uri = self.latest_checkpoint()
        if uri is None:
            logger.info("[arcstore-ckpt] no complete checkpoint to resume from; starting fresh")
            return 0
        logger.info("[arcstore-ckpt] auto-resuming from %s", uri)
        return self.load(uri, model, optimizer, extras=extras, strict=strict)

    def _stage_from_s3(self, source: str) -> Path:
        name = source.rstrip("/").split("/")[-1]
        stage = cache_dir("ckpt-stage", create=False) / name
        if is_local_main():
            download_dir(source, str(stage), workers=self.io_workers)
        barrier()
        return stage

    def _load_deepspeed(self, engine, source, extras) -> int:
        if is_s3(source):
            stage = self._stage_from_s3(source)
            load_dir, tag = str(stage.parent), stage.name
        else:
            p = Path(source)
            load_dir, tag = str(p.parent), p.name
        load_path, _ = engine.load_checkpoint(load_dir, tag=tag)
        if load_path is None:
            raise RuntimeError(f"DeepSpeed failed to load checkpoint from {source}")
        return read_extras(Path(load_dir) / tag, extras)

    def _load_dcp(self, model, optimizer, source, extras, strict) -> int:
        prime_optim_state(model, optimizer)
        load_dir = str(self._stage_from_s3(source)) if is_s3(source) else source
        app = {"app": make_app_state(model, optimizer)}
        dcp_load_app(app, load_dir, strict=strict)
        move_optim_state_to_param_devices(optimizer)
        return read_extras(load_dir, extras)
