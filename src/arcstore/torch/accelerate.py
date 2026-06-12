"""Accelerate / DeepSpeed full-state checkpoint helpers.

These helpers cover the non-DCP path used by DeepSpeed/Accelerate:

* save full state to node-local storage with ``accelerator.save_state``;
* upload the directory to S3 from each node's local main process;
* stage S3 checkpoints back to local SSD before ``accelerator.load_state``.

They intentionally avoid a hard dependency on accelerate; any object with the
usual ``save_state`` / ``load_state`` / ``wait_for_everyone`` attributes works.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from .._env import default_workers, env_str, local_ssd_or_tmp
from ..location import is_s3
from ..uploads import download_dir, track_future, upload_dir

logger = logging.getLogger(__name__)

_ACCEL_STAGE = "/local-ssd/arcstore/accelerate_load"
_ACCEL_STAGE_FALLBACK = "/tmp/arcstore/accelerate_load"
_POOL: ThreadPoolExecutor | None = None


def _pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="arcstore-accel")
    return _POOL


def _wait(accelerator) -> None:
    wait = getattr(accelerator, "wait_for_everyone", None)
    if callable(wait):
        wait()


def _is_main(accelerator) -> bool:
    return bool(getattr(accelerator, "is_main_process", True))


def _is_local_main(accelerator) -> bool:
    return bool(getattr(accelerator, "is_local_main_process", _is_main(accelerator)))


def _load_stage_root() -> str:
    default = local_ssd_or_tmp(_ACCEL_STAGE, _ACCEL_STAGE_FALLBACK)
    return env_str("ARCSTORE_ACCELERATE_STAGE_DIR", default)


def _stage_dir_for_s3(root: str, uri: str) -> str:
    norm = uri.rstrip("/")
    basename = norm.rsplit("/", 1)[-1] or "accelerate-state"
    key = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]
    return os.path.join(root, f"{key}__{basename}")


def _parse_step(path: str) -> int:
    name = path.rstrip("/").rsplit("/", 1)[-1]
    for pat in (r"checkpoint[-_](\d+)$", r"checkpoint_model_(\d+)$"):
        m = re.search(pat, name)
        if m:
            return int(m.group(1))
    return 0


def _save_ema(local_dir: str, ema) -> None:
    if ema is None:
        return
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    state = ema.state_dict() if hasattr(ema, "state_dict") else ema
    torch.save(state, str(Path(local_dir) / "ema.pt"))


def _load_ema(local_dir: str, ema) -> bool:
    if ema is None:
        return False
    path = Path(local_dir) / "ema.pt"
    if not path.exists():
        logger.warning("[arcstore-accel] no ema.pt at %s; skipping EMA restore", local_dir)
        return False
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    ema.load_state_dict(state)
    return True


def _upload_state_dir(local_dir: str, s3_uri: str, keep_local: bool, workers: int) -> None:
    upload_dir(local_dir, s3_uri, workers=workers)
    logger.info("[arcstore-accel] uploaded full state %s -> %s", local_dir, s3_uri)
    if not keep_local:
        shutil.rmtree(local_dir, ignore_errors=True)


def save_accelerate_state(
    accelerator,
    local_dir: str,
    s3_uri: str | None = None,
    *,
    ema=None,
    async_upload: bool = False,
    keep_local: bool = False,
    upload_workers: int | None = None,
) -> None:
    """Save an Accelerate/DeepSpeed full-state checkpoint.

    ``local_dir`` should be under local SSD. If ``s3_uri`` is provided, each
    node's local main process uploads its local shards to the same S3 prefix.
    """
    accelerator.save_state(local_dir)
    if _is_main(accelerator):
        _save_ema(local_dir, ema)
    _wait(accelerator)

    if not s3_uri:
        logger.info("[arcstore-accel] saved full state to %s", local_dir)
        return
    if not is_s3(s3_uri):
        raise ValueError(f"save_accelerate_state expects s3:// destination, got {s3_uri!r}")

    workers = upload_workers if upload_workers is not None else default_workers()
    if _is_local_main(accelerator):
        if async_upload:
            fut = _pool().submit(_upload_state_dir, local_dir, s3_uri, keep_local, workers)
            track_future(fut)
            logger.info("[arcstore-accel] queued upload %s -> %s", local_dir, s3_uri)
        else:
            _upload_state_dir(local_dir, s3_uri, keep_local, workers)


def load_accelerate_state(
    accelerator,
    source: str,
    *,
    local_dir: str | None = None,
    ema=None,
    download_workers: int | None = None,
    required_files=None,
) -> int:
    """Load an Accelerate/DeepSpeed full-state checkpoint; return parsed step."""
    if is_s3(source):
        load_dir = local_dir or _stage_dir_for_s3(_load_stage_root(), source)
        workers = download_workers if download_workers is not None else default_workers()
        if _is_local_main(accelerator):
            download_dir(
                source,
                load_dir,
                workers=workers,
                required_files=required_files,
                require_nonempty=True,
            )
            logger.info("[arcstore-accel] downloaded %s -> %s", source, load_dir)
        _wait(accelerator)
    else:
        load_dir = source

    accelerator.load_state(load_dir)
    if _is_main(accelerator):
        _load_ema(load_dir, ema)
    _wait(accelerator)
    return _parse_step(source)


save_deepspeed_state = save_accelerate_state
load_deepspeed_state = load_accelerate_state

__all__ = [
    "load_accelerate_state",
    "load_deepspeed_state",
    "save_accelerate_state",
    "save_deepspeed_state",
]
