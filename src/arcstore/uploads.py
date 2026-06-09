"""Write-back primitives: sync/async S3 uploads + bulk download.

Writes ALWAYS go through the S3 API (s5cmd preferred, boto3 fallback) —
never through a FUSE mount, which rejects overwrites. Lifted from
CausalVideoDiffusion ``src/utils/s3_io.py``.

Background upload pool
----------------------

Single worker on purpose: concurrent multi-GiB ``s5cmd cp`` invocations
from the same node fight over the S3 egress bandwidth and starve each
other; one queued worker, multiple pending tasks, deterministic order.

``wait_for_uploads()`` flushes pending work and re-raises the first failure
on the main thread — call it before process exit. An atexit hook is also
registered as a safety net, but it only LOGS failures (atexit must not mask
the real exit reason); the explicit call remains the loud-failure path.
"""
from __future__ import annotations

import atexit
import logging
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from ._env import default_workers
from .location import is_s3, split_s3

logger = logging.getLogger(__name__)

_UPLOAD_POOL: ThreadPoolExecutor | None = None
_PENDING: list = []
_POOL_LOCK = threading.Lock()
_ATEXIT_REGISTERED = False


def _pool() -> ThreadPoolExecutor:
    global _UPLOAD_POOL, _ATEXIT_REGISTERED
    with _POOL_LOCK:
        if _UPLOAD_POOL is None:
            _UPLOAD_POOL = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="arcstore-upload"
            )
        if not _ATEXIT_REGISTERED:
            atexit.register(_atexit_flush)
            _ATEXIT_REGISTERED = True
        return _UPLOAD_POOL


def track_future(fut) -> None:
    """Register an externally-created Future for the shutdown flush.

    Used e.g. by ``arcstore.torch.dcp`` to fold DCP ``async_save`` futures
    into the same flush barrier as regular uploads.
    """
    with _POOL_LOCK:
        _PENDING.append(fut)


def wait_for_uploads(timeout_s: float | None = None) -> None:
    """Block until queued background uploads finish; re-raise any failures.

    Call this on shutdown (after the training loop, before process exit).
    Failures during training are intentionally silent in the background
    pool — surfacing them at shutdown via ``Future.result()`` is
    loud-enough-on-time.
    """
    with _POOL_LOCK:
        pending = list(_PENDING)
        _PENDING.clear()
    if not pending:
        return
    logger.info(f"[arcstore] flushing {len(pending)} background upload(s)")
    deadline = (time.monotonic() + timeout_s) if timeout_s is not None else None
    for fut in pending:
        wait = None if deadline is None else max(0.0, deadline - time.monotonic())
        fut.result(timeout=wait)  # raises on background failure


def _atexit_flush() -> None:
    try:
        wait_for_uploads()
    except Exception:  # noqa: BLE001
        logger.exception("[arcstore] background upload failed during atexit flush")


def upload_file(local_path: str, s3_uri: str, *, workers: int | None = None) -> None:
    """Upload a single file to ``s3://bucket/key`` (s5cmd preferred).

    ``s5cmd cp`` always uploads (never compares mtimes), which is what we
    want for checkpoints — the file is fresh and a duplicated call just
    re-uploads the same bytes.
    """
    if not is_s3(s3_uri):
        raise ValueError(f"upload_file expects an s3:// destination, got {s3_uri!r}")
    workers = workers if workers is not None else default_workers()
    if shutil.which("s5cmd"):
        cmd = ["s5cmd", "--numworkers", str(workers), "cp", local_path, s3_uri]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            return
        logger.warning(
            f"[arcstore] s5cmd cp {local_path} -> {s3_uri} failed "
            f"(rc={proc.returncode}): {(proc.stderr or '').strip()[:300]}; "
            "falling back to boto3."
        )
    _boto3_upload_file(local_path, s3_uri)


def upload_dir(
    local_dir: str, s3_uri: str, *, workers: int | None = None, recursive: bool = True
) -> None:
    """Upload a directory tree to ``s3://...`` (s5cmd cp, recursive)."""
    if not is_s3(s3_uri):
        raise ValueError(f"upload_dir expects an s3:// destination, got {s3_uri!r}")
    workers = workers if workers is not None else default_workers()
    src = local_dir.rstrip("/") + "/"
    dst = s3_uri.rstrip("/") + "/"
    if shutil.which("s5cmd"):
        cmd = ["s5cmd", "--numworkers", str(workers), "cp", src, dst]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            return
        logger.warning(
            f"[arcstore] s5cmd cp dir failed (rc={proc.returncode}): "
            f"{(proc.stderr or '').strip()[:300]}; falling back to boto3."
        )
    _boto3_upload_dir(local_dir, s3_uri)


def upload_file_async(local_path: str, s3_uri: str) -> None:
    """Queue ``upload_file`` on the background pool; track for shutdown flush."""
    fut = _pool().submit(upload_file, local_path, s3_uri)
    track_future(fut)
    logger.info(f"[arcstore] queued background upload {local_path} -> {s3_uri}")


def upload_dir_async(local_dir: str, s3_uri: str) -> None:
    """Queue ``upload_dir`` on the background pool; track for shutdown flush."""
    fut = _pool().submit(upload_dir, local_dir, s3_uri)
    track_future(fut)
    logger.info(f"[arcstore] queued background upload {local_dir}/ -> {s3_uri}/")


def download_dir(s3_uri: str, local_dir: str, *, workers: int | None = None) -> None:
    """Download an S3 prefix to a local directory (s5cmd preferred).

    Always uses the S3 API even when the bucket is FUSE-mounted: s5cmd's
    multipart fan-out beats FUSE reads for multi-GiB transfers.
    """
    if not is_s3(s3_uri):
        raise ValueError(f"download_dir expects an s3:// source, got {s3_uri!r}")
    workers = workers if workers is not None else default_workers()
    os.makedirs(local_dir, exist_ok=True)
    src = s3_uri.rstrip("/") + "/*"
    dst = local_dir.rstrip("/") + "/"
    if shutil.which("s5cmd"):
        cmd = ["s5cmd", "--numworkers", str(workers), "cp", src, dst]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            return
        logger.warning(
            f"[arcstore] s5cmd cp down failed (rc={proc.returncode}): "
            f"{(proc.stderr or '').strip()[:300]}; falling back to boto3."
        )
    _boto3_download_dir(s3_uri, local_dir)


# ---------------------------------------------------------------------------
# boto3 fallbacks
# ---------------------------------------------------------------------------
def _boto3_upload_file(local_path: str, s3_uri: str) -> None:
    import boto3

    bucket, key = split_s3(s3_uri)
    boto3.client("s3").upload_file(local_path, bucket, key)


def _boto3_upload_dir(local_dir: str, s3_uri: str) -> None:
    import boto3

    bucket, key_prefix = split_s3(s3_uri)
    client = boto3.client("s3")
    base = local_dir.rstrip("/")
    for root, _dirs, files in os.walk(base):
        for fn in files:
            p = os.path.join(root, fn)
            rel = os.path.relpath(p, base).replace(os.sep, "/")
            client.upload_file(p, bucket, f"{key_prefix.rstrip('/')}/{rel}")


def _boto3_download_dir(s3_uri: str, local_dir: str) -> None:
    import boto3

    bucket, key_prefix = split_s3(s3_uri)
    client = boto3.client("s3")
    paginator = client.get_paginator("list_objects_v2")
    base = local_dir.rstrip("/")
    for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix.rstrip("/") + "/"):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(key_prefix.rstrip("/")) + 1:]
            if not rel:
                continue
            dst = os.path.join(base, rel)
            os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
            client.download_file(bucket, obj["Key"], dst)
