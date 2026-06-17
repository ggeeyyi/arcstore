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
import contextlib
import logging
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import IO, Iterator

from ._env import default_workers, s3_retries
from .location import is_s3, split_s3
from .s3cli import have_aws, have_s5cmd, run_cli_candidates

logger = logging.getLogger(__name__)


def _run_transfer_with_retry(candidates, *, context: str):
    """Run CLI transfer candidates with exponential backoff on transient failure.

    Returns ``(result, last_err, not_found)`` like :func:`run_cli_candidates`.
    Only genuine backend failures are retried; a missing object short-circuits
    immediately (no point retrying a 404). The caller still owns the boto3
    fallback for an exhausted/empty candidate list.
    """
    retries = s3_retries()
    delay = 1.0
    last_err: str | None = None
    for attempt in range(1, retries + 1):
        result, last_err, not_found = run_cli_candidates(candidates)
        if result is not None or not_found is not None:
            return result, last_err, not_found
        if attempt < retries:
            logger.warning(
                "[arcstore] %s failed (attempt %d/%d), retrying in %.0fs: %s",
                context,
                attempt,
                retries,
                delay,
                (last_err or "")[-200:],
            )
            time.sleep(delay)
            delay *= 2
    return None, last_err, None

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
    candidates: list[tuple[str, list[str]]] = []
    if have_s5cmd():
        candidates.append(
            (
                "s5cmd",
                ["s5cmd", "--numworkers", str(workers), "cp", local_path, s3_uri],
            )
        )
    if have_aws():
        candidates.append(("aws", ["aws", "s3", "cp", local_path, s3_uri]))
    result, last_err, _not_found = _run_transfer_with_retry(
        candidates, context=f"upload {local_path} -> {s3_uri}"
    )
    if result is not None:
        return
    if last_err is not None:
        logger.warning(
            f"[arcstore] CLI upload {local_path} -> {s3_uri} failed "
            f"({last_err}); falling back to boto3."
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
    candidates: list[tuple[str, list[str]]] = []
    if have_s5cmd():
        candidates.append(
            ("s5cmd", ["s5cmd", "--numworkers", str(workers), "cp", src, dst])
        )
    if have_aws():
        candidates.append(("aws", ["aws", "s3", "cp", "--recursive", src, dst]))
    result, last_err, _not_found = _run_transfer_with_retry(
        candidates, context=f"upload dir {local_dir} -> {s3_uri}"
    )
    if result is not None:
        return
    if last_err is not None:
        logger.warning(
            f"[arcstore] CLI upload dir {local_dir} -> {s3_uri} failed "
            f"({last_err}); falling back to boto3."
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


def put(
    local_path: str,
    dest: str,
    *,
    recursive: bool | None = None,
    async_: bool = False,
    workers: int | None = None,
) -> None:
    """Unified upload entry — file or directory, sync or async.

    Mirrors the read side's single-entry style: one call covers the
    ``upload_file`` / ``upload_dir`` × sync / async matrix.

    * ``recursive=None`` (default) auto-detects from ``local_path``
      (a directory uploads recursively, a file uploads as one object).
    * ``async_=True`` queues on the background pool (flushed by
      :func:`wait_for_uploads`); ``workers`` is ignored in async mode
      (the background variants use the default worker count).

    ``dest`` must be an ``s3://`` destination — writes never touch a mount.
    """
    if recursive is None:
        recursive = os.path.isdir(local_path)
    if async_:
        if recursive:
            upload_dir_async(local_path, dest)
        else:
            upload_file_async(local_path, dest)
        return
    if recursive:
        upload_dir(local_path, dest, workers=workers)
    else:
        upload_file(local_path, dest, workers=workers)


def put_async(local_path: str, dest: str, *, keep_local: bool = True) -> None:
    """Queue a file/dir upload on the background pool; optionally drop the local copy.

    Auto-detects file vs directory from ``local_path`` at submit time. With
    ``keep_local=False`` the local path is removed *after* a successful upload
    (never on failure — the local copy is then the only surviving one). The job
    is tracked for :func:`wait_for_uploads`.
    """
    if not is_s3(dest):
        raise ValueError(f"put_async expects an s3:// destination, got {dest!r}")
    recursive = os.path.isdir(local_path)

    def _do() -> None:
        import shutil

        if recursive:
            upload_dir(local_path, dest)
        else:
            upload_file(local_path, dest)
        if not keep_local:
            if recursive:
                shutil.rmtree(local_path, ignore_errors=True)
            elif os.path.isfile(local_path):
                _safe_remove(local_path)

    fut = _pool().submit(_do)
    track_future(fut)
    logger.info(f"[arcstore] queued background put {local_path} -> {dest}")


def write_bytes(path: str, data: bytes) -> None:
    """Write ``data`` to a local path or an ``s3://`` object.

    The write-side mirror of :func:`arcstore.read_bytes`. For an ``s3://``
    destination the bytes land in a temp file that is then uploaded via the
    S3 API (never the mount, which rejects overwrites); for a local path the
    parent directory is created and the file is written directly.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError(f"write_bytes expects bytes-like data, got {type(data).__name__}")
    if is_s3(path):
        suffix = os.path.splitext(path)[1]
        fd, tmp = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            upload_file(tmp, path)
        finally:
            _safe_remove(tmp)
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


@contextlib.contextmanager
def open_write(path: str, mode: str = "wb") -> Iterator[IO]:
    """Open a local path or ``s3://`` object for writing (``"wb"`` / ``"w"``).

    The write-side mirror of :func:`arcstore.open_read`. For an ``s3://``
    destination the handle points at a temp file that is uploaded on a clean
    exit of the ``with`` block (an exception inside the block skips the upload
    so no partial object is published); for a local path the parent directory
    is created and the file is opened directly.
    """
    if mode not in ("wb", "w"):
        raise ValueError(f"open_write supports 'wb'/'w', got {mode!r}")
    if is_s3(path):
        suffix = os.path.splitext(path)[1]
        fd, tmp = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        try:
            with open(tmp, mode) as f:
                yield f
            upload_file(tmp, path)
        finally:
            _safe_remove(tmp)
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, mode) as f:
        yield f


def _safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _count_local_files(local_dir: str) -> int:
    total = 0
    for _root, _dirs, files in os.walk(local_dir):
        total += len(files)
    return total


def _missing_required_files(local_dir: str, required_files) -> list[str]:
    if not required_files:
        return []
    missing: list[str] = []
    for rel in required_files:
        if not os.path.isfile(os.path.join(local_dir, rel)):
            missing.append(rel)
    return missing


def _download_verified(
    local_dir: str,
    *,
    required_files=None,
    require_nonempty: bool = True,
) -> tuple[bool, str | None]:
    missing = _missing_required_files(local_dir, required_files)
    if missing:
        return False, f"missing required file(s): {', '.join(missing)}"
    if require_nonempty and _count_local_files(local_dir) == 0:
        return False, "downloaded zero files"
    return True, None


def download_dir(
    s3_uri: str,
    local_dir: str,
    *,
    workers: int | None = None,
    retries: int = 3,
    required_files=None,
    require_nonempty: bool = True,
) -> None:
    """Download an S3 prefix to a local directory (s5cmd preferred).

    Always uses the S3 API even when the bucket is FUSE-mounted: s5cmd's
    multipart fan-out beats FUSE reads for multi-GiB transfers.

    ``s5cmd cp prefix/*`` can occasionally return success before a just-written
    prefix is visible through LIST. We verify that files landed, optionally
    check marker files such as DCP ``.metadata``, and retry before falling back
    to boto3.
    """
    if not is_s3(s3_uri):
        raise ValueError(f"download_dir expects an s3:// source, got {s3_uri!r}")
    workers = workers if workers is not None else default_workers()
    os.makedirs(local_dir, exist_ok=True)
    src = s3_uri.rstrip("/") + "/*"
    dst = local_dir.rstrip("/") + "/"
    candidates: list[tuple[str, list[str]]] = []
    if have_s5cmd():
        candidates.append(
            ("s5cmd", ["s5cmd", "--numworkers", str(workers), "cp", src, dst])
        )
    if have_aws():
        candidates.append(
            ("aws", ["aws", "s3", "cp", "--recursive", s3_uri.rstrip("/") + "/", dst])
        )
    last_err = None
    not_found = None
    verify_err = None
    if candidates:
        for attempt in range(max(1, int(retries))):
            result, last_err, not_found = run_cli_candidates(candidates)
            if result is not None:
                ok, reason = _download_verified(
                    local_dir,
                    required_files=required_files,
                    require_nonempty=require_nonempty,
                )
                if ok:
                    return
                verify_err = (
                    f"{result.tool} reported success but {reason} "
                    f"after downloading {s3_uri}"
                )
                last_err = verify_err
                time.sleep(1.0 * (attempt + 1))
                continue
            if not_found is not None:
                break
            time.sleep(1.0 * (attempt + 1))
    if not_found is not None:
        raise FileNotFoundError(f"S3 prefix does not exist: {s3_uri!r} ({not_found})")
    if last_err is not None:
        logger.warning(
            f"[arcstore] CLI download dir {s3_uri} -> {local_dir} failed "
            f"({last_err}); falling back to boto3."
        )
    try:
        _boto3_download_dir(s3_uri, local_dir)
    except Exception as e:
        if verify_err is not None:
            raise FileNotFoundError(
                f"S3 prefix did not produce a complete local dir: {s3_uri!r} "
                f"({verify_err})"
            ) from e
        raise
    ok, reason = _download_verified(
        local_dir,
        required_files=required_files,
        require_nonempty=require_nonempty,
    )
    if not ok:
        raise FileNotFoundError(f"S3 prefix did not produce a complete local dir: {s3_uri!r} ({reason})")


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
    found = False
    for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix.rstrip("/") + "/"):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(key_prefix.rstrip("/")) + 1:]
            if not rel:
                continue
            found = True
            dst = os.path.join(base, rel)
            os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
            client.download_file(bucket, obj["Key"], dst)
    if not found:
        raise FileNotFoundError(f"S3 prefix does not exist or is empty: {s3_uri!r}")
