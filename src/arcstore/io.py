"""Mount-aware read primitives.

Reads consult :func:`arcstore.location.resolve`: a local path or a mounted
bucket reads straight off the filesystem; a direct ``s3://`` source goes
through s5cmd/boto3. Bulk transfers (:func:`download_file` /
:func:`download_dir`) always use the S3 API — multipart fan-out beats FUSE
for big objects; the mount only serves open-style reads.

Note on mount staleness: mountpoint-s3 caches directory listings, so a
mounted ``exists()`` can briefly miss an object that was just written.
``exists()`` therefore treats the mount as a fast-positive only and falls
through to a direct S3 check on a mount miss.
"""
from __future__ import annotations

import glob as _glob
import io as _io
import logging
import os
from typing import IO

from . import s3cli
from .location import resolve, split_s3
from .uploads import (  # noqa: F401  (re-exported write primitives)
    download_dir,
    track_future,
    upload_dir,
    upload_dir_async,
    upload_file,
    upload_file_async,
    wait_for_uploads,
)

logger = logging.getLogger(__name__)


def exists(path: str) -> bool:
    """True iff the file/object exists (local path, mounted read, or S3)."""
    loc = resolve(path)
    rp = loc.read_path()
    if rp is not None and os.path.exists(rp):
        return True
    if loc.is_s3:
        # Mount miss may be a stale FUSE listing — confirm against S3.
        return s3cli.head_object(loc.s3_uri()) is not None
    return False


def read_bytes(path: str) -> bytes:
    """Read the full content of a file/object."""
    loc = resolve(path)
    rp = loc.read_path()
    if rp is not None and os.path.isfile(rp):
        with open(rp, "rb") as f:
            return f.read()
    if loc.is_s3:
        import boto3

        bucket, key = split_s3(loc.s3_uri())
        body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"]
        return body.read()
    raise FileNotFoundError(path)


def open_read(path: str, mode: str = "rb") -> IO:
    """Open a file/object for reading (``"rb"`` or ``"r"``)."""
    if mode not in ("rb", "r"):
        raise ValueError(f"open_read supports 'rb'/'r', got {mode!r}")
    loc = resolve(path)
    rp = loc.read_path()
    if rp is not None and os.path.isfile(rp):
        return open(rp, mode)
    if loc.is_s3:
        raw = _io.BytesIO(read_bytes(path))
        if mode == "r":
            return _io.TextIOWrapper(raw, encoding="utf-8")
        return raw
    raise FileNotFoundError(path)


def list_prefix(path: str) -> list[str]:
    """Immediate children of a directory/prefix; subdirectories carry ``/``.

    Local and mounted sources use ``os.scandir``; direct S3 uses
    s5cmd/aws/boto3 listing. Missing dir/prefix returns ``[]``.
    """
    loc = resolve(path)
    rp = loc.read_path()
    if rp is not None and os.path.isdir(rp):
        out: list[str] = []
        with os.scandir(rp) as it:
            for de in it:
                out.append(de.name + "/" if de.is_dir(follow_symlinks=True) else de.name)
        return sorted(out)
    if loc.is_s3:
        return sorted(e.name for e in s3cli.ls_prefix(loc.s3_uri()))
    return []


def glob_files(path_or_prefix: str, suffix: str) -> list[str]:
    """Files under a dir/prefix ending in ``suffix``, sorted.

    Returns local paths when a filesystem read path is available (local dir
    or mounted bucket), ``s3://`` URIs otherwise — consistent with
    ``Location.readable()``.
    """
    loc = resolve(path_or_prefix)
    rp = loc.read_path()
    if rp is not None and os.path.isdir(rp):
        return sorted(_glob.glob(os.path.join(rp, f"*{suffix}")))
    if loc.is_s3:
        base = loc.s3_uri().rstrip("/")
        return sorted(
            f"{base}/{e.name}"
            for e in s3cli.ls_prefix(base)
            if not e.is_dir and e.name.endswith(suffix)
        )
    return []


def download_file(s3_uri: str, local_path: str, *, label: str = "arcstore") -> None:
    """Download one object via the S3 API (s5cmd -> aws -> boto3).

    Raises ``FileNotFoundError`` for a missing object. Always direct-S3,
    never the mount (multipart fan-out wins for big files).
    """
    s3cli.download_file(s3_uri, local_path, label=label)
