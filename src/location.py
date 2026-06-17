"""Path resolution: the single place that understands schemes and S3 mounts.

Every arcstore API takes a plain string path that is either a local
filesystem path or an ``s3://bucket/key`` URI. :func:`resolve` classifies it
into a :class:`Location` that all read primitives consult.

Mount table
-----------

Pods sometimes FUSE-mount S3 buckets (mountpoint-s3), e.g.::

    ARCSTORE_S3_MOUNTS="arcwm-code-us-west-2=/threed-code,arcwm-asset-us-west-2=/asset"

When a bucket appears in the table AND mounts are enabled
(``ARCSTORE_USE_MOUNTS`` != 0) AND the mount directory actually exists,
``Location.read_path()`` translates ``s3://bucket/key`` to
``<mountdir>/<key>`` so reads go through the kernel page cache instead of
the S3 API.

Writes NEVER use the mount: mountpoint-s3 rejects overwrites and restricts
rename, so every write primitive in arcstore goes local-write + S3-API push.
This is structural — no write code path ever consults ``mount_root``.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass

from ._env import env_bool, env_opt

logger = logging.getLogger(__name__)

_S3_PREFIX = "s3://"

_MOUNT_LOCK = threading.Lock()
_MOUNT_TABLE: dict[str, str] | None = None  # bucket -> mount dir (raw, from env)
_MOUNT_OK: dict[str, bool] = {}  # bucket -> isdir() result cache


def is_s3(path) -> bool:
    """True iff ``path`` is an ``s3://`` URI."""
    return isinstance(path, str) and path.startswith(_S3_PREFIX)


def split_s3(s3_uri: str) -> tuple[str, str]:
    """Split ``s3://bucket/key`` into ``(bucket, key)``."""
    if not is_s3(s3_uri):
        raise ValueError(f"not an s3:// URI: {s3_uri!r}")
    bucket, _, key = s3_uri[len(_S3_PREFIX):].partition("/")
    return bucket, key


def _parse_mount_table() -> dict[str, str]:
    raw = env_opt("ARCSTORE_S3_MOUNTS")
    if not raw:
        return {}
    table: dict[str, str] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        bucket, sep, mountdir = entry.partition("=")
        bucket, mountdir = bucket.strip(), mountdir.strip()
        if not sep or not bucket or not mountdir:
            logger.warning(
                f"[arcstore] malformed ARCSTORE_S3_MOUNTS entry {entry!r}; skipped"
            )
            continue
        table[bucket] = mountdir.rstrip("/")
    return table


def refresh_mounts() -> None:
    """Re-read ``ARCSTORE_S3_MOUNTS`` and drop the isdir cache (tests, late env)."""
    global _MOUNT_TABLE
    with _MOUNT_LOCK:
        _MOUNT_TABLE = None
        _MOUNT_OK.clear()


def mount_root_for(bucket: str) -> str | None:
    """Mount directory for ``bucket`` if usable, else None.

    Usable = listed in ``ARCSTORE_S3_MOUNTS`` + ``ARCSTORE_USE_MOUNTS`` enabled
    + the mount directory exists (checked once per bucket per process; a
    missing dir silently falls back to direct S3 with one INFO log).
    """
    if not env_bool("ARCSTORE_USE_MOUNTS", True):
        return None
    global _MOUNT_TABLE
    with _MOUNT_LOCK:
        if _MOUNT_TABLE is None:
            _MOUNT_TABLE = _parse_mount_table()
        mountdir = _MOUNT_TABLE.get(bucket)
        if mountdir is None:
            return None
        ok = _MOUNT_OK.get(bucket)
        if ok is None:
            ok = os.path.isdir(mountdir)
            _MOUNT_OK[bucket] = ok
            if not ok:
                logger.info(
                    f"[arcstore] mount dir {mountdir} for bucket {bucket} not "
                    f"present; falling back to direct S3 reads."
                )
        return mountdir if ok else None


@dataclass(frozen=True)
class Location:
    """A classified path. Use :func:`resolve` to construct."""

    raw: str
    scheme: str  # "s3" | "file"
    bucket: str | None = None
    key: str | None = None
    local_path: str | None = None  # file scheme only
    mount_root: str | None = None  # s3 scheme only, when the bucket is mounted

    @property
    def is_s3(self) -> bool:
        return self.scheme == "s3"

    def read_path(self) -> str | None:
        """Local filesystem path this object can be READ from, or None.

        Never used for writes (mountpoint-s3 rejects overwrites). Does not
        check that the object itself exists — use :func:`arcstore.exists`.
        """
        if self.scheme == "file":
            return self.local_path
        if self.mount_root is not None:
            return f"{self.mount_root}/{self.key}" if self.key else self.mount_root
        return None

    def readable(self) -> str:
        """``read_path()`` when available, else the ``s3://`` URI."""
        p = self.read_path()
        return p if p is not None else self.s3_uri()

    def s3_uri(self) -> str:
        if self.scheme != "s3":
            raise ValueError(f"not an s3 location: {self.raw!r}")
        return self.raw


def resolve(path) -> Location:
    """Classify a path string into a :class:`Location`."""
    p = os.fspath(path)
    if is_s3(p):
        norm = p.rstrip("/") if p != _S3_PREFIX else p
        bucket, key = split_s3(norm)
        return Location(
            raw=norm,
            scheme="s3",
            bucket=bucket,
            key=key,
            mount_root=mount_root_for(bucket),
        )
    return Location(raw=p, scheme="file", local_path=p)
