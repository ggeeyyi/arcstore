"""Run-level storage layout helpers.

``RunStorage`` turns one run root into predictable local and S3 locations:
``logs/``, ``artifacts/`` and ``checkpoints/``. It is intentionally thin and
uses the same local-write + S3-upload primitives as the rest of arcstore.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from dataclasses import field

from ._env import default_workers
from .location import is_s3
from .uploads import upload_dir, upload_dir_async, upload_file, upload_file_async
from .workspace import split_workdir

logger = logging.getLogger(__name__)


def _join(base: str | None, *parts: str) -> str | None:
    if base is None:
        return None
    out = base.rstrip("/")
    for part in parts:
        out += "/" + part.strip("/")
    return out


def sync_artifacts(
    local_dir: str,
    s3_uri: str | None,
    *,
    async_: bool = True,
    workers: int | None = None,
) -> None:
    """Upload a local artifact directory to S3 if it exists and is non-empty."""
    if not s3_uri:
        return
    if not is_s3(s3_uri):
        raise ValueError(f"sync_artifacts expects s3:// destination, got {s3_uri!r}")
    if not os.path.isdir(local_dir):
        return
    has_file = any(files for _root, _dirs, files in os.walk(local_dir))
    if not has_file:
        return
    if async_:
        upload_dir_async(local_dir, s3_uri)
    else:
        upload_dir(
            local_dir,
            s3_uri,
            workers=workers if workers is not None else default_workers(),
        )
    logger.info("[arcstore-run] synced artifacts %s -> %s", local_dir, s3_uri)


@dataclass(frozen=True)
class RunStorage:
    """Local/S3 layout for one training run."""

    root: str
    local_root: str | None = None

    def __post_init__(self):
        local, remote = split_workdir(self.root, local_root=self.local_root)
        object.__setattr__(self, "local_dir", local)
        object.__setattr__(self, "s3_dir", remote)

    local_dir: str = field(init=False, default="")
    s3_dir: str | None = field(init=False, default=None)

    @property
    def local_logs_dir(self) -> str:
        return os.path.join(self.local_dir, "logs")

    @property
    def local_artifacts_dir(self) -> str:
        return os.path.join(self.local_dir, "artifacts")

    @property
    def local_checkpoints_dir(self) -> str:
        return os.path.join(self.local_dir, "checkpoints")

    @property
    def logs_s3(self) -> str | None:
        return _join(self.s3_dir, "logs")

    @property
    def artifacts_s3(self) -> str | None:
        return _join(self.s3_dir, "artifacts")

    @property
    def checkpoints_s3(self) -> str | None:
        return _join(self.s3_dir, "checkpoints")

    def ensure_local_dirs(self) -> None:
        for path in (
            self.local_dir,
            self.local_logs_dir,
            self.local_artifacts_dir,
            self.local_checkpoints_dir,
        ):
            os.makedirs(path, exist_ok=True)

    def sync_artifacts(self, subdir: str | None = None, *, async_: bool = True) -> None:
        local = (
            self.local_artifacts_dir
            if subdir is None
            else os.path.join(self.local_artifacts_dir, subdir)
        )
        remote = self.artifacts_s3 if subdir is None else _join(self.artifacts_s3, subdir)
        sync_artifacts(local, remote, async_=async_)

    def upload_file(self, local_path: str, remote_rel: str, *, async_: bool = False) -> None:
        remote = _join(self.s3_dir, remote_rel)
        if remote is None:
            return
        if async_:
            upload_file_async(local_path, remote)
        else:
            upload_file(local_path, remote)


__all__ = ["RunStorage", "sync_artifacts"]
