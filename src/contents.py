"""Transparent S3 persistence by local *path*: yield a filesystem path, upload on exit.

Ported from ``arc_toolkit.contents``. :class:`ContentsManager` complements the
streaming primitives in :mod:`arcstore.io` (:func:`open_read` / :func:`open_write`
yield *file handles*) for the common case where a third-party save/load API only
accepts a *path* (``model.save_pretrained(dir)``, ``torch.save(obj, path)``,
``cv2.imwrite(path)``, ...).

* write mode (``"w"`` / ``"wb"``) — yields a local scratch path; on a clean
  exit of the ``with`` block the file/dir is uploaded to S3 in the background
  (non-blocking, flushed by :func:`arcstore.wait_for_uploads`).
* read mode (``"r"`` / ``"rb"``) — downloads the object from S3 first (S3 API,
  never the mount), then yields the local path.

Local paths pass through unchanged in both directions. The scratch root
defaults to :func:`arcstore._env.cache_dir` (``$ARCSTORE_CACHE_DIR`` >
``/local-ssd/arcstore`` > tempdir); nothing is created until an S3 URI is
actually opened.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from ._env import cache_dir
from .io import download_file
from .location import is_s3, split_s3
from .uploads import put_async

logger = logging.getLogger(__name__)

__all__ = ["ContentsManager", "local_mirror_path"]


def local_mirror_path(local_cache: str | Path, s3_uri: str) -> Path:
    """Map ``s3://bucket/prefix/key`` -> ``local_cache/bucket/prefix/key``."""
    if not is_s3(s3_uri):
        raise ValueError(f"local_mirror_path expects an s3:// URI, got {s3_uri!r}")
    bucket, key = split_s3(s3_uri)
    return Path(local_cache) / bucket / key


class ContentsManager:
    """Map S3 URIs to local scratch paths; background-upload writes on context exit.

    ``local_cache`` defaults to :func:`arcstore._env.cache_dir` under the
    ``contents`` subdir. ``keep_local=False`` removes the local copy after a
    successful upload (the upload is still asynchronous; deletion happens only
    once the transfer lands).
    """

    def __init__(self, local_cache: str | Path | None = None, keep_local: bool = True):
        self.local_cache = (
            Path(local_cache) if local_cache else cache_dir("contents", create=False)
        )
        self.keep_local = keep_local

    @contextmanager
    def open(self, uri: str, mode: Literal["w", "wb", "r", "rb"] = "w") -> Iterator[str]:
        """Yield a local filesystem path suitable for any save/load API.

        See the class docstring for the write/read/local semantics.
        """
        if mode not in ("w", "wb", "r", "rb"):
            raise ValueError(f"ContentsManager.open supports w/wb/r/rb, got {mode!r}")
        if not is_s3(uri):
            p = Path(uri)
            if mode.startswith("w"):
                p.parent.mkdir(parents=True, exist_ok=True)
            yield str(p)
            return

        local_path = local_mirror_path(self.local_cache, uri)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if mode.startswith("r"):
            if not local_path.exists():
                download_file(uri, str(local_path), label="arcstore-contents")
            yield str(local_path)
            return

        yield str(local_path)
        if local_path.exists():
            put_async(str(local_path), uri, keep_local=self.keep_local)
            logger.debug("[arcstore-contents] queued upload %s -> %s", local_path, uri)
