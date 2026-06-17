"""Dataset access-mode resolution: the single place that decides whether a
dataset path is read from the local filesystem, a FUSE-mounted S3 bucket, or
streamed directly from S3.

This centralizes the per-backend logic that previously lived inline in
:class:`arcstore.torch.scatter.ScatterPtDataset` and
:mod:`arcstore.torch.wds`. Every dataset backend consults
:func:`resolve_dataset_access` so the mount/local/direct-S3 decision is made
once, consistently, and honors the same ``read_policy`` knobs as the rest of
arcstore.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .._env import read_policy as _read_policy
from ..location import resolve

#: Access modes returned by :func:`resolve_dataset_access`.
LOCAL = "local"
MOUNT = "mount"
DIRECT_S3 = "direct_s3"


@dataclass(frozen=True)
class DatasetAccess:
    """How a dataset path should be read.

    * ``mode == "local"``     — a plain local directory; ``local_dir`` set.
    * ``mode == "mount"``     — an ``s3://`` source read through its FUSE
      mount; ``local_dir`` is the on-disk mount path, ``s3_uri`` the source.
    * ``mode == "direct_s3"`` — an ``s3://`` source streamed via the S3 API
      (s3torchconnector / ``pipe:s5cmd cat``); only ``s3_uri`` set.
    """

    mode: str
    local_dir: str | None = None
    s3_uri: str | None = None

    @property
    def is_local_read(self) -> bool:
        """True when reads come off the filesystem (local dir or mount)."""
        return self.local_dir is not None


def resolve_dataset_access(
    path: str,
    *,
    read_policy: str | None = None,
    env_name: str = "ARCSTORE_DATA_READ_POLICY",
    default: str = "auto",
) -> DatasetAccess:
    """Classify ``path`` into a :class:`DatasetAccess` decision.

    Rules (matching the ``ScatterPtDataset`` / ``tar_url`` behavior, lifted
    into one place):

    * a local filesystem path -> ``local``;
    * an ``s3://`` URI with ``read_policy`` resolving to ``direct_s3`` ->
      ``direct_s3``;
    * an ``s3://`` URI with ``read_policy`` in ``{auto, mount}`` (the default
      is ``auto``) whose bucket is listed in ``ARCSTORE_S3_MOUNTS`` AND whose
      mount directory exists -> ``mount`` (reads go through the local mount
      path, e.g. ``/threed-code/...``);
    * any other ``s3://`` URI -> ``direct_s3``.

    So when a bucket is FUSE-mounted the default is to read it as a local
    path; set ``read_policy="direct_s3"`` (or ``ARCSTORE_DATA_READ_POLICY=
    direct_s3``) to stream from S3 even when a mount exists.
    """
    loc = resolve(path)
    if not loc.is_s3:
        return DatasetAccess(LOCAL, local_dir=loc.local_path)

    policy = _read_policy(read_policy, env_name=env_name, default=default)
    rp = loc.read_path()  # <mount_root>/key, or None when not mounted
    if policy in ("auto", "mount") and rp is not None and os.path.isdir(rp):
        return DatasetAccess(MOUNT, local_dir=rp, s3_uri=loc.s3_uri())
    return DatasetAccess(DIRECT_S3, s3_uri=loc.s3_uri())


__all__ = [
    "DIRECT_S3",
    "LOCAL",
    "MOUNT",
    "DatasetAccess",
    "resolve_dataset_access",
]
