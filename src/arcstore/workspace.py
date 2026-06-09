"""Experiment/work-dir split: one ``s3://`` URI as the durable home, a
deterministic local NVMe mirror for fast writes.

Lifted from CausalVideoDiffusion ``src/utils/cli.py::split_expdir_io``.
"""
from __future__ import annotations

import os

from ._env import env_str
from .location import is_s3

#: Local scratch root used when the workdir is itself an ``s3://`` URI.
DEFAULT_LOCAL_ROOT = "/local-ssd/arcstore/workdirs"


def split_workdir(workdir: str, *, local_root: str | None = None) -> tuple[str, str | None]:
    """Resolve a workdir into ``(local_workdir, s3_workdir)``.

    A single ``workdir = s3://bucket/key`` declares the durable S3 home; a
    deterministic local mirror ``<local_root>/<bucket>/<key>`` is derived so
    the fast local filesystem backs every write before it is pushed to S3:

    * ``local_workdir`` — what the training/inference process writes to.
    * ``s3_workdir`` — the original ``s3://`` URI: upload mirror, resume
      source, default parent for derived outputs.

    A plain local workdir is returned unchanged with ``s3_workdir=None``.
    ``local_root`` resolution: argument > ``ARCSTORE_LOCAL_ROOT`` env >
    :data:`DEFAULT_LOCAL_ROOT`.
    """
    if not is_s3(workdir):
        return workdir, None
    root = local_root or env_str("ARCSTORE_LOCAL_ROOT", DEFAULT_LOCAL_ROOT)
    s3_uri = workdir.rstrip("/")
    key = s3_uri[len("s3://"):]
    return os.path.join(root, key), s3_uri
