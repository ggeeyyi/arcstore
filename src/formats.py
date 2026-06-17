"""Dataset format detection from path morphology (no I/O on direct S3).

Lifted from CausalVideoDiffusion ``src/dataset/factory.py::_detect_default_format``
and made mount-aware: a FUSE-mounted bucket is inspected like a local
directory, which notably makes LMDB-on-S3 legal when (and only when) the
bucket is mounted — read-only mmap through FUSE works.
"""
from __future__ import annotations

import glob as _glob
import os

from .location import resolve


def _has_lmdb_payload(local_dir: str) -> bool:
    if os.path.isfile(os.path.join(local_dir, "data.mdb")):
        return True
    return bool(_glob.glob(os.path.join(local_dir, "shard*", "data.mdb")))


def detect_format(path: str) -> str:
    """Classify a dataset path into ``{"jsonl", "wds", "scatter", "lmdb"}``.

    * ``*.jsonl``                          -> ``jsonl`` (manifest of ``.pt``)
    * ``*.tar`` / ``shards`` / glob chars  -> ``wds``   (tar shards)
    * dir with ``data.mdb`` (local/mounted)-> ``lmdb``
    * dir of ``*.pt`` (local/mounted)      -> ``scatter``
    * unmounted ``s3://`` prefix           -> ``scatter`` (the only
      streamable per-sample S3 layout; LMDB cannot be mmap'd from S3)
    * otherwise (local)                    -> ``lmdb`` (back-compat default)
    """
    p = path.lower().rstrip("/\\")
    if p.endswith(".jsonl"):
        return "jsonl"
    base = os.path.basename(p)
    if (
        base == "shards"
        or p.endswith("shards")
        or p.endswith(".tar")
        or any(c in p for c in "*{[")
    ):
        return "wds"

    loc = resolve(path)
    rp = loc.read_path()
    if rp is not None and os.path.isdir(rp):
        if _has_lmdb_payload(rp):
            return "lmdb"
        if _glob.glob(os.path.join(rp, "*.pt")):
            return "scatter"
        return "scatter" if loc.is_s3 else "lmdb"
    if loc.is_s3:
        return "scatter"
    return "lmdb"
