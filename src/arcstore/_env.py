"""Env-var helpers shared across arcstore modules.

All arcstore knobs use the ``ARCSTORE_`` prefix and are read at *call time*
(never at import time) so tests and late ``os.environ`` writes behave.
"""
from __future__ import annotations

import os

_FALSY = ("0", "false", "no", "off")
LOCAL_SSD_ROOT = "/local-ssd"
DEFAULT_STREAMER_MEMORY_LIMIT = "34359738368"


def env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v else default


def env_opt(name: str) -> str | None:
    v = os.environ.get(name)
    return v if v else None


def env_bool(name: str, default: bool = True) -> bool:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v.strip().lower() not in _FALSY


def env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if not v:
        return default
    try:
        return int(float(v))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if not v:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def aws_region() -> str:
    return (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "us-west-2"
    )


def default_workers() -> int:
    """Default ``--numworkers`` for s5cmd transfers."""
    return env_int("ARCSTORE_S5CMD_WORKERS", 32)


def streamer_concurrency(default: int = 32) -> int:
    """Default run:ai Model Streamer concurrency."""
    return env_int("ARCSTORE_STREAMER_CONCURRENCY", default)


def streamer_memory_limit(default: str = DEFAULT_STREAMER_MEMORY_LIMIT) -> str:
    """Bounded run:ai Model Streamer memory limit."""
    return env_str("ARCSTORE_STREAMER_MEMORY_LIMIT", default)


def read_policy(
    value: str | None = None,
    *,
    env_name: str = "ARCSTORE_READ_POLICY",
    default: str = "auto",
) -> str:
    """Normalize S3 read policy: ``direct_s3``, ``mount`` or ``auto``.

    ``direct_s3`` means ignore FUSE mounts for S3 URIs. This is the
    preferred training hot-path policy on Koala/AWS. ``mount`` means use a
    configured mount when present and fall back to direct S3 otherwise.
    ``auto`` preserves the historical behavior of using mounts when they
    exist.
    """
    raw = value or os.environ.get(env_name) or os.environ.get("ARCSTORE_READ_POLICY") or default
    norm = raw.strip().lower().replace("-", "_")
    if norm in ("direct", "s3", "s3_native", "native"):
        return "direct_s3"
    if norm in ("fuse", "mounted"):
        return "mount"
    if norm in ("direct_s3", "mount", "auto"):
        return norm
    return default


def _local_ssd_usable(root: str = LOCAL_SSD_ROOT) -> bool:
    """True when Koala's local NVMe root exists and can host scratch dirs."""
    return os.path.isdir(root) and os.access(root, os.W_OK | os.X_OK)


def local_ssd_or_tmp(local_ssd_path: str, tmp_path: str) -> str:
    """Prefer a /local-ssd path, with a /tmp fallback for non-Koala hosts."""
    return local_ssd_path if _local_ssd_usable() else tmp_path
