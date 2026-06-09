"""Env-var helpers shared across arcstore modules.

All arcstore knobs use the ``ARCSTORE_`` prefix and are read at *call time*
(never at import time) so tests and late ``os.environ`` writes behave.
"""
from __future__ import annotations

import os

_FALSY = ("0", "false", "no", "off")


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
