"""Importing this package registers the four torch checkpoint backends.

:mod:`arcstore.checkpoint.registry` triggers this import best-effort so that
``save_checkpoint`` / ``load_checkpoint`` can dispatch ``full_state`` /
``accelerate`` / ``deepspeed`` / ``blob`` / ``safetensors`` kinds.
"""
from __future__ import annotations

from . import (  # noqa: F401
    accelerate_backend,
    blob_backend,
    full_state_backend,
    safetensors_backend,
)

__all__ = [
    "accelerate_backend",
    "blob_backend",
    "full_state_backend",
    "safetensors_backend",
]
