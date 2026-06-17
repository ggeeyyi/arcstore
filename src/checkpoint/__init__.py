"""Unified checkpoint read/write layer.

``arcstore.save_checkpoint(path, kind, ...)`` / ``arcstore.load_checkpoint(
path, kind, ...)`` are the single entry points: the caller picks the checkpoint
``kind`` explicitly and the registry dispatches to the right backend (FSDP DCP
full-state, Accelerate/DeepSpeed, a plain ``.pt`` blob, or safetensors weights).

The dispatch + registry live in :mod:`arcstore.checkpoint.registry`; concrete
torch backends register themselves from :mod:`arcstore.torch.ckpt_backends`.
"""
from __future__ import annotations

from .registry import (
    available_checkpoint_kinds,
    load_checkpoint,
    register_checkpoint_backend,
    save_checkpoint,
)

__all__ = [
    "available_checkpoint_kinds",
    "load_checkpoint",
    "register_checkpoint_backend",
    "save_checkpoint",
]
