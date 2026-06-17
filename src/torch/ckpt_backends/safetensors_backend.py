"""Safetensors weights backend for the unified checkpoint interface.

* load -> :func:`arcstore.torch.safetensors.load_safetensors_auto` (run:ai
  streamer for direct S3, fast path for local/mount).
* save -> :func:`arcstore.torch.safetensors.save_safetensors_weights` (gathers
  the full state dict via DCP utilities; rank 0 writes + optionally uploads).
"""
from __future__ import annotations

from typing import Any

from ...checkpoint.registry import register_checkpoint_backend
from ...location import is_s3
from ..safetensors import load_safetensors_auto, save_safetensors_weights


def _save(
    path: str,
    *,
    model: Any,
    state_dict: Any = None,
    workers: int | None = None,
) -> str | None:
    """Export full safetensors weights to ``path`` (local dir/file or ``s3://``).

    Collective when ``state_dict`` is omitted (gathered via DCP utils on all
    ranks); pass ``state_dict=accelerator.get_state_dict(model)`` for DeepSpeed
    ZeRO-3 / accelerate.
    """
    if is_s3(path):
        from ..._env import cache_dir

        stage = str(cache_dir("safetensors-save"))
        return save_safetensors_weights(
            model, stage, state_dict=state_dict, s3_prefix=path, workers=workers
        )
    return save_safetensors_weights(model, path, state_dict=state_dict, workers=workers)


def _load(
    path: str,
    *,
    concurrency: int | None = None,
    memory_limit: str | None = None,
    read_policy: str | None = None,
) -> dict:
    return load_safetensors_auto(
        path,
        concurrency=concurrency,
        memory_limit=memory_limit,
        read_policy=read_policy,
    )


register_checkpoint_backend("safetensors", save=_save, load=_load)

__all__ = ["_load", "_save"]
