"""FSDP DCP full-state backend for the unified checkpoint interface.

Adapts :func:`arcstore.torch.dcp.save_full_state` / ``load_full_state`` onto
the ``(path, **kwargs)`` dispatch shape. ``models`` and ``optimizers`` are
required; everything else is forwarded as-is.
"""
from __future__ import annotations

from typing import Any

from ...checkpoint.registry import register_checkpoint_backend
from ..dcp import load_full_state, save_full_state


def _save(
    path: str,
    *,
    models: Any,
    optimizers: Any,
    step: int = 0,
    scheduler: Any = None,
    ema: Any = None,
    extra_state: Any = None,
    side_files: Any = None,
    async_save: bool = False,
    thread_count: int = 8,
) -> None:
    return save_full_state(
        path,
        models,
        optimizers,
        step=step,
        scheduler=scheduler,
        ema=ema,
        extra_state=extra_state,
        side_files=side_files,
        async_save=async_save,
        thread_count=thread_count,
    )


def _load(
    path: str,
    *,
    models: Any,
    optimizers: Any,
    scheduler: Any = None,
    ema: Any = None,
    return_meta: bool = False,
):
    return load_full_state(
        path,
        models,
        optimizers,
        scheduler=scheduler,
        ema=ema,
        return_meta=return_meta,
    )


register_checkpoint_backend("full_state", save=_save, load=_load)

__all__ = ["_load", "_save"]
