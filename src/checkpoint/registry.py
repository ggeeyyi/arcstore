"""Backend registry + the unified :func:`save_checkpoint` / :func:`load_checkpoint`
entry points.

Checkpoint operations differ a lot in argument shape (DCP needs models +
optimizers, Accelerate needs an ``accelerator``, a plain blob is just a dict,
safetensors returns a state_dict), so the unified functions are thin
dispatchers: the caller picks a ``kind`` explicitly (no auto-detection) and the
registered backend forwards to the existing implementation.

Backend availability:

* ``full_state`` / ``accelerate`` / ``blob`` / ``safetensors`` are registered
  by importing :mod:`arcstore.torch.ckpt_backends` (requires the
  ``arcstore[torch]`` extra). The dispatchers trigger that import best-effort
  before looking up the backend, so a missing torch surfaces as a clear error.

Raw DeepSpeed engines and "find latest + resume" orchestration are served by
:class:`arcstore.torch.CheckpointManager`, not by these stateless dispatchers.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

#: kind -> save callable(path, **kwargs)
_SAVE: dict[str, Callable[..., Any]] = {}
#: kind -> load callable(path, **kwargs)
_LOAD: dict[str, Callable[..., Any]] = {}


def register_checkpoint_backend(
    kind: str,
    *,
    save: Callable[..., Any] | None = None,
    load: Callable[..., Any] | None = None,
) -> None:
    """Register ``save`` and/or ``load`` callables for checkpoint ``kind``.

    Either side may be omitted (e.g. a load-only kind). Re-registering a kind
    overwrites the previous callable on the provided side(s).
    """
    if save is not None:
        _SAVE[kind] = save
    if load is not None:
        _LOAD[kind] = load


def available_checkpoint_kinds() -> list[str]:
    """Sorted union of kinds that have a save and/or load backend registered."""
    _ensure_backends()
    return sorted(set(_SAVE) | set(_LOAD))


def _ensure_backends() -> None:
    """Best-effort import that registers the torch checkpoint backends."""
    try:
        import arcstore.torch.ckpt_backends  # noqa: F401
    except ImportError:
        logger.debug("[arcstore] torch checkpoint backends unavailable; torch missing?")


def save_checkpoint(path: str, kind: str, **backend_kwargs: Any) -> Any:
    """Save a checkpoint of ``kind`` to ``path``.

    ``kind`` is explicit (no auto-detection):

    * ``"full_state"`` — FSDP DCP full training state; requires ``models`` and
      ``optimizers`` (plus optional ``step`` / ``scheduler`` / ``ema`` / ...).
    * ``"accelerate"`` — Accelerate full state (incl. its DeepSpeed plugin);
      requires ``accelerator``; an ``s3://`` ``path`` also needs ``local_dir``.
    * ``"blob"`` — a single ``torch.save`` object; requires ``obj``.
    * ``"safetensors"`` — exports full weights; requires ``model``.
    """
    _ensure_backends()
    fn = _SAVE.get(kind)
    if fn is None:
        raise ValueError(
            f"[arcstore] save_checkpoint: no save backend for kind {kind!r} "
            f"(path={path!r}); available: {sorted(_SAVE)}"
        )
    return fn(path, **backend_kwargs)


def load_checkpoint(path: str, kind: str, **backend_kwargs: Any) -> Any:
    """Load a checkpoint of ``kind`` from ``path``.

    ``kind`` is explicit (no auto-detection):

    * ``"full_state"`` — requires ``models`` and ``optimizers``; returns the
      global ``step`` (or the meta dict with ``return_meta=True``).
    * ``"accelerate"`` — requires ``accelerator``; returns the parsed step.
    * ``"blob"`` — returns the loaded object dict.
    * ``"safetensors"`` — returns a CPU ``state_dict``.
    """
    _ensure_backends()
    fn = _LOAD.get(kind)
    if fn is None:
        raise ValueError(
            f"[arcstore] load_checkpoint: no load backend for kind {kind!r} "
            f"(path={path!r}); available: {sorted(_LOAD)}"
        )
    return fn(path, **backend_kwargs)


__all__ = [
    "available_checkpoint_kinds",
    "load_checkpoint",
    "register_checkpoint_backend",
    "save_checkpoint",
]
