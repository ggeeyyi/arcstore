"""Backend registry + the unified :func:`open_dataset` entry point.

``open_dataset`` classifies a dataset path with
:func:`arcstore.detect_format` (unless ``format=`` overrides it) and dispatches
to the registered backend for that format. Backends are plain callables
registered via :func:`register_backend`; they return a
``torch.utils.data.IterableDataset`` of decoded samples.

Backend availability:

* ``scatter`` / ``wds`` / ``mds`` / ``synthetic`` are registered by importing
  :mod:`arcstore.torch.backends` (requires the ``arcstore[torch]`` extra).
  ``open_dataset`` triggers that import best-effort before dispatching.
* ``jsonl`` / ``lmdb`` are registered here as not-yet-implemented
  placeholders that raise an informative :class:`NotImplementedError`.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from ..formats import detect_format

logger = logging.getLogger(__name__)

#: format string -> backend callable(path, **kwargs) -> IterableDataset
_BACKENDS: dict[str, Callable[..., Any]] = {}


def register_backend(fmt: str, backend: Callable[..., Any] | None = None):
    """Register ``backend`` for dataset ``fmt``.

    Usable as a decorator (``@register_backend("scatter")``) or a direct call
    (``register_backend("scatter", fn)``). Re-registering a format overwrites
    the previous backend (lets the torch backends shadow placeholders).
    """
    if backend is not None:
        _BACKENDS[fmt] = backend
        return backend

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        _BACKENDS[fmt] = fn
        return fn

    return deco


def available_backends() -> list[str]:
    """Sorted list of currently registered dataset formats."""
    return sorted(_BACKENDS)


def _unsupported(fmt: str, hint: str) -> Callable[..., Any]:
    def _raise(path: str, **_kwargs: Any):
        raise NotImplementedError(
            f"[arcstore] open_dataset: format {fmt!r} is not implemented yet "
            f"(path={path!r}). {hint}"
        )

    return _raise


# Reserved-but-unimplemented formats. Registered eagerly (no torch needed) so
# open_dataset gives a clear, actionable error instead of a KeyError.
register_backend(
    "jsonl",
    _unsupported(
        "jsonl",
        "Localize the manifest with arcstore.ensure_local_file(...) and parse "
        "it yourself for now.",
    ),
)
register_backend(
    "lmdb",
    _unsupported(
        "lmdb",
        "LMDB is only readable from a FUSE-mounted bucket or a local dir and "
        "has no loader yet; for direct S3 use a scatter .pt layout instead.",
    ),
)
register_backend(
    "mds",
    _unsupported(
        "mds",
        "Mosaic StreamingDataset (MDS) support requires importing "
        "arcstore.torch.backends and the arcstore[mosaic] extra.",
    ),
)


def _ensure_torch_backends() -> None:
    """Best-effort import that registers the scatter/wds torch backends."""
    try:
        import arcstore.torch.backends  # noqa: F401
    except ImportError:
        # torch (or a torch-layer dep) is missing. Leave _BACKENDS as-is; if
        # the requested format needs a torch backend, open_dataset surfaces a
        # clear error below.
        logger.debug("[arcstore] torch backends unavailable; torch not installed?")


def open_dataset(
    path: str,
    *,
    format: str | None = None,
    decode: Callable[[dict], Any] | None = None,
    read_policy: str | None = None,
    shuffle_buffer: int = 1000,
    length: int | None = None,
    region: str | None = None,
    **backend_kwargs: Any,
):
    """Open ``path`` as a dataset, dispatching by detected (or given) format.

    Returns a ``torch.utils.data.IterableDataset`` yielding decoded samples.
    Samples are normalized to WebDataset-style ``dict`` (``{"__key__": ...,
    "<ext>": bytes, ...}``); ``decode(sample) -> Any`` maps each dict to the
    final sample (``decode=None`` yields the raw dicts).

    Parameters mirror the per-backend constructors:

    * ``format`` — override :func:`arcstore.detect_format`.
    * ``read_policy`` — ``direct_s3`` / ``mount`` / ``auto`` (default from
      ``ARCSTORE_DATA_READ_POLICY``, itself defaulting to ``auto``).
    * ``shuffle_buffer`` — sample-level reservoir shuffle buffer.
    * ``length`` — artificial epoch length for ``len(ds)``.
    * ``region`` — AWS region for direct-S3 reads.
    * ``backend_kwargs`` — forwarded to the concrete backend.
    """
    _ensure_torch_backends()
    fmt = format or detect_format(path)
    backend = _BACKENDS.get(fmt)
    if backend is None:
        raise ValueError(
            f"[arcstore] open_dataset: no backend for format {fmt!r} "
            f"(path={path!r}); available: {available_backends()}"
        )
    return backend(
        path,
        decode=decode,
        read_policy=read_policy,
        shuffle_buffer=shuffle_buffer,
        length=length,
        region=region,
        **backend_kwargs,
    )


__all__ = [
    "available_backends",
    "open_dataset",
    "register_backend",
]
