"""Concrete dataset backends for :func:`arcstore.open_dataset`.

Importing this package registers the torch-dependent backends (scatter, wds,
mds, synthetic) into :data:`arcstore.data.registry._BACKENDS`.
``open_dataset`` imports it best-effort before dispatching.
"""
from __future__ import annotations

from . import mosaic_backend  # noqa: F401  (registers "mds")
from . import scatter_backend  # noqa: F401  (registers "scatter")
from . import synthetic_backend  # noqa: F401  (registers "synthetic")
from . import wds_backend  # noqa: F401  (registers "wds")

__all__ = ["mosaic_backend", "scatter_backend", "synthetic_backend", "wds_backend"]
