"""Compatibility exports for WebDataset helpers.

Historically some projects imported ``arcstore.webdataset.expand_urls``.
The implementation now lives in :mod:`arcstore.torch.wds` because it is part
of the torch/data-loading extension surface.
"""
from __future__ import annotations

from .torch.wds import build_wds_dataset, expand_urls, shard_urls, tar_url

__all__ = [
    "build_wds_dataset",
    "expand_urls",
    "shard_urls",
    "tar_url",
]
