"""WebDataset tar-URL helpers: pick the cheapest read path per shard.

* direct ``s3://`` -> ``pipe:s5cmd cat s3://.../<name>`` (one subprocess
  per shard, streaming);
* FUSE-mounted ``s3://`` -> the mount path as a plain file (webdataset
  opens it directly — saves the per-shard subprocess);
* local prefix -> plain join.
"""
from __future__ import annotations

import os
from typing import Iterable, Sequence

from ..location import resolve


def tar_url(prefix: str, name: str) -> str:
    """URL for one tar shard ``name`` under ``prefix``, for webdataset."""
    loc = resolve(prefix)
    rp = loc.read_path()
    if rp is not None:
        return os.path.join(rp, name)
    return f"pipe:s5cmd cat {loc.s3_uri().rstrip('/')}/{name}"


def shard_urls(
    prefix: str,
    shard_keys: Sequence[str] | Iterable[str],
    *,
    name_prefix: str = "",
    suffix: str = ".tar",
) -> list[str]:
    """Tar URLs for many shards; keys without ``suffix`` get it appended."""
    urls = []
    for key in shard_keys:
        name = key if key.endswith(suffix) else f"{name_prefix}{key}{suffix}"
        urls.append(tar_url(prefix, name))
    return urls
