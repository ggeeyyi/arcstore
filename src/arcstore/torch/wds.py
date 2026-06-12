"""WebDataset helpers for S3-native tar-shard reads.

Training hot paths default to **direct S3**: ``s3://`` shards become
``pipe:s5cmd cat ...`` even if the bucket is FUSE-mounted. Set
``read_policy="mount"`` (or ``ARCSTORE_WDS_READ_POLICY=mount``) for the
explicit compatibility path that opens mounted files directly.
"""
from __future__ import annotations

import fnmatch
import os
import re
import time
from collections.abc import Iterable, Sequence
from typing import Callable

from .._env import env_int
from .._env import read_policy as _read_policy
from ..io import list_prefix
from ..location import is_s3
from ..location import resolve

_BRACE_RANGE_RE = re.compile(r"\{(\d+)\.\.(\d+)(?:\.\.(-?\d+))?\}")
_GLOB_CHARS = set("*?[")


def _policy(value: str | None = None) -> str:
    return _read_policy(value, env_name="ARCSTORE_WDS_READ_POLICY", default="direct_s3")


def tar_url(prefix: str, name: str, *, read_policy: str | None = None) -> str:
    """URL for one tar shard ``name`` under ``prefix``, for webdataset."""
    loc = resolve(prefix)
    policy = _policy(read_policy)
    rp = loc.read_path()
    if loc.is_s3 and policy == "direct_s3":
        rp = None
    if rp is not None:
        return os.path.join(rp, name)
    return f"pipe:s5cmd cat {loc.s3_uri().rstrip('/')}/{name}"


def shard_urls(
    prefix: str,
    shard_keys: Sequence[str] | Iterable[str],
    *,
    name_prefix: str = "",
    suffix: str = ".tar",
    read_policy: str | None = None,
) -> list[str]:
    """Tar URLs for many shards; keys without ``suffix`` get it appended."""
    urls = []
    for key in shard_keys:
        name = key if key.endswith(suffix) else f"{name_prefix}{key}{suffix}"
        urls.append(tar_url(prefix, name, read_policy=read_policy))
    return urls


def _expand_brace_ranges(pattern: str) -> list[str]:
    match = _BRACE_RANGE_RE.search(pattern)
    if match is None:
        return [pattern]
    start_s, end_s, step_s = match.groups()
    start, end = int(start_s), int(end_s)
    step = int(step_s) if step_s is not None else (1 if end >= start else -1)
    if step == 0:
        raise ValueError(f"invalid zero step in brace range: {pattern!r}")
    if (end - start) * step < 0:
        step = -step
    width = max(len(start_s), len(end_s))
    stop = end + (1 if step > 0 else -1)
    out: list[str] = []
    for value in range(start, stop, step):
        repl = f"{value:0{width}d}"
        expanded = pattern[: match.start()] + repl + pattern[match.end() :]
        out.extend(_expand_brace_ranges(expanded))
    return out


def _split_prefix_name(pattern: str) -> tuple[str, str] | None:
    if "/" not in pattern:
        return None
    prefix, name = pattern.rsplit("/", 1)
    if not prefix or not name:
        return None
    return prefix, name


def _expand_one_pattern(pattern: str, *, read_policy: str | None = None) -> list[str]:
    if pattern.startswith("pipe:"):
        return [pattern]
    split = _split_prefix_name(pattern)
    if split is None:
        return [pattern]
    prefix, name_pattern = split

    if not any(ch in name_pattern for ch in _GLOB_CHARS):
        return shard_urls(prefix, [name_pattern], read_policy=read_policy)

    names: list[str] = []
    attempts = max(1, env_int("ARCSTORE_WDS_LIST_RETRIES", 4))
    for attempt in range(attempts):
        names = [
            name
            for name in list_prefix(prefix, read_policy=_policy(read_policy))
            if not name.endswith("/") and fnmatch.fnmatchcase(name, name_pattern)
        ]
        if names or not is_s3(prefix):
            break
        time.sleep(1.0 * (attempt + 1))
    return shard_urls(prefix, sorted(names), read_policy=read_policy)


def expand_urls(
    pattern: str | Sequence[str] | Iterable[str],
    *,
    read_policy: str | None = None,
    log_summary: bool = True,
    stage: bool | None = None,
) -> list[str]:
    """Expand WDS shard patterns to concrete WebDataset URLs.

    ``stage`` is accepted for compatibility with older project facades; arcstore
    no longer stages WDS shards by default because the Koala/AWS production
    path is direct S3 streaming.
    """
    _ = stage, log_summary
    patterns = [pattern] if isinstance(pattern, str) else list(pattern)
    urls: list[str] = []
    for item in patterns:
        for expanded in _expand_brace_ranges(str(item)):
            urls.extend(_expand_one_pattern(expanded, read_policy=read_policy))
    if not urls:
        raise FileNotFoundError(f"No WebDataset shard URLs matched {pattern!r}")
    return urls


def build_wds_dataset(
    pattern: str | Sequence[str] | Iterable[str],
    *,
    read_policy: str | None = None,
    shuffle_shards: bool = True,
    shard_shuffle: int = 100,
    sample_shuffle: int = 0,
    sample_shuffle_initial: int | None = None,
    sample_map: Callable | None = None,
):
    """Build a generic WebDataset ``DataPipeline`` from a shard pattern.

    The returned pipeline handles multi-node and DataLoader-worker sharding via
    ``split_by_node`` and ``split_by_worker``. Consumers provide ``sample_map``
    when they need to decode project-specific tar members.
    """
    import webdataset as wds

    urls = expand_urls(pattern, read_policy=read_policy)
    pipeline: list = [wds.SimpleShardList(urls)]
    if shuffle_shards and shard_shuffle > 0:
        pipeline.append(wds.shuffle(shard_shuffle))
    pipeline.extend([wds.split_by_node, wds.split_by_worker, wds.tarfile_to_samples()])
    if sample_shuffle > 0:
        kwargs = {}
        if sample_shuffle_initial is not None:
            kwargs["initial"] = sample_shuffle_initial
        pipeline.append(wds.shuffle(sample_shuffle, **kwargs))
    if sample_map is not None:
        pipeline.append(wds.map(sample_map))
    return wds.DataPipeline(*pipeline)


__all__ = [
    "build_wds_dataset",
    "expand_urls",
    "shard_urls",
    "tar_url",
]
