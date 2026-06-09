"""S3-native safetensors loading (run:ai Model Streamer) with mount rewrite.

Lifted from CausalVideoDiffusion ``src/utils/streamer_load.py``.
``load_safetensors_auto`` (the old ``load_safetensors_via_node_cache``) is
mount-aware: an ``s3://`` source whose bucket is FUSE-mounted is rewritten
to the mount path and takes the local branch — rank 0 streams, sibling
ranks mmap the same FUSE-backed files and the kernel page cache dedups, so
there is no per-rank S3 read amplification.

Critical: ``RUNAI_STREAMER_MEMORY_LIMIT`` MUST be bounded. ``-1``
(unlimited) deadlocks on multi-shard reads because the streamer tries to
allocate a single buffer == total bytes. We default to 32 GiB and force-set
the env BEFORE the streamer import (the C++ layer captures it at init).
"""
from __future__ import annotations

import logging
import os
import time

import torch

from ..io import glob_files
from ..location import is_s3 as _is_s3
from ..location import resolve

logger = logging.getLogger(__name__)

#: 32 GiB as a string (the run:ai streamer reads the env var as text).
DEFAULT_MEMORY_LIMIT = "34359738368"

#: Default per-streamer concurrency (ARC's reference default).
DEFAULT_CONCURRENCY = 32


def _force_streamer_envs(concurrency: int) -> None:
    """OVERWRITE the streamer env vars unconditionally.

    Why not ``setdefault``: a leftover ``RUNAI_STREAMER_MEMORY_LIMIT=-1`` in
    the container env would silently keep the documented deadlock.
    """
    os.environ["RUNAI_STREAMER_CONCURRENCY"] = str(concurrency)
    os.environ["RUNAI_STREAMER_MEMORY_LIMIT"] = DEFAULT_MEMORY_LIMIT


def _list_safetensors(prefix: str) -> list[str]:
    """Sorted ``*.safetensors`` files under ``prefix`` (file, dir, or s3)."""
    if prefix.endswith(".safetensors"):
        return [prefix]

    try:
        from runai_model_streamer import list_safetensors as _runai_list

        files = sorted(_runai_list(prefix.rstrip("/")))
        if not files:
            raise FileNotFoundError(f"no *.safetensors under {prefix!r}")
        return files
    except ImportError:
        pass  # fall through for dev/test environments without runai

    files = glob_files(prefix, ".safetensors")
    if not files:
        raise FileNotFoundError(f"No *.safetensors under {prefix!r}")
    return files


def load_safetensors_streamer(
    uri_or_dir: str,
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> dict[str, torch.Tensor]:
    """Return a CPU state_dict by streaming ``*.safetensors`` from ``uri_or_dir``.

    Tries run:ai Model Streamer first; falls back to
    ``safetensors.torch.load_file`` for **local** paths. ``s3://`` inputs
    require ``runai-model-streamer-s3``.
    """
    files = _list_safetensors(uri_or_dir)
    n_files = len(files)
    log_target = files[0] if n_files == 1 else f"{n_files} shards in {uri_or_dir}"
    logger.info(
        f"[arcstore-streamer] loading {log_target} (concurrency={concurrency}, "
        f"memory_limit={DEFAULT_MEMORY_LIMIT} bytes)"
    )

    # Force-set envs BEFORE the streamer import — the C++ runtime captures
    # them on first SafetensorsStreamer() instantiation.
    _force_streamer_envs(concurrency)

    t0 = time.perf_counter()
    try:
        from runai_model_streamer import SafetensorsStreamer
    except ImportError as e:
        if any(_is_s3(f) for f in files):
            raise RuntimeError(
                "runai-model-streamer not installed but source is s3://. "
                "Install with: pip install 'arcstore[torch]'"
            ) from e
        from safetensors.torch import load_file

        out: dict[str, torch.Tensor] = {}
        for f in files:
            out.update(load_file(f, device="cpu"))
        dt = time.perf_counter() - t0
        n_bytes = sum(v.numel() * v.element_size() for v in out.values())
        logger.info(
            f"[arcstore-streamer-fallback] loaded {n_bytes / 1024**3:.2f} GiB "
            f"({len(out)} tensors) in {dt:.2f}s via safetensors.load_file"
        )
        return out

    out = {}
    with SafetensorsStreamer() as streamer:
        streamer.stream_files(files)
        for name, tensor in streamer.get_tensors():
            # ``clone`` because the streamer's internal buffer is reused
            # across yields.
            out[name] = tensor.clone().detach()
    dt = time.perf_counter() - t0
    n_bytes = sum(v.numel() * v.element_size() for v in out.values())
    rate = n_bytes / 1024**3 / dt if dt > 0 else 0.0
    logger.info(
        f"[arcstore-streamer] loaded {n_bytes / 1024**3:.2f} GiB ({len(out)} "
        f"tensors) in {dt:.2f}s ({rate:.2f} GiB/s, concurrency={concurrency})"
    )
    return out


def _load_file_mmap(local_path: str) -> dict[str, torch.Tensor]:
    """mmap-backed safetensors load — cheap from many ranks on one node."""
    from safetensors.torch import load_file

    return load_file(local_path, device="cpu")


def load_safetensors_auto(
    uri_or_dir: str,
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> dict[str, torch.Tensor]:
    """Load safetensors choosing the best path per source.

    * **direct ``s3://``** — every rank streams via run:ai Model Streamer.
    * **mounted ``s3://``** — rewritten to the mount path, then handled as
      local (page-cache shared across ranks, no S3 read amplification).
    * **local file/dir** — rank 0 uses the streamer (fast); other local
      ranks mmap the same on-disk shards. No extra copy.
    """
    loc = resolve(uri_or_dir)
    rp = loc.read_path()
    if loc.is_s3 and rp is not None and os.path.exists(rp):
        logger.info(f"[arcstore-streamer] using mounted path {rp} for {uri_or_dir}")
        uri_or_dir = rp
    elif loc.is_s3:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        logger.info(
            f"[arcstore-streamer] rank{local_rank} loading S3 safetensors "
            f"directly from {uri_or_dir}"
        )
        return load_safetensors_streamer(uri_or_dir, concurrency=concurrency)

    # Local (or mounted) source: rank 0 uses the streamer; others mmap.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank == 0:
        return load_safetensors_streamer(uri_or_dir, concurrency=concurrency)
    files = _list_safetensors(uri_or_dir)
    out: dict[str, torch.Tensor] = {}
    for f in files:
        out.update(_load_file_mmap(f))
    logger.info(
        f"[arcstore-mmap] rank{local_rank} loaded {len(out)} tensors "
        f"from {len(files)} file(s) under {uri_or_dir}"
    )
    return out


__all__ = [
    "DEFAULT_CONCURRENCY",
    "DEFAULT_MEMORY_LIMIT",
    "load_safetensors_auto",
    "load_safetensors_streamer",
]
