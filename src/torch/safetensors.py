"""S3-native safetensors loading (run:ai Model Streamer) with mount rewrite.

Lifted from CausalVideoDiffusion ``src/utils/streamer_load.py``.
``load_safetensors_auto`` is mount-aware: an ``s3://`` source whose bucket
is FUSE-mounted is rewritten to the mount path and **every rank just
mmaps the on-disk shards directly** (kernel page cache dedups across
ranks; no S3 read amplification).

Why the streamer is NOT used on FUSE-mounted paths
--------------------------------------------------
Empirically observed deadlock: when ``runai-model-streamer`` is pointed
at a path under a ``mountpoint-s3`` mount with concurrency=32, the
streamer's many concurrent range reads can hang the FUSE layer
indefinitely (no error, no progress). The streamer is designed for
direct ``s3://`` reads — let it do that, and let mmap handle FUSE.

Resulting decision matrix:

* direct ``s3://`` (no mount)   — every rank uses run:ai streamer
* ``s3://`` rewritten to mount  — every rank mmaps the mount path
* plain local file/dir          — rank 0 uses streamer, siblings mmap

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

from .._env import read_policy as _read_policy
from .._env import streamer_concurrency, streamer_memory_limit
from ..io import glob_files
from ..location import is_s3 as _is_s3
from ..location import resolve

logger = logging.getLogger(__name__)

#: 32 GiB as a string (the run:ai streamer reads the env var as text).
DEFAULT_MEMORY_LIMIT = "34359738368"

#: Default per-streamer concurrency (ARC's reference default).
DEFAULT_CONCURRENCY = 32


def _force_streamer_envs(concurrency: int, memory_limit: str) -> None:
    """OVERWRITE the streamer env vars unconditionally.

    Why not ``setdefault``: a leftover ``RUNAI_STREAMER_MEMORY_LIMIT=-1`` in
    the container env would silently keep the documented deadlock.
    """
    os.environ["RUNAI_STREAMER_CONCURRENCY"] = str(concurrency)
    os.environ["RUNAI_STREAMER_MEMORY_LIMIT"] = str(memory_limit)


def _list_safetensors(prefix: str, *, read_policy: str | None = None) -> list[str]:
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

    files = glob_files(prefix, ".safetensors", read_policy=read_policy)
    if not files:
        raise FileNotFoundError(f"No *.safetensors under {prefix!r}")
    return files


def load_safetensors_streamer(
    uri_or_dir: str,
    *,
    concurrency: int | None = None,
    memory_limit: str | None = None,
) -> dict[str, torch.Tensor]:
    """Return a CPU state_dict by streaming ``*.safetensors`` from ``uri_or_dir``.

    Tries run:ai Model Streamer first; falls back to
    ``safetensors.torch.load_file`` for **local** paths. ``s3://`` inputs
    require ``runai-model-streamer-s3``.
    """
    files = _list_safetensors(uri_or_dir, read_policy="direct_s3")
    concurrency = int(
        concurrency
        if concurrency is not None
        else streamer_concurrency(DEFAULT_CONCURRENCY)
    )
    memory_limit = memory_limit or streamer_memory_limit(DEFAULT_MEMORY_LIMIT)
    n_files = len(files)
    log_target = files[0] if n_files == 1 else f"{n_files} shards in {uri_or_dir}"
    logger.info(
        f"[arcstore-streamer] loading {log_target} (concurrency={concurrency}, "
        f"memory_limit={memory_limit} bytes)"
    )

    # Force-set envs BEFORE the streamer import — the C++ runtime captures
    # them on first SafetensorsStreamer() instantiation.
    _force_streamer_envs(concurrency, memory_limit)

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
    concurrency: int | None = None,
    memory_limit: str | None = None,
    read_policy: str | None = None,
) -> dict[str, torch.Tensor]:
    """Load safetensors choosing the best path per source.

    * **direct ``s3://``** — every rank streams via run:ai Model Streamer.
    * **mounted ``s3://``** — rewritten to the FUSE mount path; every rank
      mmaps the on-disk shards directly. The kernel page cache dedups
      across same-node ranks, so there is no S3 read amplification, and
      we sidestep the streamer-on-FUSE deadlock (32-way range reads on a
      mountpoint-s3 path can hang indefinitely).
    * **local file/dir** — rank 0 uses the streamer (fast); other local
      ranks mmap the same on-disk shards. No extra copy.
    """
    loc = resolve(uri_or_dir)
    policy = _read_policy(
        read_policy,
        env_name="ARCSTORE_MODEL_READ_POLICY",
        default="direct_s3",
    )
    rp = loc.read_path()
    if (
        loc.is_s3
        and policy in ("auto", "mount")
        and rp is not None
        and os.path.exists(rp)
    ):
        # FUSE-mounted s3:// source. Bypass the streamer entirely on
        # every rank — the streamer's concurrent range reads deadlock
        # on mountpoint-s3 paths. mmap is safe and the kernel page
        # cache dedups across ranks.
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        files = _list_safetensors(rp)
        logger.info(
            f"[arcstore-mmap] rank{local_rank} mmap-loading {len(files)} "
            f"shard(s) from mounted path {rp} (rewritten from {uri_or_dir}); "
            f"runai-streamer skipped to avoid FUSE deadlock"
        )
        out: dict[str, torch.Tensor] = {}
        for f in files:
            out.update(_load_file_mmap(f))
        return out
    if loc.is_s3:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        logger.info(
            f"[arcstore-streamer] rank{local_rank} loading S3 safetensors "
            f"directly from {uri_or_dir}"
        )
        return load_safetensors_streamer(
            uri_or_dir,
            concurrency=concurrency,
            memory_limit=memory_limit,
        )

    # Plain local source: rank 0 uses the streamer; others mmap.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank == 0:
        return load_safetensors_streamer(
            uri_or_dir,
            concurrency=concurrency,
            memory_limit=memory_limit,
        )
    files = _list_safetensors(uri_or_dir)
    out = {}
    for f in files:
        out.update(_load_file_mmap(f))
    logger.info(
        f"[arcstore-mmap] rank{local_rank} loaded {len(out)} tensors "
        f"from {len(files)} file(s) under {uri_or_dir}"
    )
    return out


#: Conventional filename for a single-file full-weights export.
WEIGHTS_NAME = "model.safetensors"


def _unwrap_compiled(model):
    return getattr(model, "_orig_mod", model)


def load_pretrained(
    model: "torch.nn.Module",
    source: str,
    *,
    concurrency: int | None = None,
    strict: bool = True,
    memory_limit: str | None = None,
    read_policy: str | None = None,
) -> dict[str, object]:
    """Load safetensors weights (local / mount / ``s3://``) into ``model``.

    Thin convenience over :func:`load_safetensors_auto` that runs
    ``model.load_state_dict`` and returns a stats dict (loader bytes, seconds,
    GiB/s, missing/unexpected key counts). The heavy lifting — run:ai streamer
    for direct S3, mmap for FUSE-mounted paths — is handled by
    :func:`load_safetensors_auto`.
    """
    t0 = time.perf_counter()
    tensors = load_safetensors_auto(
        source,
        concurrency=concurrency,
        memory_limit=memory_limit,
        read_policy=read_policy,
    )
    n_bytes = sum(v.numel() * v.element_size() for v in tensors.values())
    incompatible = model.load_state_dict(tensors, strict=strict)
    dt = time.perf_counter() - t0
    bytes_gib = n_bytes / 1024**3
    stats = {
        "source": source,
        "n_tensors": len(tensors),
        "bytes_gib": bytes_gib,
        "seconds": dt,
        "gibps": bytes_gib / dt if dt > 0 else 0.0,
        "missing_keys": len(incompatible.missing_keys),
        "unexpected_keys": len(incompatible.unexpected_keys),
    }
    logger.info(
        "[arcstore] load_pretrained %s: %.2f GiB, %d tensors in %.2fs (%.2f GiB/s); "
        "missing=%d unexpected=%d",
        source,
        bytes_gib,
        stats["n_tensors"],
        dt,
        stats["gibps"],
        stats["missing_keys"],
        stats["unexpected_keys"],
    )
    return stats


def save_safetensors_weights(
    model: "torch.nn.Module",
    out_dir: str,
    *,
    state_dict: dict[str, "torch.Tensor"] | None = None,
    s3_prefix: str | None = None,
    workers: int | None = None,
) -> str | None:
    """Export full (unsharded) model weights to safetensors, optionally to S3.

    Without ``state_dict`` the full state dict is gathered via DCP utilities —
    a collective call, so invoke it on **all ranks**; only rank 0 writes and
    the destination path is returned there (None elsewhere). DeepSpeed ZeRO-3 /
    accelerate users should pass ``state_dict=accelerator.get_state_dict(model)``
    instead.

    Returns the S3 URI when ``s3_prefix`` is given (rank 0), else the local file
    path (rank 0), else None (non-main ranks).
    """
    from safetensors.torch import save_file

    from .._env import default_workers
    from ..uploads import upload_file
    from .runtime import is_main

    if state_dict is None:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
        )

        state_dict = get_model_state_dict(
            _unwrap_compiled(model),
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
    if not is_main():
        return None

    out_path = os.path.dirname(out_dir) if out_dir.endswith(".safetensors") else out_dir
    os.makedirs(out_path, exist_ok=True)
    local_file = out_dir if out_dir.endswith(".safetensors") else os.path.join(out_dir, WEIGHTS_NAME)
    cleaned = {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}
    save_file(cleaned, local_file, metadata={"format": "pt"})
    if s3_prefix:
        dest = (
            s3_prefix
            if s3_prefix.endswith(".safetensors")
            else s3_prefix.rstrip("/") + "/" + os.path.basename(local_file)
        )
        upload_file(local_file, dest, workers=workers if workers is not None else default_workers())
        return dest
    return local_file


__all__ = [
    "DEFAULT_CONCURRENCY",
    "DEFAULT_MEMORY_LIMIT",
    "WEIGHTS_NAME",
    "load_pretrained",
    "load_safetensors_auto",
    "load_safetensors_streamer",
    "save_safetensors_weights",
]
