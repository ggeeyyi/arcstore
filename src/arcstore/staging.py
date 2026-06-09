"""Local NVMe staging cache for slow-filesystem / S3-backed artifacts.

Generalized from CausalVideoDiffusion ``src/utils/ckpt_cache.py``:

* :func:`stage_to_local` — mirror a checkpoint-style file (plus an optional
  curated ``siblings`` set from the same directory) into a local cache and
  return the staged path. The sibling list used to be hardcoded
  (``model_ema.pt`` / ``action_module.pt``); it is now a parameter so any
  codebase can declare its own checkpoint layout.
* :func:`ensure_local_file` — small-file localization (jsonl manifests,
  configs): size-checked cache hit, atomic tmp+rename download.

Concurrency: per-checkpoint ``fcntl.flock`` + a ``.stage_done`` sentinel.
Failure handling: every external dependency falls through to returning the
*original* path — a flaky cache must never block a run.

Env knobs (read at call time, overridable per-call):

* ``ARCSTORE_CACHE_DIR``         — cache root (default ``/tmp/arcstore-cache``)
* ``ARCSTORE_CACHE_ENABLE``      — ``0``/``false`` bypasses staging
* ``ARCSTORE_CACHE_BUDGET_GIB``  — LRU eviction budget (default 200)
* ``ARCSTORE_STAGE_PREFIXES``    — optional comma-separated whitelist of
  path prefixes eligible for staging (unset = stage everything outside the
  cache dir itself)
"""
from __future__ import annotations

import hashlib
import logging
import os
import posixpath
import time
from typing import Optional, Sequence

from . import s3cli
from ._env import env_bool, env_float, env_str
from .location import is_s3, resolve

DEFAULT_CACHE_DIR = "/tmp/arcstore-cache"

_module_logger = logging.getLogger(__name__)


def _get_logger(logger):
    return logger if logger is not None else _module_logger


def _cache_root(override: str | None = None) -> str:
    return override or env_str("ARCSTORE_CACHE_DIR", DEFAULT_CACHE_DIR)


def _budget_bytes(override_gib: float | None = None) -> int:
    gib = (
        override_gib
        if override_gib is not None
        else env_float("ARCSTORE_CACHE_BUDGET_GIB", 200.0)
    )
    return int(gib * (1024**3))


def _stage_prefixes() -> tuple[str, ...]:
    raw = os.environ.get("ARCSTORE_STAGE_PREFIXES", "")
    return tuple(p for p in raw.split(",") if p)


def _should_stage(path: str, cache_root: str) -> bool:
    if not env_bool("ARCSTORE_CACHE_ENABLE", True):
        return False
    prefixes = _stage_prefixes()
    if prefixes:
        return path.startswith(prefixes)
    # Default: stage everything that isn't already in the cache root.
    try:
        path_abs = os.path.realpath(path)
        cache_abs = os.path.realpath(cache_root)
    except OSError:
        return True
    return not path_abs.startswith(cache_abs.rstrip("/") + "/")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def stage_to_local(
    path: str,
    *,
    siblings: Sequence[str] = (),
    cache_root: str | None = None,
    budget_gib: float | None = None,
    copy_whole_dir: bool = True,
    logger=None,
) -> str:
    """Mirror a slow-FS / S3 artifact into a local cache; return the new path.

    ``siblings`` lists extra basenames from the same directory to stage
    best-effort next to the primary file (e.g. ``("model_ema.pt",)`` so a
    later ``os.path.dirname()`` lookup finds them locally). For an S3 source
    only the primary + siblings are downloaded — never the whole prefix, so
    a mispointed path can't mirror a 10k-shard data dir onto local disk.
    For a local (slow-mount) source the containing directory's flat files
    are copied when ``copy_whole_dir`` is true.

    On any error returns the original ``path`` so the caller's load still
    works, just slowly.
    """
    log = _get_logger(logger)
    root = _cache_root(cache_root)

    if is_s3(path):
        return _stage_s3_dir(
            path,
            siblings=siblings,
            cache_root=root,
            budget_gib=budget_gib,
            logger=log,
        )

    if not _should_stage(path, root):
        return path
    return _stage_local_dir(
        path,
        cache_root=root,
        budget_gib=budget_gib,
        copy_whole_dir=copy_whole_dir,
        logger=log,
    )


def ensure_local_file(
    path: str,
    *,
    cache_dir: str | None = None,
    label: str = "file",
    logger=None,
) -> str:
    """Resolve a small file (manifest, config) to a local readable path.

    ``s3://`` sources are downloaded into ``cache_dir`` (default
    ``<ARCSTORE_CACHE_DIR>/files``) with a byte-size cache-hit check and an
    atomic tmp+rename so concurrent workers never observe a half-written
    file. Mounted buckets short-circuit to the mount path (no copy). Local
    paths pass through unchanged.
    """
    log = _get_logger(logger)
    loc = resolve(path)
    if not loc.is_s3:
        return path
    rp = loc.read_path()
    if rp is not None and os.path.isfile(rp):
        return rp

    cache_dir = cache_dir or os.path.join(_cache_root(), "files")
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, os.path.basename(loc.raw))

    # Size-match cache hit: S3 objects are atomic uploads, size never
    # changes mid-write, and mtime isn't cheaply available.
    remote_size = None
    try:
        remote_size = s3cli.head_object(loc.s3_uri())
    except Exception:  # noqa: BLE001
        pass
    if (
        remote_size is not None
        and os.path.isfile(local_path)
        and os.path.getsize(local_path) == remote_size
    ):
        log.info(f"[arcstore] {label}: cache hit {path} -> {local_path}")
        return local_path

    tmp_path = local_path + f".tmp.{os.getpid()}"
    t0 = time.perf_counter()
    try:
        s3cli.download_file(loc.s3_uri(), tmp_path, label=label)
        os.replace(tmp_path, local_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    dt = time.perf_counter() - t0
    sz = os.path.getsize(local_path)
    log.info(
        f"[arcstore] {label}: downloaded {path} -> {local_path} "
        f"({sz / (1 << 20):.1f} MiB in {dt:.2f}s)"
    )
    return local_path


# ---------------------------------------------------------------------------
# S3 branch
# ---------------------------------------------------------------------------
def _stage_s3_dir(
    s3_path: str,
    *,
    siblings: Sequence[str],
    cache_root: str,
    budget_gib: float | None,
    logger,
) -> str:
    s3_parent = posixpath.dirname(s3_path.rstrip("/"))
    fname = posixpath.basename(s3_path.rstrip("/"))
    if not fname:
        raise ValueError(f"[arcstore] s3 path has no basename: {s3_path!r}")

    # SHA1 of the s3 parent prefix gives a stable, collision-free local
    # subdir name; the prefix's last segment is appended for readability.
    dir_key = hashlib.sha1(s3_parent.encode("utf-8")).hexdigest()[:16]
    last_segment = posixpath.basename(s3_parent.rstrip("/")) or "ckpt"
    cache_subdir = os.path.join(cache_root, f"{dir_key}__{last_segment}")
    sentinel = os.path.join(cache_subdir, ".stage_done")
    staged_path = os.path.join(cache_subdir, fname)

    try:
        os.makedirs(cache_subdir, exist_ok=True)
    except OSError as e:
        logger.warning(
            f"[arcstore] cannot create {cache_subdir} ({e}); "
            f"returning original s3 path."
        )
        return s3_path

    if os.path.isfile(sentinel) and os.path.isfile(staged_path):
        try:
            os.utime(sentinel, None)  # bump LRU mtime
        except OSError:
            pass
        logger.info(f"[arcstore] stage hit (s3): {staged_path}")
        return staged_path

    targets: list[tuple[str, bool]] = [(fname, True)]  # (basename, required)
    for sib in siblings:
        if sib != fname:
            targets.append((sib, False))

    def _download_one(basename: str, required: bool) -> bool:
        local = os.path.join(cache_subdir, basename)
        if os.path.isfile(local):
            return True
        s3_uri = f"{s3_parent.rstrip('/')}/{basename}"
        try:
            s3cli.download_file(s3_uri, local, label="arcstore-stage")
            return True
        except FileNotFoundError:
            if required:
                logger.warning(f"[arcstore] required object missing: {s3_uri}")
            return False
        except Exception as e:  # noqa: BLE001
            level = logger.warning if required else logger.info
            level(f"[arcstore] fetch {s3_uri} failed ({e}); "
                  + ("" if required else "continuing."))
            return False

    logger.info(
        f"[arcstore] s3-staging {fname} (+{len(targets) - 1} optional sibling(s)) "
        f"from {s3_parent}/ -> {cache_subdir}/"
    )
    t0 = time.perf_counter()
    if not _download_one(fname, required=True):
        # Last resort: a mounted bucket still serves reads without any CLI.
        rp = resolve(s3_path).read_path()
        if rp is not None and os.path.isfile(rp):
            logger.info(f"[arcstore] falling back to mounted read path {rp}")
            return rp
        logger.warning("[arcstore] primary fetch failed; returning original s3 path.")
        return s3_path
    for basename, required in targets[1:]:
        _download_one(basename, required=required)

    with open(sentinel, "w") as f:
        f.write(f"staged {s3_parent} -> {cache_subdir}\n")

    dt = time.perf_counter() - t0
    try:
        sz = os.path.getsize(staged_path) / (1024**3)
        rate = f", {sz * 1024 / dt:.0f} MiB/s" if dt > 0 else ""
        logger.info(f"[arcstore] s3-staged {sz:.1f} GiB in {dt:.1f}s{rate} -> {staged_path}")
    except OSError:
        logger.info(f"[arcstore] s3-staged in {dt:.1f}s -> {staged_path}")

    _evict_lru(
        cache_root=cache_root,
        budget_bytes=_budget_bytes(budget_gib),
        keep_subdir=cache_subdir,
        logger=logger,
    )
    return staged_path


# ---------------------------------------------------------------------------
# Local (slow mount) branch
# ---------------------------------------------------------------------------
def _stage_local_dir(
    path: str,
    *,
    cache_root: str,
    budget_gib: float | None,
    copy_whole_dir: bool,
    logger,
) -> str:
    import fcntl
    import shutil

    src_dir = os.path.dirname(path)
    fname = os.path.basename(path)
    # Hash the source *directory* so all artifacts of one ckpt step share a
    # single staged dir / sentinel / lock.
    dir_key = hashlib.sha1(src_dir.encode("utf-8")).hexdigest()[:16]
    cache_subdir = os.path.join(cache_root, f"{dir_key}__{os.path.basename(src_dir)}")
    sentinel = os.path.join(cache_subdir, ".stage_done")
    lock_path = os.path.join(cache_root, f".{dir_key}.lock")
    staged_path = os.path.join(cache_subdir, fname)

    try:
        os.makedirs(cache_root, exist_ok=True)
    except OSError as e:
        logger.warning(
            f"[arcstore] cannot create cache dir {cache_root} ({e}); "
            f"falling back to source path."
        )
        return path

    if os.path.isfile(sentinel) and os.path.isfile(staged_path):
        try:
            os.utime(sentinel, None)
        except OSError:
            pass
        logger.info(f"[arcstore] stage hit: {staged_path}")
        return staged_path

    logger.info(
        f"[arcstore] staging {src_dir} -> {cache_subdir} "
        f"(once per ckpt; reused on subsequent runs)"
    )
    t0 = time.perf_counter()
    os.makedirs(cache_subdir, exist_ok=True)
    with open(lock_path, "w") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            # Re-check after lock acquisition: a sibling rank may have
            # finished while we were waiting.
            if os.path.isfile(sentinel) and os.path.isfile(staged_path):
                logger.info(
                    f"[arcstore] another process finished staging while we "
                    f"waited; using {staged_path}"
                )
            else:
                try:
                    if copy_whole_dir:
                        # Flat files only — checkpoint step dirs are flat by
                        # convention; never recurse into subdirectories.
                        for entry in os.scandir(src_dir):
                            if not entry.is_file(follow_symlinks=False):
                                continue
                            shutil.copy2(
                                entry.path, os.path.join(cache_subdir, entry.name)
                            )
                    else:
                        shutil.copy2(path, staged_path)
                except OSError as e:
                    logger.warning(
                        f"[arcstore] copy failed ({e}); falling back to {path}"
                    )
                    return path
                if not os.path.isfile(staged_path):
                    logger.warning(
                        f"[arcstore] expected staged file {staged_path} missing "
                        f"after copy; falling back to {path}"
                    )
                    return path
                with open(sentinel, "w") as f:
                    f.write(f"staged {src_dir} -> {cache_subdir}\n")
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

    dt = time.perf_counter() - t0
    try:
        sz = os.path.getsize(staged_path) / (1024**3)
        rate = f", {sz * 1024 / dt:.0f} MiB/s" if dt > 0 else ""
        logger.info(f"[arcstore] staged {sz:.1f} GiB in {dt:.1f}s{rate} -> {staged_path}")
    except OSError:
        logger.info(f"[arcstore] staged in {dt:.1f}s -> {staged_path}")

    _evict_lru(
        cache_root=cache_root,
        budget_bytes=_budget_bytes(budget_gib),
        keep_subdir=cache_subdir,
        logger=logger,
    )
    return staged_path


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------
def _evict_lru(
    *,
    cache_root: str,
    budget_bytes: int,
    keep_subdir: Optional[str],
    logger,
) -> None:
    """LRU-evict staged dirs (by ``.stage_done`` mtime) until under budget."""
    try:
        entries: list[tuple[float, str, int]] = []  # (mtime, dir, bytes)
        with os.scandir(cache_root) as it:
            for de in it:
                if not de.is_dir(follow_symlinks=False):
                    continue
                sub = de.path
                sentinel = os.path.join(sub, ".stage_done")
                if not os.path.isfile(sentinel):
                    continue  # in-progress or unrelated; skip
                try:
                    mt = os.path.getmtime(sentinel)
                except OSError:
                    continue
                total = 0
                for root, _, files in os.walk(sub):
                    for fn in files:
                        try:
                            total += os.path.getsize(os.path.join(root, fn))
                        except OSError:
                            pass
                entries.append((mt, sub, total))
    except FileNotFoundError:
        return

    total_bytes = sum(e[2] for e in entries)
    if total_bytes <= budget_bytes:
        return

    import shutil

    entries.sort(key=lambda e: e[0])  # oldest first
    over = total_bytes - budget_bytes
    logger.info(
        f"[arcstore] LRU eviction: {total_bytes / 1024**3:.1f} GiB > "
        f"{budget_bytes / 1024**3:.1f} GiB budget, freeing {over / 1024**3:.1f} GiB"
    )
    for _mt, sub, sz in entries:
        if total_bytes <= budget_bytes:
            break
        if keep_subdir is not None and os.path.realpath(sub) == os.path.realpath(
            keep_subdir
        ):
            continue
        try:
            shutil.rmtree(sub)
            total_bytes -= sz
            logger.info(f"[arcstore]   evicted {sub} ({sz / 1024**3:.1f} GiB)")
        except OSError as e:
            logger.warning(f"[arcstore]   failed to evict {sub}: {e}")
