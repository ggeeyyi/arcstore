"""Latest-checkpoint discovery for in-stage resume.

Generalized from CausalVideoDiffusion ``s3_io.find_latest_s3_ckpt``:

* the step-extracting regex and the required marker file are parameters;
* works on local directories, FUSE-mounted buckets, and direct ``s3://``
  prefixes alike (all through :func:`arcstore.io.list_prefix`);
* candidates are scanned highest-step first and the first one passing the
  ``required_file`` check wins, so a partial upload of the newest step
  falls back to the previous step instead of aborting resume.
"""
from __future__ import annotations

import logging
import re

from .io import list_prefix

logger = logging.getLogger(__name__)

DEFAULT_CKPT_PATTERN = r"checkpoint_model_(\d+)"


def find_latest_ckpt(
    ckpts_uri: str,
    *,
    pattern: str = DEFAULT_CKPT_PATTERN,
    required_file: str | None = "model.pt",
) -> tuple[str, int] | None:
    """Return ``(path_or_uri, step)`` of the newest complete checkpoint, or None.

    Lists ``ckpts_uri`` for child directories matching ``pattern`` (group 1
    = integer step). When ``required_file`` is set the returned path points
    at that file inside the winning directory and a directory missing it is
    skipped as a partial upload; with ``required_file=None`` the directory
    itself is returned unverified.

    For an ``s3://`` input the returned path is always an ``s3://`` URI
    (even when listing went through a mount) so downstream staging /
    download flows behave identically.
    """
    ckpt_re = re.compile(pattern.rstrip("$") + r"/?$")

    try:
        children = list_prefix(ckpts_uri)
    except RuntimeError as e:
        logger.warning(f"[arcstore] cannot list {ckpts_uri}: {e}")
        return None

    candidates: list[tuple[int, str]] = []
    for name in children:
        if not name.endswith("/"):
            continue
        m = ckpt_re.match(name)
        if m is None:
            continue
        candidates.append((int(m.group(1)), name.rstrip("/")))
    if not candidates:
        return None

    base = ckpts_uri.rstrip("/")

    for step, dirname in sorted(candidates, reverse=True):
        ckpt_dir = f"{base}/{dirname}"
        if required_file is None:
            return ckpt_dir, step
        try:
            inside = list_prefix(ckpt_dir)
        except RuntimeError:
            continue
        if required_file in inside:
            return f"{ckpt_dir}/{required_file}", step
        logger.warning(
            f"[arcstore] {ckpt_dir}/ has no {required_file}; "
            f"skipping partial upload."
        )
    return None
