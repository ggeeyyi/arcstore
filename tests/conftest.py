"""Shared fixtures: a fake-s5cmd PATH shim backed by a local dir, env reset.

The fake ``s5cmd`` maps ``s3://<bucket>/<key>`` to
``$FAKE_S3_ROOT/<bucket>/<key>`` and implements the ``cp`` / ``ls`` / ``cat``
subset arcstore drives, including the ``DIR`` listing rows and the
``no object found`` stderr marker. This exercises the s5cmd-first code
paths that moto cannot reach.
"""
from __future__ import annotations

import os
import stat
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import arcstore  # noqa: E402
import arcstore.location  # noqa: E402

_FAKE_S5CMD = r"""#!/usr/bin/env bash
set -u
ROOT="${FAKE_S3_ROOT:?FAKE_S3_ROOT not set}"

args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --numworkers) shift 2 ;;
    *) args+=("$1"); shift ;;
  esac
done
cmd="${args[0]:-}"

to_local() { local p="${1#s3://}"; printf '%s/%s' "$ROOT" "$p"; }

case "$cmd" in
  ls)
    target="${args[1]}"
    local_path=$(to_local "$target")
    if [[ "$target" == */ ]]; then
      dir="${local_path%/}"
      if [[ ! -d "$dir" ]]; then
        echo "ERROR \"ls ${target}\": no object found" >&2; exit 1
      fi
      shopt -s nullglob
      found=0
      for e in "$dir"/*; do
        found=1
        name=$(basename "$e")
        if [[ -d "$e" ]]; then
          printf '%34s  %s/\n' DIR "$name"
        else
          sz=$(stat -c%s "$e")
          printf '2026/01/01 00:00:00 %12d %s\n' "$sz" "$name"
        fi
      done
      if [[ $found -eq 0 ]]; then
        echo "ERROR \"ls ${target}\": no object found" >&2; exit 1
      fi
    else
      if [[ -f "$local_path" ]]; then
        sz=$(stat -c%s "$local_path")
        printf '2026/01/01 00:00:00 %12d %s\n' "$sz" "$(basename "$local_path")"
      else
        echo "ERROR \"ls ${target}\": no object found" >&2; exit 1
      fi
    fi
    ;;
  cp)
    src="${args[1]}"; dst="${args[2]}"
    if [[ "$src" == s3://* ]]; then
      lsrc=$(to_local "$src")
      if [[ "$src" == *'/*' ]]; then
        srcdir="${lsrc%/\*}"
        if [[ ! -d "$srcdir" ]]; then echo "no object found" >&2; exit 1; fi
        mkdir -p "${dst%/}"
        cp -r "$srcdir"/. "${dst%/}/"
      else
        if [[ ! -f "$lsrc" ]]; then echo "ERROR: NoSuchKey: $src" >&2; exit 1; fi
        if [[ "$dst" == */ || -d "$dst" ]]; then
          mkdir -p "${dst%/}"
          cp "$lsrc" "${dst%/}/$(basename "$lsrc")"
        else
          mkdir -p "$(dirname "$dst")"
          cp "$lsrc" "$dst"
        fi
      fi
    else
      ldst=$(to_local "$dst")
      if [[ "$src" == */ ]]; then
        if [[ ! -d "${src%/}" ]]; then echo "ERROR: no such directory: $src" >&2; exit 1; fi
        mkdir -p "${ldst%/}"
        cp -r "${src%/}"/. "${ldst%/}/"
      else
        if [[ ! -f "$src" ]]; then echo "ERROR: no such file: $src" >&2; exit 1; fi
        if [[ "$dst" == */ ]]; then
          mkdir -p "${ldst%/}"
          cp "$src" "${ldst%/}/$(basename "$src")"
        else
          mkdir -p "$(dirname "$ldst")"
          cp "$src" "$ldst"
        fi
      fi
    fi
    ;;
  cat)
    local_path=$(to_local "${args[1]}")
    cat "$local_path" || exit 1
    ;;
  *)
    echo "fake s5cmd: unsupported command: $cmd" >&2; exit 1
    ;;
esac
exit 0
"""


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    """Isolate every test from real arcstore env + reset the mount cache."""
    for var in list(os.environ):
        if var.startswith("ARCSTORE_"):
            monkeypatch.delenv(var, raising=False)
    # Keep caches inside tmp so tests never touch /tmp/arcstore-cache.
    monkeypatch.setenv("ARCSTORE_CACHE_DIR", str(tmp_path / "arcstore-cache"))
    # Never let a stray boto3 call hang on IMDS or hit real AWS.
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    arcstore.refresh_mounts()
    yield
    arcstore.refresh_mounts()


@pytest.fixture
def fake_s3_root(tmp_path):
    root = tmp_path / "fake-s3"
    root.mkdir()
    return root


@pytest.fixture
def fake_s5cmd(monkeypatch, tmp_path, fake_s3_root):
    """Put a fake ``s5cmd`` on PATH mapping s3:// to ``fake_s3_root``."""
    bindir = tmp_path / "fakebin"
    bindir.mkdir()
    exe = bindir / "s5cmd"
    exe.write_text(_FAKE_S5CMD)
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_S3_ROOT", str(fake_s3_root))
    return fake_s3_root


@pytest.fixture
def no_cli(monkeypatch):
    """Hide s5cmd / aws from ``shutil.which`` so boto3 fallbacks are exercised."""
    import shutil

    real_which = shutil.which

    def fake_which(name, *args, **kwargs):
        if name in ("s5cmd", "aws"):
            return None
        return real_which(name, *args, **kwargs)

    monkeypatch.setattr("shutil.which", fake_which)
