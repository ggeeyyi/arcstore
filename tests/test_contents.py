"""ContentsManager: path-yielding S3 round-trip + local passthrough (fake s5cmd)."""
from __future__ import annotations

from pathlib import Path

import arcstore
from arcstore.contents import ContentsManager, local_mirror_path


def test_local_mirror_path():
    p = local_mirror_path("/cache", "s3://bkt/a/b/c.pt")
    assert p == Path("/cache/bkt/a/b/c.pt")


def test_local_passthrough_write(tmp_path):
    cm = ContentsManager(local_cache=tmp_path / "cache")
    dest = tmp_path / "out" / "f.txt"
    with cm.open(str(dest), "w") as path:
        Path(path).write_text("hello")
    assert dest.read_text() == "hello"  # local path written in place, parent created


def test_s3_write_then_read_roundtrip(fake_s5cmd, tmp_path):
    cm = ContentsManager(local_cache=tmp_path / "cache")
    uri = "s3://bkt/run/model.bin"

    with cm.open(uri, "wb") as path:
        Path(path).write_bytes(b"\x00\x01\x02weights")
    arcstore.wait_for_uploads()  # flush the background put_async

    # object landed in the fake S3 root
    assert (fake_s5cmd / "bkt" / "run" / "model.bin").read_bytes() == b"\x00\x01\x02weights"

    # a fresh manager (cold local cache) downloads on read
    cm2 = ContentsManager(local_cache=tmp_path / "cache2")
    with cm2.open(uri, "rb") as path:
        assert Path(path).read_bytes() == b"\x00\x01\x02weights"


def test_open_rejects_bad_mode(tmp_path):
    cm = ContentsManager(local_cache=tmp_path / "cache")
    import pytest

    with pytest.raises(ValueError):
        with cm.open("s3://bkt/x", "x"):
            pass
