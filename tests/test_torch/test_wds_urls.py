import pytest

pytest.importorskip("torch")

import arcstore  # noqa: E402
from arcstore.torch import shard_urls, tar_url  # noqa: E402


def test_tar_url_s3_pipes_s5cmd():
    assert tar_url("s3://bkt/shards", "clip-000.tar") == (
        "pipe:s5cmd cat s3://bkt/shards/clip-000.tar"
    )


def test_tar_url_local_joins(tmp_path):
    assert tar_url(str(tmp_path), "a.tar") == str(tmp_path / "a.tar")


def test_tar_url_mounted(monkeypatch, tmp_path):
    mnt = tmp_path / "mnt"
    (mnt / "shards").mkdir(parents=True)
    monkeypatch.setenv("ARCSTORE_S3_MOUNTS", f"bkt={mnt}")
    arcstore.refresh_mounts()
    assert tar_url("s3://bkt/shards", "a.tar") == str(mnt / "shards" / "a.tar")


def test_shard_urls_suffix():
    urls = shard_urls("s3://bkt/sh", ["000", "001.tar"], name_prefix="clip-")
    assert urls == [
        "pipe:s5cmd cat s3://bkt/sh/clip-000.tar",
        "pipe:s5cmd cat s3://bkt/sh/001.tar",
    ]
