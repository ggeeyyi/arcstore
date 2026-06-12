import pytest

pytest.importorskip("torch")

import arcstore  # noqa: E402
from arcstore.torch import expand_urls, shard_urls, tar_url  # noqa: E402


def test_tar_url_s3_pipes_s5cmd():
    assert tar_url("s3://bkt/shards", "clip-000.tar") == (
        "pipe:s5cmd cat s3://bkt/shards/clip-000.tar"
    )


def test_tar_url_local_joins(tmp_path):
    assert tar_url(str(tmp_path), "a.tar") == str(tmp_path / "a.tar")


def test_tar_url_mounted_defaults_to_direct_s3(monkeypatch, tmp_path):
    mnt = tmp_path / "mnt"
    (mnt / "shards").mkdir(parents=True)
    monkeypatch.setenv("ARCSTORE_S3_MOUNTS", f"bkt={mnt}")
    arcstore.refresh_mounts()
    assert tar_url("s3://bkt/shards", "a.tar") == (
        "pipe:s5cmd cat s3://bkt/shards/a.tar"
    )


def test_tar_url_mounted_explicit_mount_policy(monkeypatch, tmp_path):
    mnt = tmp_path / "mnt"
    (mnt / "shards").mkdir(parents=True)
    monkeypatch.setenv("ARCSTORE_S3_MOUNTS", f"bkt={mnt}")
    arcstore.refresh_mounts()
    assert tar_url("s3://bkt/shards", "a.tar", read_policy="mount") == str(
        mnt / "shards" / "a.tar"
    )


def test_shard_urls_suffix():
    urls = shard_urls("s3://bkt/sh", ["000", "001.tar"], name_prefix="clip-")
    assert urls == [
        "pipe:s5cmd cat s3://bkt/sh/clip-000.tar",
        "pipe:s5cmd cat s3://bkt/sh/001.tar",
    ]


def test_expand_urls_glob_direct_s3(fake_s5cmd):
    shard_dir = fake_s5cmd / "bkt" / "shards"
    shard_dir.mkdir(parents=True)
    (shard_dir / "latent_shard-001.tar").write_bytes(b"1")
    (shard_dir / "latent_shard-000.tar").write_bytes(b"0")
    (shard_dir / "notes.txt").write_text("x")

    assert expand_urls("s3://bkt/shards/latent_shard-*.tar") == [
        "pipe:s5cmd cat s3://bkt/shards/latent_shard-000.tar",
        "pipe:s5cmd cat s3://bkt/shards/latent_shard-001.tar",
    ]


def test_expand_urls_brace_range():
    assert expand_urls("s3://bkt/shards/shard-{000..002}.tar") == [
        "pipe:s5cmd cat s3://bkt/shards/shard-000.tar",
        "pipe:s5cmd cat s3://bkt/shards/shard-001.tar",
        "pipe:s5cmd cat s3://bkt/shards/shard-002.tar",
    ]
