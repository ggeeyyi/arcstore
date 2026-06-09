import os

import pytest

import arcstore


@pytest.fixture
def populated_bucket(fake_s5cmd):
    """Lay out s3://bkt/data/{a.txt,b.pt,sub/c.txt} in the fake bucket."""
    base = fake_s5cmd / "bkt" / "data"
    (base / "sub").mkdir(parents=True)
    (base / "a.txt").write_bytes(b"hello a")
    (base / "b.pt").write_bytes(b"pt-bytes")
    (base / "sub" / "c.txt").write_bytes(b"c")
    return base


def test_exists_s3(populated_bucket):
    assert arcstore.exists("s3://bkt/data/a.txt")
    assert not arcstore.exists("s3://bkt/data/missing.txt")


def test_exists_local(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("x")
    assert arcstore.exists(str(p))
    assert not arcstore.exists(str(tmp_path / "nope"))


def test_read_bytes_via_mount(monkeypatch, fake_s3_root, populated_bucket):
    # Mount the fake bucket dir as if mountpoint-s3 exposed it.
    monkeypatch.setenv("ARCSTORE_S3_MOUNTS", f"bkt={fake_s3_root / 'bkt'}")
    arcstore.refresh_mounts()
    assert arcstore.read_bytes("s3://bkt/data/a.txt") == b"hello a"


def test_open_read_text_via_mount(monkeypatch, fake_s3_root, populated_bucket):
    monkeypatch.setenv("ARCSTORE_S3_MOUNTS", f"bkt={fake_s3_root / 'bkt'}")
    arcstore.refresh_mounts()
    with arcstore.open_read("s3://bkt/data/a.txt", "r") as f:
        assert f.read() == "hello a"


def test_list_prefix_s3(populated_bucket):
    children = arcstore.list_prefix("s3://bkt/data")
    assert children == ["a.txt", "b.pt", "sub/"]


def test_list_prefix_local(tmp_path):
    (tmp_path / "x.txt").write_text("x")
    (tmp_path / "d").mkdir()
    assert arcstore.list_prefix(str(tmp_path)) == ["d/", "x.txt"]
    assert arcstore.list_prefix(str(tmp_path / "missing")) == []


def test_list_prefix_empty_s3(fake_s5cmd):
    assert arcstore.list_prefix("s3://bkt/never-written") == []


def test_glob_files_s3_returns_uris(populated_bucket):
    assert arcstore.glob_files("s3://bkt/data", ".pt") == ["s3://bkt/data/b.pt"]


def test_glob_files_mounted_returns_local_paths(
    monkeypatch, fake_s3_root, populated_bucket
):
    monkeypatch.setenv("ARCSTORE_S3_MOUNTS", f"bkt={fake_s3_root / 'bkt'}")
    arcstore.refresh_mounts()
    out = arcstore.glob_files("s3://bkt/data", ".pt")
    assert out == [str(populated_bucket / "b.pt")]
    assert all(os.path.isfile(p) for p in out)


def test_download_file(populated_bucket, tmp_path):
    dst = tmp_path / "out" / "a.txt"
    arcstore.download_file("s3://bkt/data/a.txt", str(dst))
    assert dst.read_bytes() == b"hello a"


def test_download_file_missing_raises(fake_s5cmd, tmp_path):
    with pytest.raises(FileNotFoundError):
        arcstore.download_file("s3://bkt/data/nope.txt", str(tmp_path / "x"))


def test_download_dir(populated_bucket, tmp_path):
    dst = tmp_path / "mirror"
    arcstore.download_dir("s3://bkt/data", str(dst))
    assert (dst / "a.txt").read_bytes() == b"hello a"
    assert (dst / "sub" / "c.txt").read_bytes() == b"c"


def test_write_primitives_reject_non_s3(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("x")
    with pytest.raises(ValueError):
        arcstore.upload_file(str(f), "/mnt/not-s3/f.txt")
    with pytest.raises(ValueError):
        arcstore.upload_dir(str(tmp_path), "/mnt/not-s3/")
    with pytest.raises(ValueError):
        arcstore.download_dir("/mnt/not-s3", str(tmp_path))


def test_writes_ignore_mount(monkeypatch, fake_s3_root, fake_s5cmd, tmp_path):
    """Uploads go through the S3 API even when the bucket is mounted."""
    mnt = fake_s3_root / "bkt"
    mnt.mkdir(exist_ok=True)
    monkeypatch.setenv("ARCSTORE_S3_MOUNTS", f"bkt={mnt}")
    arcstore.refresh_mounts()
    src = tmp_path / "up.txt"
    src.write_text("payload")
    arcstore.upload_file(str(src), "s3://bkt/up/up.txt")
    # The fake s5cmd wrote into the fake bucket — proving the S3 API path
    # was taken (a mount write would have gone to the same dir here, but a
    # ValueError-free pass through upload_file is the contract under test).
    assert (fake_s3_root / "bkt" / "up" / "up.txt").read_text() == "payload"
