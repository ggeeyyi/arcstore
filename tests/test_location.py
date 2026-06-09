import arcstore
from arcstore.location import mount_root_for


def test_is_s3():
    assert arcstore.is_s3("s3://bkt/key")
    assert not arcstore.is_s3("/local/path")
    assert not arcstore.is_s3(None)
    assert not arcstore.is_s3(123)


def test_split_s3():
    assert arcstore.split_s3("s3://bkt/a/b.pt") == ("bkt", "a/b.pt")
    assert arcstore.split_s3("s3://bkt") == ("bkt", "")


def test_resolve_local():
    loc = arcstore.resolve("/data/foo.pt")
    assert loc.scheme == "file"
    assert not loc.is_s3
    assert loc.read_path() == "/data/foo.pt"
    assert loc.readable() == "/data/foo.pt"


def test_resolve_s3_no_mount():
    loc = arcstore.resolve("s3://bkt/a/b")
    assert loc.is_s3
    assert loc.bucket == "bkt"
    assert loc.key == "a/b"
    assert loc.read_path() is None
    assert loc.readable() == "s3://bkt/a/b"
    assert loc.s3_uri() == "s3://bkt/a/b"


def test_mount_translation(monkeypatch, tmp_path):
    mnt = tmp_path / "mnt"
    mnt.mkdir()
    monkeypatch.setenv("ARCSTORE_S3_MOUNTS", f"bkt={mnt}")
    arcstore.refresh_mounts()

    loc = arcstore.resolve("s3://bkt/a/b.pt")
    assert loc.mount_root == str(mnt)
    assert loc.read_path() == f"{mnt}/a/b.pt"
    assert loc.readable() == f"{mnt}/a/b.pt"
    # The s3 identity is preserved for writes / re-uploads.
    assert loc.s3_uri() == "s3://bkt/a/b.pt"

    # Unlisted bucket is untouched.
    other = arcstore.resolve("s3://other/a")
    assert other.read_path() is None


def test_mount_missing_dir_falls_back(monkeypatch, tmp_path):
    monkeypatch.setenv("ARCSTORE_S3_MOUNTS", f"bkt={tmp_path}/definitely-missing")
    arcstore.refresh_mounts()
    assert arcstore.resolve("s3://bkt/a").read_path() is None


def test_use_mounts_kill_switch(monkeypatch, tmp_path):
    mnt = tmp_path / "mnt"
    mnt.mkdir()
    monkeypatch.setenv("ARCSTORE_S3_MOUNTS", f"bkt={mnt}")
    monkeypatch.setenv("ARCSTORE_USE_MOUNTS", "0")
    arcstore.refresh_mounts()
    assert arcstore.resolve("s3://bkt/a").read_path() is None


def test_malformed_mount_entries_skipped(monkeypatch, tmp_path):
    mnt = tmp_path / "mnt"
    mnt.mkdir()
    monkeypatch.setenv("ARCSTORE_S3_MOUNTS", f"nonsense, bkt={mnt}, =, x=")
    arcstore.refresh_mounts()
    assert mount_root_for("bkt") == str(mnt)
    assert mount_root_for("nonsense") is None


def test_refresh_mounts_rereads_env(monkeypatch, tmp_path):
    mnt = tmp_path / "mnt"
    mnt.mkdir()
    arcstore.refresh_mounts()
    assert arcstore.resolve("s3://bkt/a").read_path() is None
    monkeypatch.setenv("ARCSTORE_S3_MOUNTS", f"bkt={mnt}")
    # Cached until refreshed.
    assert arcstore.resolve("s3://bkt/a").read_path() is None
    arcstore.refresh_mounts()
    assert arcstore.resolve("s3://bkt/a").read_path() == f"{mnt}/a"
