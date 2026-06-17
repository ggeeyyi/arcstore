import arcstore
from arcstore import resolve_dataset_access


def test_local_path():
    acc = resolve_dataset_access("/data/latents")
    assert acc.mode == "local"
    assert acc.local_dir == "/data/latents"
    assert acc.s3_uri is None
    assert acc.is_local_read


def test_direct_s3_is_default():
    acc = resolve_dataset_access("s3://bkt/latents")
    assert acc.mode == "direct_s3"
    assert acc.local_dir is None
    assert acc.s3_uri == "s3://bkt/latents"
    assert not acc.is_local_read


def test_mount_used_by_default(monkeypatch, tmp_path):
    mnt = tmp_path / "mnt"
    (mnt / "latents").mkdir(parents=True)
    monkeypatch.setenv("ARCSTORE_S3_MOUNTS", f"bkt={mnt}")
    arcstore.refresh_mounts()

    # Default (auto) reads a usable mount as a local path.
    acc = resolve_dataset_access("s3://bkt/latents")
    assert acc.mode == "mount"
    assert acc.local_dir == str(mnt / "latents")
    assert acc.s3_uri == "s3://bkt/latents"
    assert resolve_dataset_access("s3://bkt/latents", read_policy="mount").mode == "mount"

    # Explicit direct_s3 opts out of the mount.
    assert resolve_dataset_access(
        "s3://bkt/latents", read_policy="direct_s3"
    ).mode == "direct_s3"


def test_mount_missing_dir_falls_back_to_direct_s3(monkeypatch, tmp_path):
    mnt = tmp_path / "mnt"
    mnt.mkdir()  # mount root exists but the key dir does not
    monkeypatch.setenv("ARCSTORE_S3_MOUNTS", f"bkt={mnt}")
    arcstore.refresh_mounts()
    acc = resolve_dataset_access("s3://bkt/missing", read_policy="mount")
    assert acc.mode == "direct_s3"


def test_env_policy_respected(monkeypatch, tmp_path):
    mnt = tmp_path / "mnt"
    (mnt / "latents").mkdir(parents=True)
    monkeypatch.setenv("ARCSTORE_S3_MOUNTS", f"bkt={mnt}")
    monkeypatch.setenv("ARCSTORE_DATA_READ_POLICY", "mount")
    arcstore.refresh_mounts()
    assert resolve_dataset_access("s3://bkt/latents").mode == "mount"
