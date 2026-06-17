import pytest

torch = pytest.importorskip("torch")

from arcstore.torch import ScatterPtDataset  # noqa: E402


@pytest.fixture
def pt_dir(tmp_path):
    for i in range(8):
        torch.save({"x": torch.full((2,), float(i)), "name": f"s{i}"}, tmp_path / f"{i:03d}.pt")
    return tmp_path


def test_default_transform(pt_dir):
    ds = ScatterPtDataset(str(pt_dir), shuffle_buffer=1)
    samples = list(ds)
    assert len(samples) == 8
    assert {s["name"] for s in samples} == {f"s{i}" for i in range(8)}


def test_custom_transform(pt_dir):
    ds = ScatterPtDataset(
        str(pt_dir), transform=lambda raw: len(raw), shuffle_buffer=1
    )
    out = list(ds)
    assert len(out) == 8
    assert all(isinstance(n, int) and n > 0 for n in out)


def test_rank_sharding(pt_dir, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    n0 = len(list(ScatterPtDataset(str(pt_dir), shuffle_buffer=1)))
    monkeypatch.setenv("RANK", "1")
    n1 = len(list(ScatterPtDataset(str(pt_dir), shuffle_buffer=1)))
    assert n0 + n1 == 8
    assert n0 == n1 == 4


def test_mount_used_by_default(pt_dir, monkeypatch):
    """A usable mount is read by default (ARCSTORE_DATA_READ_POLICY=auto)."""
    monkeypatch.setenv("ARCSTORE_S3_MOUNTS", f"bkt={pt_dir}")
    import arcstore

    arcstore.refresh_mounts()
    ds = ScatterPtDataset("s3://bkt", shuffle_buffer=1)
    assert ds._local_dir == str(pt_dir)
    assert len(list(ds)) == 8
    # Explicit direct_s3 opts out of the mount.
    ds_direct = ScatterPtDataset("s3://bkt", read_policy="direct_s3", shuffle_buffer=1)
    assert ds_direct._local_dir is None


def test_use_mount_false_forces_s3(pt_dir, monkeypatch):
    monkeypatch.setenv("ARCSTORE_S3_MOUNTS", f"bkt={pt_dir}")
    import arcstore

    arcstore.refresh_mounts()
    ds = ScatterPtDataset("s3://bkt", use_mount=False, shuffle_buffer=1)
    assert ds._local_dir is None  # would go through s3torchconnector


def test_len_requires_length(pt_dir):
    ds = ScatterPtDataset(str(pt_dir))
    with pytest.raises(TypeError):
        len(ds)
    assert len(ScatterPtDataset(str(pt_dir), length=8)) == 8
