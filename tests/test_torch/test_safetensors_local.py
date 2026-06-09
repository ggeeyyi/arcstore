import pytest

torch = pytest.importorskip("torch")
st = pytest.importorskip("safetensors.torch")

from arcstore.torch import load_safetensors_auto, load_safetensors_streamer  # noqa: E402


@pytest.fixture
def st_dir(tmp_path):
    st.save_file({"a": torch.ones(4), "b": torch.zeros(2)}, str(tmp_path / "m1.safetensors"))
    st.save_file({"c": torch.full((3,), 2.0)}, str(tmp_path / "m2.safetensors"))
    return tmp_path


def test_streamer_fallback_local_dir(st_dir):
    out = load_safetensors_streamer(str(st_dir))
    assert set(out) == {"a", "b", "c"}
    assert torch.equal(out["a"], torch.ones(4))


def test_single_file(st_dir):
    out = load_safetensors_streamer(str(st_dir / "m1.safetensors"))
    assert set(out) == {"a", "b"}


def test_auto_local_nonzero_rank_mmaps(st_dir, monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "1")
    out = load_safetensors_auto(str(st_dir))
    assert set(out) == {"a", "b", "c"}


def test_auto_mount_rewrite(st_dir, monkeypatch):
    monkeypatch.setenv("ARCSTORE_S3_MOUNTS", f"bkt={st_dir}")
    import arcstore

    arcstore.refresh_mounts()
    out = load_safetensors_auto("s3://bkt")
    assert set(out) == {"a", "b", "c"}


def test_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_safetensors_streamer(str(tmp_path))
