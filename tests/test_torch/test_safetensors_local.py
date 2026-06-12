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


def test_auto_mount_default_uses_direct_s3(st_dir, monkeypatch):
    monkeypatch.setenv("ARCSTORE_S3_MOUNTS", f"bkt={st_dir}")
    import arcstore
    from arcstore.torch import safetensors as st_mod

    arcstore.refresh_mounts()
    seen = {}

    def fake_streamer(path, **kwargs):
        seen["path"] = path
        return {"ok": torch.ones(1)}

    monkeypatch.setattr(st_mod, "load_safetensors_streamer", fake_streamer)
    out = load_safetensors_auto("s3://bkt")
    assert set(out) == {"ok"}
    assert seen["path"] == "s3://bkt"


def test_auto_mount_rewrite(st_dir, monkeypatch):
    monkeypatch.setenv("ARCSTORE_S3_MOUNTS", f"bkt={st_dir}")
    import arcstore

    arcstore.refresh_mounts()
    out = load_safetensors_auto("s3://bkt", read_policy="mount")
    assert set(out) == {"a", "b", "c"}


def test_auto_mount_rewrite_skips_streamer(st_dir, monkeypatch):
    """FUSE-mounted s3:// must mmap, NOT call run:ai streamer (deadlock)."""
    monkeypatch.setenv("ARCSTORE_S3_MOUNTS", f"bkt={st_dir}")
    import arcstore
    from arcstore.torch import safetensors as st_mod

    arcstore.refresh_mounts()

    called = {"n": 0}

    def _boom(*a, **kw):
        called["n"] += 1
        raise AssertionError("streamer must not be called on mount-rewritten path")

    monkeypatch.setattr(st_mod, "load_safetensors_streamer", _boom)
    out = load_safetensors_auto("s3://bkt", read_policy="mount")
    assert set(out) == {"a", "b", "c"}
    assert called["n"] == 0


def test_auto_mount_rewrite_nonzero_rank_also_mmaps(st_dir, monkeypatch):
    """Even rank 0 must mmap on a mount — no streamer."""
    monkeypatch.setenv("ARCSTORE_S3_MOUNTS", f"bkt={st_dir}")
    monkeypatch.setenv("LOCAL_RANK", "0")
    import arcstore
    from arcstore.torch import safetensors as st_mod

    arcstore.refresh_mounts()
    monkeypatch.setattr(
        st_mod,
        "load_safetensors_streamer",
        lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("streamer must not be called on mount-rewritten path")
        ),
    )
    out = load_safetensors_auto("s3://bkt", read_policy="mount")
    assert set(out) == {"a", "b", "c"}


def test_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_safetensors_streamer(str(tmp_path))
