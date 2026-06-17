import pytest

torch = pytest.importorskip("torch")

from arcstore import load_checkpoint, save_checkpoint  # noqa: E402


def test_local_save_load_roundtrip(tmp_path):
    path = str(tmp_path / "sub" / "ckpt.pt")
    save_checkpoint(path, "blob", obj={"step": 7, "w": torch.ones(3)})

    blob = load_checkpoint(path, "blob")
    assert blob["step"] == 7
    assert torch.equal(blob["w"], torch.ones(3))


def test_s3_save_load_roundtrip(fake_s5cmd):
    uri = "s3://bkt/run/ckpt.pt"
    save_checkpoint(uri, "blob", obj={"step": 3, "w": torch.full((2,), 2.0)})

    assert (fake_s5cmd / "bkt" / "run" / "ckpt.pt").is_file()

    blob = load_checkpoint(uri, "blob")
    assert blob["step"] == 3
    assert torch.equal(blob["w"], torch.full((2,), 2.0))
