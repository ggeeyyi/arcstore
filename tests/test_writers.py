import pytest

import arcstore


def test_write_bytes_local_roundtrip(tmp_path):
    p = tmp_path / "sub" / "a.bin"
    arcstore.write_bytes(str(p), b"hello")
    assert arcstore.read_bytes(str(p)) == b"hello"


def test_write_bytes_s3_roundtrip(fake_s5cmd, tmp_path):
    arcstore.write_bytes("s3://bkt/run/a.bin", b"hello")
    assert (fake_s5cmd / "bkt" / "run" / "a.bin").read_bytes() == b"hello"
    assert arcstore.read_bytes("s3://bkt/run/a.bin") == b"hello"


def test_write_bytes_rejects_non_bytes(tmp_path):
    with pytest.raises(TypeError):
        arcstore.write_bytes(str(tmp_path / "x.txt"), "not bytes")


def test_open_write_local_text(tmp_path):
    p = tmp_path / "deep" / "b.txt"
    with arcstore.open_write(str(p), "w") as f:
        f.write("world")
    assert arcstore.read_bytes(str(p)) == b"world"


def test_open_write_s3_uploads_on_clean_exit(fake_s5cmd):
    with arcstore.open_write("s3://bkt/run/b.bin") as f:
        f.write(b"payload")
    assert (fake_s5cmd / "bkt" / "run" / "b.bin").read_bytes() == b"payload"


def test_open_write_s3_skips_upload_on_exception(fake_s5cmd):
    with pytest.raises(RuntimeError):
        with arcstore.open_write("s3://bkt/run/partial.bin") as f:
            f.write(b"partial")
            raise RuntimeError("boom")
    assert not (fake_s5cmd / "bkt" / "run" / "partial.bin").exists()


def test_open_write_rejects_bad_mode(tmp_path):
    with pytest.raises(ValueError):
        with arcstore.open_write(str(tmp_path / "x"), "rb"):
            pass


def test_put_file_autodetect(fake_s5cmd, tmp_path):
    src = tmp_path / "model.pt"
    src.write_bytes(b"weights")
    arcstore.put(str(src), "s3://bkt/ckpt/model.pt")
    assert (fake_s5cmd / "bkt" / "ckpt" / "model.pt").read_bytes() == b"weights"


def test_put_dir_autodetect(fake_s5cmd, tmp_path):
    d = tmp_path / "step"
    (d / "nested").mkdir(parents=True)
    (d / "model.pt").write_bytes(b"m")
    (d / "nested" / "x.json").write_text("{}")
    arcstore.put(str(d), "s3://bkt/ckpts/step")
    assert (fake_s5cmd / "bkt" / "ckpts" / "step" / "model.pt").read_bytes() == b"m"
    assert (fake_s5cmd / "bkt" / "ckpts" / "step" / "nested" / "x.json").exists()


def test_put_async_flushes(fake_s5cmd, tmp_path):
    src = tmp_path / "a.bin"
    src.write_bytes(b"abc")
    arcstore.put(str(src), "s3://bkt/async/a.bin", async_=True)
    arcstore.wait_for_uploads(timeout_s=30)
    assert (fake_s5cmd / "bkt" / "async" / "a.bin").read_bytes() == b"abc"


def test_put_explicit_recursive_overrides_autodetect(fake_s5cmd, tmp_path):
    d = tmp_path / "step"
    d.mkdir()
    (d / "model.pt").write_bytes(b"m")
    arcstore.put(str(d), "s3://bkt/forced/step", recursive=True)
    assert (fake_s5cmd / "bkt" / "forced" / "step" / "model.pt").read_bytes() == b"m"
