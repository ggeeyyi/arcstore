import pytest

import arcstore


def test_upload_file_roundtrip(fake_s5cmd, tmp_path):
    src = tmp_path / "model.pt"
    src.write_bytes(b"weights")
    arcstore.upload_file(str(src), "s3://bkt/ckpt/model.pt")
    assert (fake_s5cmd / "bkt" / "ckpt" / "model.pt").read_bytes() == b"weights"


def test_upload_dir_roundtrip(fake_s5cmd, tmp_path):
    d = tmp_path / "step"
    (d / "nested").mkdir(parents=True)
    (d / "model.pt").write_bytes(b"m")
    (d / "nested" / "x.json").write_text("{}")
    arcstore.upload_dir(str(d), "s3://bkt/ckpts/step")
    assert (fake_s5cmd / "bkt" / "ckpts" / "step" / "model.pt").read_bytes() == b"m"
    assert (fake_s5cmd / "bkt" / "ckpts" / "step" / "nested" / "x.json").exists()


def test_async_upload_and_flush(fake_s5cmd, tmp_path):
    src = tmp_path / "a.bin"
    src.write_bytes(b"abc")
    arcstore.upload_file_async(str(src), "s3://bkt/async/a.bin")
    arcstore.wait_for_uploads(timeout_s=30)
    assert (fake_s5cmd / "bkt" / "async" / "a.bin").read_bytes() == b"abc"


def test_async_failure_surfaces_in_wait(fake_s5cmd, tmp_path, no_cli):
    # Source file missing + no CLI tools: boto3 raises FileNotFoundError in
    # the background; wait_for_uploads must re-raise it on the main thread.
    arcstore.upload_file_async(str(tmp_path / "missing.bin"), "s3://bkt/x/missing.bin")
    with pytest.raises(Exception):
        arcstore.wait_for_uploads(timeout_s=30)


def test_wait_for_uploads_noop():
    arcstore.wait_for_uploads()  # no pending futures — must not raise


def test_atexit_flush_logs_not_raises(fake_s5cmd, tmp_path, no_cli, caplog):
    from arcstore.uploads import _atexit_flush

    arcstore.upload_file_async(str(tmp_path / "missing.bin"), "s3://bkt/x/m.bin")
    _atexit_flush()  # must swallow + log, not raise
    assert any("atexit" in r.message for r in caplog.records)


def test_download_dir_requires_marker(fake_s5cmd, tmp_path):
    src = fake_s5cmd / "bkt" / "dcp"
    src.mkdir(parents=True)
    (src / "shard.bin").write_bytes(b"x")

    with pytest.raises(FileNotFoundError):
        arcstore.download_dir(
            "s3://bkt/dcp",
            str(tmp_path / "out"),
            required_files=(".metadata",),
            retries=1,
        )


def test_download_dir_incomplete_cli_wraps_boto3_failure(fake_s5cmd, tmp_path, monkeypatch):
    src = fake_s5cmd / "bkt" / "dcp"
    src.mkdir(parents=True)
    (src / "shard.bin").write_bytes(b"x")
    monkeypatch.setattr(
        "arcstore.uploads._boto3_download_dir",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bad credentials")),
    )

    with pytest.raises(FileNotFoundError, match="complete local dir"):
        arcstore.download_dir(
            "s3://bkt/dcp",
            str(tmp_path / "out"),
            required_files=(".metadata",),
            retries=1,
        )


def test_download_dir_marker_passes(fake_s5cmd, tmp_path):
    src = fake_s5cmd / "bkt" / "dcp"
    src.mkdir(parents=True)
    (src / ".metadata").write_bytes(b"meta")
    (src / "shard.bin").write_bytes(b"x")

    out = tmp_path / "out"
    arcstore.download_dir(
        "s3://bkt/dcp",
        str(out),
        required_files=(".metadata",),
        retries=1,
    )

    assert (out / ".metadata").read_bytes() == b"meta"
