import arcstore


def _mk_ckpts(root, steps, *, with_model=True):
    for s in steps:
        d = root / f"checkpoint_model_{s:06d}"
        d.mkdir(parents=True)
        if with_model:
            (d / "model.pt").write_bytes(b"w")


def test_find_latest_local(tmp_path):
    _mk_ckpts(tmp_path, [100, 2000, 500])
    hit = arcstore.find_latest_ckpt(str(tmp_path))
    assert hit == (f"{tmp_path}/checkpoint_model_002000/model.pt", 2000)


def test_find_latest_s3(fake_s5cmd):
    base = fake_s5cmd / "bkt" / "run" / "checkpoints"
    _mk_ckpts(base, [100, 300])
    hit = arcstore.find_latest_ckpt("s3://bkt/run/checkpoints")
    assert hit == ("s3://bkt/run/checkpoints/checkpoint_model_000300/model.pt", 300)


def test_partial_upload_falls_back_to_previous(fake_s5cmd):
    base = fake_s5cmd / "bkt" / "run" / "checkpoints"
    _mk_ckpts(base, [100])
    # Newest step is a partial upload: dir exists, no model.pt.
    (base / "checkpoint_model_000200").mkdir()
    (base / "checkpoint_model_000200" / "other.txt").write_text("x")
    hit = arcstore.find_latest_ckpt("s3://bkt/run/checkpoints")
    assert hit == ("s3://bkt/run/checkpoints/checkpoint_model_000100/model.pt", 100)


def test_empty_prefix_returns_none(fake_s5cmd, tmp_path):
    assert arcstore.find_latest_ckpt("s3://bkt/none") is None
    assert arcstore.find_latest_ckpt(str(tmp_path / "missing")) is None


def test_custom_pattern_and_required_file(tmp_path):
    for s in (10, 20):
        d = tmp_path / f"step-{s}"
        d.mkdir()
        (d / "weights.safetensors").write_bytes(b"w")
    hit = arcstore.find_latest_ckpt(
        str(tmp_path),
        pattern=r"step-(\d+)",
        required_file="weights.safetensors",
    )
    assert hit == (f"{tmp_path}/step-20/weights.safetensors", 20)


def test_required_file_none_returns_dir(tmp_path):
    _mk_ckpts(tmp_path, [42], with_model=False)
    hit = arcstore.find_latest_ckpt(str(tmp_path), required_file=None)
    assert hit == (f"{tmp_path}/checkpoint_model_000042", 42)


def test_s3_via_mount(monkeypatch, fake_s3_root, fake_s5cmd):
    base = fake_s3_root / "bkt" / "run" / "checkpoints"
    _mk_ckpts(base, [7])
    monkeypatch.setenv("ARCSTORE_S3_MOUNTS", f"bkt={fake_s3_root / 'bkt'}")
    arcstore.refresh_mounts()
    hit = arcstore.find_latest_ckpt("s3://bkt/run/checkpoints")
    # Listing went through the mount, but the result stays an s3:// URI.
    assert hit == ("s3://bkt/run/checkpoints/checkpoint_model_000007/model.pt", 7)
