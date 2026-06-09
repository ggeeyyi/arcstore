import os

import arcstore


def _make_ckpt_bucket(fake_s3_root):
    base = fake_s3_root / "bkt" / "run" / "checkpoints" / "checkpoint_model_000100"
    base.mkdir(parents=True)
    (base / "model.pt").write_bytes(b"primary")
    (base / "model_ema.pt").write_bytes(b"ema")
    return base


def test_stage_s3_with_siblings(fake_s5cmd, tmp_path):
    _make_ckpt_bucket(fake_s5cmd)
    uri = "s3://bkt/run/checkpoints/checkpoint_model_000100/model.pt"
    staged = arcstore.stage_to_local(uri, siblings=("model_ema.pt", "absent.pt"))
    assert staged != uri
    assert open(staged, "rb").read() == b"primary"
    # Sibling landed next to the primary; the absent one was best-effort.
    d = os.path.dirname(staged)
    assert open(os.path.join(d, "model_ema.pt"), "rb").read() == b"ema"
    assert not os.path.exists(os.path.join(d, "absent.pt"))
    assert os.path.isfile(os.path.join(d, ".stage_done"))


def test_stage_s3_hit_is_fast_path(fake_s5cmd):
    _make_ckpt_bucket(fake_s5cmd)
    uri = "s3://bkt/run/checkpoints/checkpoint_model_000100/model.pt"
    first = arcstore.stage_to_local(uri)
    second = arcstore.stage_to_local(uri)
    assert first == second


def test_stage_s3_failure_returns_original(fake_s5cmd, monkeypatch):
    # Every download backend fails -> the original s3 path comes back.
    monkeypatch.setattr(
        "arcstore.s3cli.download_file",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("backend down")),
    )
    uri = "s3://bkt/none/model.pt"
    assert arcstore.stage_to_local(uri) == uri


def test_stage_s3_mount_fallback(monkeypatch, fake_s3_root, fake_s5cmd):
    """Download backends down, but the bucket is mounted -> mount path."""
    monkeypatch.setattr(
        "arcstore.s3cli.download_file",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("backend down")),
    )
    base = _make_ckpt_bucket(fake_s3_root)
    monkeypatch.setenv("ARCSTORE_S3_MOUNTS", f"bkt={fake_s3_root / 'bkt'}")
    arcstore.refresh_mounts()
    uri = "s3://bkt/run/checkpoints/checkpoint_model_000100/model.pt"
    staged = arcstore.stage_to_local(uri)
    assert staged == str(base / "model.pt")


def test_stage_local_dir(tmp_path, monkeypatch):
    src = tmp_path / "slowmount" / "step"
    src.mkdir(parents=True)
    (src / "model.pt").write_bytes(b"w")
    (src / "model_ema.pt").write_bytes(b"e")
    staged = arcstore.stage_to_local(str(src / "model.pt"))
    assert staged != str(src / "model.pt")
    assert open(staged, "rb").read() == b"w"
    # Whole flat dir copied by default.
    assert os.path.isfile(os.path.join(os.path.dirname(staged), "model_ema.pt"))


def test_stage_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCSTORE_CACHE_ENABLE", "0")
    src = tmp_path / "d"
    src.mkdir()
    (src / "f.pt").write_bytes(b"x")
    assert arcstore.stage_to_local(str(src / "f.pt")) == str(src / "f.pt")


def test_stage_prefix_whitelist(tmp_path, monkeypatch):
    src = tmp_path / "d"
    src.mkdir()
    (src / "f.pt").write_bytes(b"x")
    monkeypatch.setenv("ARCSTORE_STAGE_PREFIXES", "/definitely/elsewhere")
    assert arcstore.stage_to_local(str(src / "f.pt")) == str(src / "f.pt")


def test_ensure_local_file_download_and_hit(fake_s5cmd, caplog):
    base = fake_s5cmd / "bkt" / "meta"
    base.mkdir(parents=True)
    (base / "manifest.jsonl").write_text('{"a": 1}\n')
    uri = "s3://bkt/meta/manifest.jsonl"
    p1 = arcstore.ensure_local_file(uri, label="manifest")
    assert open(p1).read() == '{"a": 1}\n'
    import logging

    with caplog.at_level(logging.INFO):
        p2 = arcstore.ensure_local_file(uri, label="manifest")
    assert p2 == p1
    assert any("cache hit" in r.message for r in caplog.records)


def test_ensure_local_file_mount_shortcircuit(monkeypatch, fake_s3_root, fake_s5cmd):
    base = fake_s3_root / "bkt" / "meta"
    base.mkdir(parents=True)
    (base / "m.jsonl").write_text("{}\n")
    monkeypatch.setenv("ARCSTORE_S3_MOUNTS", f"bkt={fake_s3_root / 'bkt'}")
    arcstore.refresh_mounts()
    p = arcstore.ensure_local_file("s3://bkt/meta/m.jsonl")
    assert p == str(base / "m.jsonl")  # no copy, served straight off the mount


def test_ensure_local_file_local_passthrough(tmp_path):
    p = tmp_path / "f.jsonl"
    p.write_text("{}")
    assert arcstore.ensure_local_file(str(p)) == str(p)


def test_lru_eviction(fake_s5cmd, tmp_path, monkeypatch):
    """Stage two ~1 MiB ckpts with a tiny budget; the older one is evicted."""
    import time

    for step in ("a", "b"):
        d = fake_s5cmd / "bkt" / step / "ck"
        d.mkdir(parents=True)
        (d / "model.pt").write_bytes(b"x" * (1024 * 1024))
    monkeypatch.setenv("ARCSTORE_CACHE_BUDGET_GIB", str(1.5 / 1024))  # ~1.5 MiB
    p_a = arcstore.stage_to_local("s3://bkt/a/ck/model.pt")
    time.sleep(0.05)
    p_b = arcstore.stage_to_local("s3://bkt/b/ck/model.pt")
    assert os.path.isfile(p_b)
    assert not os.path.exists(p_a)  # evicted to make room
