import arcstore


def test_jsonl():
    assert arcstore.detect_format("s3://bkt/x/manifest.jsonl") == "jsonl"
    assert arcstore.detect_format("/data/m.jsonl") == "jsonl"


def test_wds():
    assert arcstore.detect_format("/data/shards") == "wds"
    assert arcstore.detect_format("s3://bkt/x/clip-000.tar") == "wds"
    assert arcstore.detect_format("/data/clips-{000..127}.tar") == "wds"
    assert arcstore.detect_format("/data/*.tar") == "wds"


def test_s3_default_scatter():
    assert arcstore.detect_format("s3://bkt/latents/") == "scatter"


def test_local_lmdb(tmp_path):
    (tmp_path / "data.mdb").write_bytes(b"")
    assert arcstore.detect_format(str(tmp_path)) == "lmdb"


def test_local_sharded_lmdb(tmp_path):
    d = tmp_path / "shard-0"
    d.mkdir()
    (d / "data.mdb").write_bytes(b"")
    assert arcstore.detect_format(str(tmp_path)) == "lmdb"


def test_local_scatter(tmp_path):
    (tmp_path / "sample.pt").write_bytes(b"")
    assert arcstore.detect_format(str(tmp_path)) == "scatter"


def test_local_fallback_lmdb(tmp_path):
    assert arcstore.detect_format(str(tmp_path)) == "lmdb"


def test_lmdb_on_mounted_s3(monkeypatch, tmp_path):
    """LMDB over S3 becomes legal when the bucket is FUSE-mounted."""
    mnt = tmp_path / "mnt"
    db = mnt / "datasets" / "mydb"
    db.mkdir(parents=True)
    (db / "data.mdb").write_bytes(b"")
    monkeypatch.setenv("ARCSTORE_S3_MOUNTS", f"bkt={mnt}")
    arcstore.refresh_mounts()
    assert arcstore.detect_format("s3://bkt/datasets/mydb") == "lmdb"
    # Without the mount the same path is scatter (LMDB can't stream from S3).
    monkeypatch.setenv("ARCSTORE_USE_MOUNTS", "0")
    assert arcstore.detect_format("s3://bkt/datasets/mydb") == "scatter"
