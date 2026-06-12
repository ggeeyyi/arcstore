import os

import arcstore


def test_run_storage_layout_for_s3(monkeypatch):
    monkeypatch.setenv("ARCSTORE_LOCAL_ROOT", "/local-root")
    run = arcstore.RunStorage("s3://bkt/user/run1")

    assert run.local_dir == "/local-root/bkt/user/run1"
    assert run.s3_dir == "s3://bkt/user/run1"
    assert run.logs_s3 == "s3://bkt/user/run1/logs"
    assert run.artifacts_s3 == "s3://bkt/user/run1/artifacts"
    assert run.checkpoints_s3 == "s3://bkt/user/run1/checkpoints"


def test_sync_artifacts_uploads_nonempty_dir(fake_s5cmd, tmp_path):
    local = tmp_path / "artifacts"
    local.mkdir()
    (local / "summary.json").write_text("{}")

    arcstore.sync_artifacts(str(local), "s3://bkt/run/artifacts", async_=False)

    assert os.path.isfile(fake_s5cmd / "bkt" / "run" / "artifacts" / "summary.json")
