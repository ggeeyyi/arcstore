import pytest

pytest.importorskip("torch")

import arcstore._env as env_mod
from arcstore.torch import dcp as dcp_mod
from arcstore.torch.dcp import _stage_dir_for_s3


def test_dcp_save_stage_root_prefers_local_ssd(monkeypatch):
    monkeypatch.delenv("ARCSTORE_DCP_SAVE_STAGE_DIR", raising=False)
    monkeypatch.setattr(env_mod, "_local_ssd_usable", lambda root="/local-ssd": True)

    assert dcp_mod._save_stage_root() == "/local-ssd/arcstore/dcp_save"


def test_dcp_save_stage_root_falls_back_to_tmp(monkeypatch):
    monkeypatch.delenv("ARCSTORE_DCP_SAVE_STAGE_DIR", raising=False)
    monkeypatch.setattr(env_mod, "_local_ssd_usable", lambda root="/local-ssd": False)

    assert dcp_mod._save_stage_root() == "/tmp/arcstore/dcp_save"


def test_dcp_save_stage_root_env_override_wins(monkeypatch, tmp_path):
    override = tmp_path / "dcp-save"
    monkeypatch.setenv("ARCSTORE_DCP_SAVE_STAGE_DIR", str(override))
    monkeypatch.setattr(env_mod, "_local_ssd_usable", lambda root="/local-ssd": True)

    assert dcp_mod._save_stage_root() == str(override)


def test_save_train_meta_uses_safe_temp_under_save_stage(monkeypatch, tmp_path):
    stage_root = tmp_path / "dcp-save"
    seen = {}

    monkeypatch.setenv("ARCSTORE_DCP_SAVE_STAGE_DIR", str(stage_root))

    def fake_torch_save(obj, path):
        with open(path, "wb") as f:
            f.write(repr(obj).encode("utf-8"))

    monkeypatch.setattr(dcp_mod.torch, "save", fake_torch_save)

    def fake_upload(local, dest):
        seen["local"] = local
        seen["dest"] = dest
        seen["exists_during_upload"] = dcp_mod.os.path.isfile(local)

    monkeypatch.setattr(dcp_mod, "upload_file", fake_upload)

    dcp_mod._save_train_meta("s3://bkt/run/dcp", 12)

    assert seen["dest"] == "s3://bkt/run/dcp/train_meta.pt"
    assert seen["exists_during_upload"] is True
    assert seen["local"].startswith(str(stage_root))
    assert not dcp_mod.os.path.exists(seen["local"])


def test_s3_stage_dirs_include_uri_hash_to_avoid_basename_collisions():
    a = _stage_dir_for_s3("/tmp/arcstore/dcp", "s3://bucket-a/runs/ckpt/dcp")
    b = _stage_dir_for_s3("/tmp/arcstore/dcp", "s3://bucket-b/runs/ckpt/dcp")

    assert a != b
    assert a.endswith("__dcp")
    assert b.endswith("__dcp")


def test_dcp_dir_exists_requires_metadata(fake_s5cmd):
    base = fake_s5cmd / "bkt" / "run" / "dcp"
    base.mkdir(parents=True)
    (base / "train_meta.pt").write_bytes(b"x")

    assert dcp_mod.dcp_dir_exists("s3://bkt/run/dcp") is False

    (base / ".metadata").write_bytes(b"meta")
    assert dcp_mod.dcp_dir_exists("s3://bkt/run/dcp") is True
