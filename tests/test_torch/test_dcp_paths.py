import pytest

pytest.importorskip("torch")

from arcstore.torch import dcp as dcp_mod
from arcstore.torch.dcp import _stage_dir_for_s3


def test_s3_stage_dirs_include_uri_hash_to_avoid_basename_collisions():
    a = _stage_dir_for_s3("/tmp/arcstore/dcp", "s3://bucket-a/runs/ckpt/dcp")
    b = _stage_dir_for_s3("/tmp/arcstore/dcp", "s3://bucket-b/runs/ckpt/dcp")

    assert a != b
    assert a.endswith("__dcp")
    assert b.endswith("__dcp")


def test_dcp_dir_exists_requires_metadata(fake_s5cmd):
    base = fake_s5cmd / "bkt" / "run" / "dcp"
    base.mkdir(parents=True)
    (base / "extras_rank0.pt").write_bytes(b"x")  # step sidecar, not a completeness signal

    assert dcp_mod.dcp_dir_exists("s3://bkt/run/dcp") is False

    (base / ".metadata").write_bytes(b"meta")
    assert dcp_mod.dcp_dir_exists("s3://bkt/run/dcp") is True
