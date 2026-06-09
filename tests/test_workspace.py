import arcstore
from arcstore.workspace import DEFAULT_LOCAL_ROOT


def test_local_workdir_passthrough():
    assert arcstore.split_workdir("/data/exp1") == ("/data/exp1", None)


def test_s3_workdir_derives_local_mirror():
    local, s3 = arcstore.split_workdir("s3://bkt/user/ckpts/run1/")
    assert s3 == "s3://bkt/user/ckpts/run1"
    assert local == f"{DEFAULT_LOCAL_ROOT}/bkt/user/ckpts/run1"


def test_local_root_env_override(monkeypatch):
    monkeypatch.setenv("ARCSTORE_LOCAL_ROOT", "/dev/shm/work")
    local, _ = arcstore.split_workdir("s3://bkt/run")
    assert local == "/dev/shm/work/bkt/run"


def test_local_root_arg_beats_env(monkeypatch):
    monkeypatch.setenv("ARCSTORE_LOCAL_ROOT", "/dev/shm/work")
    local, _ = arcstore.split_workdir("s3://bkt/run", local_root="/scratch")
    assert local == "/scratch/bkt/run"
