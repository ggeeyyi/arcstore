import os
import subprocess
import sys

import arcstore


def test_cli_tee_roundtrip(fake_s5cmd, fake_s3_root, tmp_path):
    local = tmp_path / "run.log"
    env = os.environ.copy()
    src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "arcstore.logtee",
            "s3://bkt/logs/run.log",
            "--local",
            str(local),
            "--interval",
            "1",
        ],
        input=b"line one\nline two\n",
        capture_output=True,
        env=env,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr.decode()
    # stdin passed through to stdout
    assert proc.stdout == b"line one\nline two\n"
    # appended locally
    assert local.read_bytes() == b"line one\nline two\n"
    # final flush uploaded to (fake) S3
    remote = fake_s3_root / "bkt" / "logs" / "run.log"
    assert remote.read_bytes() == b"line one\nline two\n"


def test_cli_rejects_non_s3(tmp_path):
    src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
    env = os.environ.copy()
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-m", "arcstore.logtee", "/not/s3/run.log"],
        input=b"",
        capture_output=True,
        env=env,
        timeout=30,
    )
    assert proc.returncode != 0


def test_logtee_inprocess(fake_s5cmd, fake_s3_root, tmp_path, capsys):
    local = tmp_path / "t.log"
    tee = arcstore.LogTee(str(local), "s3://bkt/logs/t.log", interval_s=60).install()
    try:
        print("hello tee")
    finally:
        tee.close()
    assert "hello tee" in local.read_text()
    remote = fake_s3_root / "bkt" / "logs" / "t.log"
    assert "hello tee" in remote.read_text()


def test_logtee_requires_s3(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        arcstore.LogTee(str(tmp_path / "x.log"), "/local/dest.log")
