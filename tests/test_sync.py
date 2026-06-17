"""arcstore-sync: config resolution, .git exclusion, submodule guard (no S3)."""
from __future__ import annotations

import argparse

import pytest

from arcstore import sync


def _write(p, text):
    p.write_text(text)
    return p


def test_resolve_config_defaults(tmp_path):
    _write(tmp_path / "pyproject.toml", '[project]\nname = "my-train"\n')
    cfg = sync.resolve_config(tmp_path, {"KOALA_USER": "wbhu"})
    assert cfg.project == "my-train"
    assert cfg.code == "s3://arcwm-code-us-west-2/wbhu/code/my-train"


def test_resolve_config_falls_back_to_dirname(tmp_path):
    cfg = sync.resolve_config(tmp_path, {"USER": "u"})  # no pyproject
    assert cfg.project == tmp_path.name


def test_resolve_config_env_overrides(tmp_path):
    _write(tmp_path / "pyproject.toml", '[project]\nname = "p"\n')
    cfg = sync.resolve_config(
        tmp_path,
        {
            "KOALA_USER": "wbhu",
            "ARCSTORE_CODE_S3": "s3://b/x/code",
            "ARCSTORE_SYNC_EXCLUDE": "foo/*, bar.txt",
        },
    )
    assert cfg.code == "s3://b/x/code"
    assert "foo/*" in cfg.excludes and "bar.txt" in cfg.excludes


def test_resolve_config_bucket_override(tmp_path):
    cfg = sync.resolve_config(tmp_path, {"USER": "u", "ARCSTORE_SYNC_BUCKET": "s3://my-bkt"})
    assert cfg.code == f"s3://my-bkt/u/code/{tmp_path.name}"


def test_git_excluded_by_default():
    for pat in (".git/*", "*/.git/*", "*/.git"):
        assert pat in sync.DEFAULT_EXCLUDES
    for d in (".venv/*", "*/__pycache__/*", "wandb/*", "outputs/*", ".env"):
        assert d in sync.DEFAULT_EXCLUDES


def test_exclude_args():
    assert sync._exclude_args(["a/*", "b"]) == ["--exclude", "a/*", "--exclude", "b"]


def test_empty_submodules(tmp_path):
    _write(
        tmp_path / ".gitmodules",
        '[submodule "vendor/arc-toolkit"]\n\tpath = vendor/arc-toolkit\n'
        "\turl = git@example.com:x/arc_toolkit.git\n"
        '[submodule "third_party/foo"]\n\tpath = third_party/foo\n\turl = x\n',
    )
    (tmp_path / "vendor" / "arc-toolkit").mkdir(parents=True)
    (tmp_path / "vendor" / "arc-toolkit" / "pyproject.toml").write_text("x")  # populated
    (tmp_path / "third_party" / "foo").mkdir(parents=True)  # empty -> flagged
    assert sync.empty_submodules(tmp_path) == ["third_party/foo"]


def test_empty_submodules_none_when_no_gitmodules(tmp_path):
    assert sync.empty_submodules(tmp_path) == []


def test_push_refuses_empty_submodule(tmp_path):
    _write(tmp_path / ".gitmodules", '[submodule "v"]\n\tpath = vendor/x\n\turl = y\n')
    (tmp_path / "vendor" / "x").mkdir(parents=True)  # empty
    cfg = sync.Config("p", "u", "s3://b/code/p", [], tmp_path)
    args = argparse.Namespace(dry_run=False, yes=True, allow_empty_submodules=False)
    with pytest.raises(SystemExit, match="not checked out"):
        sync.cmd_push(cfg, args)
