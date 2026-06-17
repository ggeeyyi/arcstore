"""CheckpointManager: latest-complete discovery + torn-checkpoint skipping (local, no dist)."""
from __future__ import annotations

import pytest

pytest.importorskip("torch")

from arcstore.torch import CheckpointManager  # noqa: E402
from arcstore.torch.manager import COMPLETE_MARKER  # noqa: E402


def _make_ckpt(root, step, *, complete=True, metadata=False):
    d = root / f"checkpoint-{step}"
    d.mkdir(parents=True)
    (d / "extras_rank0.pt").write_bytes(b"x")
    if complete:
        (d / COMPLETE_MARKER).write_text("ok")
    if metadata:
        (d / ".metadata").write_bytes(b"m")
    return d


def test_latest_checkpoint_none_when_empty(tmp_path):
    mgr = CheckpointManager(local_dir=tmp_path / "ckpts")
    assert mgr.latest_checkpoint() is None


def test_latest_checkpoint_numeric_order(tmp_path):
    root = tmp_path / "ckpts"
    _make_ckpt(root, 5)
    _make_ckpt(root, 100)
    _make_ckpt(root, 20)
    mgr = CheckpointManager(local_dir=root)
    assert mgr.latest_checkpoint().endswith("checkpoint-100")  # not lexicographic "20"


def test_latest_checkpoint_skips_torn(tmp_path):
    root = tmp_path / "ckpts"
    _make_ckpt(root, 10, complete=True)
    _make_ckpt(root, 20, complete=False)  # torn: no marker, no .metadata
    mgr = CheckpointManager(local_dir=root)
    assert mgr.latest_checkpoint().endswith("checkpoint-10")


def test_metadata_counts_as_complete(tmp_path):
    root = tmp_path / "ckpts"
    _make_ckpt(root, 7, complete=False, metadata=True)  # pre-marker DCP layout
    mgr = CheckpointManager(local_dir=root)
    assert mgr.latest_checkpoint().endswith("checkpoint-7")


def test_prune_keeps_previous_complete_when_current_is_torn(tmp_path):
    root = tmp_path / "ckpts"
    _make_ckpt(root, 10, complete=True)
    _make_ckpt(root, 20, complete=True)
    _make_ckpt(root, 30, complete=False)

    mgr = CheckpointManager(local_dir=root, keep_last=1)
    mgr._prune_old(30)

    assert not (root / "checkpoint-10").exists()
    assert (root / "checkpoint-20").exists()
    assert (root / "checkpoint-30").exists()
