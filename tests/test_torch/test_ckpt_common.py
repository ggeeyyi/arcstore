"""Shared checkpoint core (_ckpt_common): app-state validation + extras sidecar.

These exercise the offline-testable pieces of the unified DCP core that both
CheckpointManager and save_full_state build on (the DCP save/load primitives
themselves need a real process group and are covered at the integration level).
"""
import pytest

torch = pytest.importorskip("torch")

from arcstore.torch._ckpt_common import (  # noqa: E402
    make_app_state,
    read_extras,
    write_extras,
)


class _Stateful:
    def __init__(self, v=0):
        self.v = v

    def state_dict(self):
        return {"v": self.v}

    def load_state_dict(self, sd):
        self.v = sd["v"]


def test_make_app_state_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        make_app_state([object(), object()], [object()])


def test_extras_roundtrip_local(tmp_path):
    dest = tmp_path / "checkpoint-7"
    write_extras(str(dest), 7, {"scheduler": _Stateful(5), "note": "hi"})

    assert (dest / "extras_rank0.pt").is_file()  # manager on-disk layout

    target = _Stateful(0)
    step = read_extras(str(dest), {"scheduler": target})
    assert step == 7
    assert target.v == 5  # stateful extra restored in place


def test_read_extras_step_from_dirname_when_missing(tmp_path):
    dest = tmp_path / "checkpoint-42"
    dest.mkdir()
    assert read_extras(str(dest), {}) == 42  # no sidecar -> step parsed from dir name
