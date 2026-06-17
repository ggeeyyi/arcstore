"""arcstore.torch.runtime: coordination primitives, cache dir, RNG state."""
from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from arcstore.torch import runtime  # noqa: E402


def test_rank_world_defaults(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert runtime.get_rank() == 0
    assert runtime.get_world_size() == 1
    assert runtime.get_local_rank() == 0
    assert runtime.is_main()
    assert runtime.is_local_main()


def test_rank_world_from_env(monkeypatch):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "1")
    assert runtime.get_rank() == 3
    assert runtime.get_world_size() == 8
    assert runtime.get_local_rank() == 1
    assert not runtime.is_main()
    assert not runtime.is_local_main()


def test_barrier_noop_without_dist():
    runtime.barrier()  # must not raise without an initialized process group


def test_rng_state_roundtrip():
    rng = runtime.RNGState()
    state = rng.state_dict()
    first = (torch.rand(3), random.random())
    rng.load_state_dict(state)
    second = (torch.rand(3), random.random())
    assert torch.equal(first[0], second[0])
    assert first[1] == second[1]
