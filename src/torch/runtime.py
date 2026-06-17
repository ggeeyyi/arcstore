"""Framework-free distributed coordination + host-environment helpers.

Ported from ``arc_toolkit.runtime``. Everything here works with or without an
initialized ``torch.distributed`` process group and degrades gracefully to
single-process behavior, so training code can call these unconditionally.

The rank/world helpers fall back to the standard launcher env vars
(``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE``) when torch.distributed is not
initialized. :class:`RNGState` and :func:`barrier` are the only torch-touching
pieces; the rest is plain ``os``/env.

The local scratch root is resolved by :func:`arcstore._env.cache_dir`
(``ARCSTORE_CACHE_DIR`` > ``/local-ssd`` > the system temp dir); import it
from there (it is re-exported as ``arcstore.torch.cache_dir``).
"""
from __future__ import annotations

import os
import random

import torch

__all__ = [
    "RNGState",
    "barrier",
    "get_local_rank",
    "get_rank",
    "get_world_size",
    "is_local_main",
    "is_main",
]


def _dist():
    import torch.distributed as dist

    return dist if dist.is_available() and dist.is_initialized() else None


def get_rank() -> int:
    """Global rank: from torch.distributed when initialized, else ``$RANK``, else 0."""
    dist = _dist()
    return dist.get_rank() if dist else int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    """Rank within the node, from ``$LOCAL_RANK`` (0 when unset)."""
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size() -> int:
    """Total process count: torch.distributed when initialized, else ``$WORLD_SIZE``, else 1."""
    dist = _dist()
    return dist.get_world_size() if dist else int(os.environ.get("WORLD_SIZE", "1"))


def is_main() -> bool:
    """True on the single global-rank-0 process."""
    return get_rank() == 0


def is_local_main() -> bool:
    """True on each node's local-rank-0 process (one per node)."""
    return get_local_rank() == 0


def barrier() -> None:
    """Synchronize all ranks; correct for any process-group flavor.

    No-op when torch.distributed is unavailable/uninitialized or world size is
    1. On NCCL-backed groups the local device is passed explicitly so NCCL does
    not have to guess.
    """
    dist = _dist()
    if dist is None or dist.get_world_size() <= 1:
        return
    if "nccl" in str(dist.get_backend()).lower() and torch.cuda.is_available():
        dist.barrier(device_ids=[get_local_rank()])
    else:
        dist.barrier()


class RNGState:
    """Process-local RNG state (python ``random``, torch CPU, torch CUDA, numpy).

    Implements the ``state_dict``/``load_state_dict`` protocol so it can be
    checkpointed like any other component::

        ckpt.save(step, model, optimizer, extras={"rng": RNGState()})
    """

    def state_dict(self) -> dict:
        state: dict = {
            "python": random.getstate(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }
        try:
            import numpy as np

            state["numpy"] = np.random.get_state()
        except ImportError:
            state["numpy"] = None
        return state

    def load_state_dict(self, state: dict) -> None:
        random.setstate(state["python"])
        torch.set_rng_state(state["torch_cpu"])
        if torch.cuda.is_available() and state.get("torch_cuda") is not None:
            torch.cuda.set_rng_state(state["torch_cuda"])
        if state.get("numpy") is not None:
            import numpy as np

            np.random.set_state(state["numpy"])
