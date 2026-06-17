"""Torch extension layer — requires the ``arcstore[torch]`` extra."""
from __future__ import annotations

if __name__ == "torch":
    import importlib.machinery as _machinery
    import importlib.util as _util
    import os as _os
    import sys as _sys

    _src_root = _os.path.dirname(_os.path.dirname(__file__))
    _src_root = _os.path.abspath(_src_root)
    _search_path = [
        p
        for p in _sys.path
        if _os.path.abspath(p or _os.getcwd()) != _src_root
    ]
    _spec = _machinery.PathFinder.find_spec("torch", _search_path)
    if _spec is None or _spec.loader is None:
        raise ModuleNotFoundError("No module named 'torch'")

    _module = _util.module_from_spec(_spec)
    _sys.modules[__name__] = _module
    _spec.loader.exec_module(_module)
    globals().update(_module.__dict__)
else:
    try:
        import torch as _torch  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "arcstore.torch requires torch; install with: pip install 'arcstore[torch]'"
        ) from e

    from .._env import cache_dir
    from .accelerate_ckpt import load_accelerate_state, save_accelerate_state
    from .dcp import dcp_dir_exists, load_full_state, save_full_state
    from .load import load_ckpt
    from .manager import CheckpointManager
    from .observability import EMA, PerfTracker, StageTimer, get_gpu_memory_stats
    from .runtime import (
        RNGState,
        barrier,
        get_local_rank,
        get_rank,
        get_world_size,
        is_local_main,
        is_main,
    )
    from .safetensors import (
        load_pretrained,
        load_safetensors_auto,
        load_safetensors_streamer,
        save_safetensors_weights,
    )
    from .scatter import ScatterPtDataset
    from .synthetic import SyntheticDataset
    from .wds import build_wds_dataset, expand_urls, shard_urls, tar_url

    __all__ = [
        "EMA",
        "CheckpointManager",
        "PerfTracker",
        "RNGState",
        "ScatterPtDataset",
        "StageTimer",
        "SyntheticDataset",
        "barrier",
        "build_wds_dataset",
        "cache_dir",
        "dcp_dir_exists",
        "expand_urls",
        "get_gpu_memory_stats",
        "get_local_rank",
        "get_rank",
        "get_world_size",
        "is_local_main",
        "is_main",
        "load_accelerate_state",
        "load_ckpt",
        "load_full_state",
        "load_pretrained",
        "load_safetensors_auto",
        "load_safetensors_streamer",
        "save_accelerate_state",
        "save_full_state",
        "save_safetensors_weights",
        "shard_urls",
        "tar_url",
    ]
