"""Torch extension layer — requires the ``arcstore[torch]`` extra."""
try:
    import torch as _torch  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "arcstore.torch requires torch; install with: pip install 'arcstore[torch]'"
    ) from e

from .accelerate import (
    load_accelerate_state,
    load_deepspeed_state,
    save_accelerate_state,
    save_deepspeed_state,
)
from .dcp import (
    dcp_dir_exists,
    load_full_state,
    patch_dcp_wrap_exception_py313,
    prime_optim_state,
    save_full_state,
)
from .load import load_ckpt
from .safetensors import (
    load_safetensors_auto,
    load_safetensors_streamer,
)
from .scatter import ScatterPtDataset, reservoir_shuffle
from .wds import build_wds_dataset, expand_urls, shard_urls, tar_url

__all__ = [
    "ScatterPtDataset",
    "build_wds_dataset",
    "dcp_dir_exists",
    "expand_urls",
    "load_accelerate_state",
    "load_ckpt",
    "load_deepspeed_state",
    "load_full_state",
    "load_safetensors_auto",
    "load_safetensors_streamer",
    "patch_dcp_wrap_exception_py313",
    "prime_optim_state",
    "reservoir_shuffle",
    "save_accelerate_state",
    "save_deepspeed_state",
    "save_full_state",
    "shard_urls",
    "tar_url",
]
