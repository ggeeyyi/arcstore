"""Optional HF accelerate adapter: one-call DDP / DeepSpeed / FSDP setup.

Ported from ``arc_toolkit.integrations.accelerate``. Requires the ``accelerate``
extra (``pip install 'arcstore[accelerate]'``). The rest of arcstore never
depends on this module — :class:`arcstore.torch.CheckpointManager` and the
runtime primitives work with raw ``torch.distributed`` just as well.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from accelerate import Accelerator

if TYPE_CHECKING:
    from accelerate import FullyShardedDataParallelPlugin

__all__ = ["ParallelBackend", "RunContext", "build_fsdp_plugin", "init_distributed"]

ParallelBackend = Literal["ddp", "deepspeed", "fsdp"]


@dataclass
class RunContext:
    """Everything :func:`init_distributed` established, in one handle."""

    accelerator: Accelerator
    backend: ParallelBackend
    world_size: int
    num_nodes: int
    gpus_per_node: int

    @property
    def is_main(self) -> bool:
        return self.accelerator.is_main_process

    @property
    def device(self) -> torch.device:
        return self.accelerator.device


def build_fsdp_plugin(
    *,
    sharding_strategy: Literal[
        "FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD", "_HYBRID_SHARD_ZERO2"
    ] = "SHARD_GRAD_OP",
    auto_wrap_policy: Literal[
        "transformer_based_wrap", "size_based_wrap", "no_wrap"
    ] = "size_based_wrap",
    min_num_params: int = int(1e7),
    state_dict_type: Literal[
        "FULL_STATE_DICT", "LOCAL_STATE_DICT", "SHARDED_STATE_DICT"
    ] = "SHARDED_STATE_DICT",
    use_orig_params: bool = True,
    cpu_offload: bool = False,
    fsdp_version: int = 1,
    **extra,
) -> "FullyShardedDataParallelPlugin":
    """FSDP plugin preset (~ZeRO-2 with sharded DCP checkpoints); every knob overridable.

    Extra keyword arguments are forwarded verbatim to accelerate's
    ``FullyShardedDataParallelPlugin``.
    """
    from accelerate import FullyShardedDataParallelPlugin
    from torch.distributed.fsdp import ShardingStrategy, StateDictType

    return FullyShardedDataParallelPlugin(
        fsdp_version=fsdp_version,
        sharding_strategy=ShardingStrategy[sharding_strategy]
        if isinstance(sharding_strategy, str)
        else sharding_strategy,
        state_dict_type=StateDictType[state_dict_type]
        if isinstance(state_dict_type, str)
        else state_dict_type,
        auto_wrap_policy=auto_wrap_policy,
        min_num_params=min_num_params,
        use_orig_params=use_orig_params,
        cpu_offload=cpu_offload,
        **extra,
    )


def init_distributed(
    *,
    backend: ParallelBackend = "ddp",
    gradient_accumulation_steps: int = 1,
    deepspeed_config: str | dict | None = None,
    mixed_precision: Literal["no", "fp16", "bf16", "fp8"] = "bf16",
    log_with: str | None = None,
    seed: int | None = None,
    fsdp_plugin: "FullyShardedDataParallelPlugin | None" = None,
) -> RunContext:
    """Initialize accelerate for the chosen parallel backend and return a RunContext.

    NCCL / fabric tuning is intentionally not handled here: export the relevant
    env vars (FI_PROVIDER, NCCL_*, ...) before launch — the training launcher
    owns that.
    """
    from accelerate.utils import set_seed

    if backend == "deepspeed" and deepspeed_config is None:
        raise ValueError("deepspeed_config is required when backend='deepspeed'")
    if deepspeed_config is not None and backend != "deepspeed":
        raise ValueError(f"deepspeed_config requires backend='deepspeed', got {backend!r}")
    if fsdp_plugin is not None and backend != "fsdp":
        raise ValueError(f"fsdp_plugin requires backend='fsdp', got {backend!r}")

    if seed is not None:
        set_seed(seed)

    ds_plugin = None
    if backend == "fsdp":
        import torch.distributed as dist

        if "RANK" in os.environ and not dist.is_initialized():
            # DCP async_save de-stages on CPU and needs a gloo backend next to nccl.
            dist.init_process_group(backend="cuda:nccl,cpu:gloo")
        fsdp_plugin = fsdp_plugin or build_fsdp_plugin()
    elif backend == "deepspeed":
        from accelerate import DeepSpeedPlugin

        ds_plugin = DeepSpeedPlugin(
            hf_ds_config=deepspeed_config,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        deepspeed_plugin=ds_plugin,
        fsdp_plugin=fsdp_plugin,
        log_with=log_with,
        mixed_precision=mixed_precision,
    )

    gpus_per_node = max(torch.cuda.device_count(), 1)
    num_nodes = max(accelerator.num_processes // gpus_per_node, 1)

    return RunContext(
        accelerator=accelerator,
        backend=backend,
        world_size=accelerator.num_processes,
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
    )
