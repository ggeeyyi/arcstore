"""Scatter ``.pt`` dataset: one ``torch.save`` object per sample.

Lifted from CausalVideoDiffusion ``src/dataset/s3conn.py`` with the
project-specific sample decoding replaced by an injectable ``transform``
callable (receives the raw object bytes, returns the sample).

Read paths by source:

* direct ``s3://`` prefix -> :class:`s3torchconnector.S3IterableDataset`
  (AWS CRT multi-threaded reads, rank/worker sharding via
  ``enable_sharding=True``);
* FUSE-mounted ``s3://`` prefix -> plain local glob over the mount with
  manual rank/worker sharding (kernel readahead, no s3torchconnector
  needed at runtime). ``use_mount=False`` forces direct S3 for A/B runs;
* local directory -> same local glob path.
"""
from __future__ import annotations

import io as _io
import logging
import os
import random
from typing import Any, Callable, Iterable, Iterator, Optional

import torch
from torch.utils.data import IterableDataset

from .._env import aws_region
from .._env import read_policy as _read_policy
from ..location import resolve

logger = logging.getLogger(__name__)

__all__ = ["ScatterPtDataset", "reservoir_shuffle"]


def _default_transform(raw: bytes) -> Any:
    return torch.load(_io.BytesIO(raw), map_location="cpu", weights_only=False)


def reservoir_shuffle(it: Iterable, buffer_size: int, seed: int = 0) -> Iterator:
    """Streaming reservoir shuffle for IterableDatasets (sample-level)."""
    if buffer_size <= 1:
        yield from it
        return
    rng = random.Random(seed + int(os.environ.get("RANK", "0")))
    buf: list = []
    for x in it:
        buf.append(x)
        if len(buf) >= buffer_size:
            j = rng.randrange(len(buf))
            buf[j], buf[-1] = buf[-1], buf[j]
            yield buf.pop()
    rng.shuffle(buf)
    yield from buf


class ScatterPtDataset(IterableDataset):
    """One-object-per-sample dataset over a local dir / mount / ``s3://`` prefix.

    ``transform`` receives each object's raw bytes and returns the sample
    dict; the default is a plain ``torch.load``. Consumers with a custom
    payload schema (latents + prompts + action metadata, ...) inject their
    own decode here.
    """

    def __init__(
        self,
        uri: str,
        *,
        transform: Callable[[bytes], Any] | None = None,
        region: Optional[str] = None,
        shuffle_buffer: int = 1000,
        length: Optional[int] = None,
        use_mount: bool | None = None,
        read_policy: str | None = None,
    ):
        super().__init__()
        self.uri = uri
        self.transform = transform if transform is not None else _default_transform
        self.region = region or aws_region()
        self.shuffle_buffer = shuffle_buffer
        self._length = length

        if use_mount is True:
            policy = "mount"
        elif use_mount is False:
            policy = "direct_s3"
        else:
            policy = _read_policy(
                read_policy,
                env_name="ARCSTORE_DATA_READ_POLICY",
                default="direct_s3",
            )

        loc = resolve(uri)
        rp = loc.read_path() if loc.is_s3 else loc.local_path
        mount_usable = loc.is_s3 and rp is not None and os.path.isdir(rp)
        if use_mount is True and loc.is_s3 and not mount_usable:
            logger.warning(
                f"[arcstore-scatter] use_mount=True but no usable mount for "
                f"{uri}; falling back to direct S3."
            )
        self._local_dir: str | None
        if not loc.is_s3:
            self._local_dir = rp
        elif mount_usable and policy in ("auto", "mount"):
            self._local_dir = rp
            logger.info(f"[arcstore-scatter] reading {uri} via mount {rp}")
        else:
            self._local_dir = None

    # -- direct S3 path ------------------------------------------------------
    def _iter_s3(self) -> Iterator[Any]:
        from s3torchconnector import S3IterableDataset

        ds = S3IterableDataset.from_prefix(
            self.uri,
            region=self.region,
            transform=lambda r: self.transform(r.read()),
            enable_sharding=True,
        )
        yield from reservoir_shuffle(iter(ds), self.shuffle_buffer)

    # -- local / mounted path --------------------------------------------------
    def _local_keys(self) -> list[str]:
        import glob as _glob

        files = sorted(_glob.glob(os.path.join(self._local_dir, "*.pt")))
        # Shard across DDP ranks, then DataLoader workers.
        rank = int(os.environ.get("RANK", "0"))
        world = int(os.environ.get("WORLD_SIZE", "1"))
        files = files[rank::world]
        info = torch.utils.data.get_worker_info()
        if info is not None:
            files = files[info.id :: info.num_workers]
        return files

    def _iter_local(self) -> Iterator[Any]:
        def _gen():
            for path in self._local_keys():
                with open(path, "rb") as f:
                    yield self.transform(f.read())

        yield from reservoir_shuffle(_gen(), self.shuffle_buffer)

    def __iter__(self) -> Iterator[Any]:
        return self._iter_local() if self._local_dir is not None else self._iter_s3()

    def __len__(self) -> int:
        if self._length is not None:
            return self._length
        raise TypeError(
            "ScatterPtDataset is an iterable dataset with no fixed length; "
            "pass length= to define an artificial epoch size."
        )
