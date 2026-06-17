"""arcstore — unified storage IO for ML training codebases.

One path-driven API over three access modes:

* direct S3 (``s3://bucket/key``, via s5cmd / boto3 / s3torchconnector)
* local filesystems (``/local-ssd``, ``/efs``, ``/tmp``)
* FUSE-mounted S3 buckets (``ARCSTORE_S3_MOUNTS="bucket=/mountdir,..."``)

Generic reads may use a configured mount; training hot-path helpers default
to direct S3 and accept explicit ``read_policy="mount"`` for compatibility.
Writes ALWAYS go through the S3 API (local write + push) because
mountpoint-s3 rejects overwrites.

Torch-dependent helpers (checkpoint loading, safetensors streaming, DCP
full-state, datasets) live in :mod:`arcstore.torch` behind the
``arcstore[torch]`` extra.
"""
import os as _os

# This repo intentionally keeps arcstore's source files flattened directly
# under src/. Expose this module as a package so imports like
# ``arcstore.location`` and ``arcstore.torch`` keep their public shape.
__package__ = __name__
__path__ = [_os.path.dirname(__file__)]
if __spec__ is not None:
    __spec__.submodule_search_locations = __path__

from ._env import aws_region, cache_dir
from .checkpoint import (
    available_checkpoint_kinds,
    load_checkpoint,
    register_checkpoint_backend,
    save_checkpoint,
)
from .contents import ContentsManager, local_mirror_path
from .data import (
    DatasetAccess,
    available_backends,
    build_dataloader,
    open_dataset,
    register_backend,
    resolve_dataset_access,
)
from .formats import detect_format
from .io import (
    download_dir,
    download_file,
    exists,
    glob_files,
    list_prefix,
    open_read,
    open_write,
    put,
    put_async,
    read_bytes,
    upload_dir,
    upload_dir_async,
    upload_file,
    upload_file_async,
    wait_for_uploads,
    write_bytes,
)
from .location import Location, is_s3, refresh_mounts, resolve, split_s3
from .logtee import LogTee
from .run import RunStorage, sync_artifacts
from .staging import ensure_local_file, stage_to_local
from .workspace import split_workdir

__version__ = "0.1.0"

__all__ = [
    "ContentsManager",
    "DatasetAccess",
    "Location",
    "LogTee",
    "RunStorage",
    "available_backends",
    "available_checkpoint_kinds",
    "aws_region",
    "build_dataloader",
    "cache_dir",
    "detect_format",
    "download_dir",
    "download_file",
    "ensure_local_file",
    "exists",
    "glob_files",
    "is_s3",
    "list_prefix",
    "load_checkpoint",
    "local_mirror_path",
    "open_dataset",
    "open_read",
    "open_write",
    "put",
    "put_async",
    "read_bytes",
    "refresh_mounts",
    "register_backend",
    "register_checkpoint_backend",
    "resolve",
    "resolve_dataset_access",
    "save_checkpoint",
    "split_s3",
    "split_workdir",
    "stage_to_local",
    "sync_artifacts",
    "upload_dir",
    "upload_dir_async",
    "upload_file",
    "upload_file_async",
    "wait_for_uploads",
    "write_bytes",
]
