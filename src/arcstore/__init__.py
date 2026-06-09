"""arcstore — unified storage IO for ML training codebases.

One path-driven API over three access modes:

* direct S3 (``s3://bucket/key``, via s5cmd / boto3 / s3torchconnector)
* local filesystems (``/local-ssd``, ``/efs``, ``/tmp``)
* FUSE-mounted S3 buckets (``ARCSTORE_S3_MOUNTS="bucket=/mountdir,..."``)

Reads may use the mount when one is configured; writes ALWAYS go through
the S3 API (local write + push) because mountpoint-s3 rejects overwrites.

Torch-dependent helpers (checkpoint loading, safetensors streaming, DCP
full-state, datasets) live in :mod:`arcstore.torch` behind the
``arcstore[torch]`` extra.
"""
from ._env import aws_region
from .discovery import find_latest_ckpt
from .formats import detect_format
from .io import (
    download_dir,
    download_file,
    exists,
    glob_files,
    list_prefix,
    open_read,
    read_bytes,
    track_future,
    upload_dir,
    upload_dir_async,
    upload_file,
    upload_file_async,
    wait_for_uploads,
)
from .location import Location, is_s3, refresh_mounts, resolve, split_s3
from .logtee import LogTee
from .staging import ensure_local_file, stage_to_local
from .workspace import split_workdir

__version__ = "0.1.0"

__all__ = [
    "Location",
    "LogTee",
    "aws_region",
    "detect_format",
    "download_dir",
    "download_file",
    "ensure_local_file",
    "exists",
    "find_latest_ckpt",
    "glob_files",
    "is_s3",
    "list_prefix",
    "open_read",
    "read_bytes",
    "refresh_mounts",
    "resolve",
    "split_s3",
    "split_workdir",
    "stage_to_local",
    "track_future",
    "upload_dir",
    "upload_dir_async",
    "upload_file",
    "upload_file_async",
    "wait_for_uploads",
]
