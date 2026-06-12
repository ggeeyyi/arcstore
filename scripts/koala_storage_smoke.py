#!/usr/bin/env python3
"""Koala/AWS storage smoke for arcstore.

This script creates a tiny dataset under local SSD, uploads it to S3, then
validates direct-S3, optional mount, and local paths through public arcstore
interfaces.
"""
from __future__ import annotations

import io
import json
import os
import socket
import tarfile
import time
from pathlib import Path

import arcstore
from arcstore.torch import (
    ScatterPtDataset,
    dcp_dir_exists,
    expand_urls,
    load_accelerate_state,
    save_accelerate_state,
    tar_url,
)


def _base_local() -> Path:
    root = Path("/local-ssd/arcstore-smoke")
    if not root.parent.exists():
        root = Path("/tmp/arcstore-smoke")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _s3_root() -> str:
    env = os.environ.get("ARCSTORE_SMOKE_S3")
    if env:
        return env.rstrip("/")
    user = os.environ.get("KOALA_USER") or os.environ.get("USER") or "unknown"
    mode = os.environ.get("ARCSTORE_SMOKE_MODE", "unknown")
    return f"s3://arcwm-code-us-west-2/{user}/arcstore-smoke/{socket.gethostname()}-{mode}"


def _write_inputs(root: Path) -> Path:
    import torch

    src = root / "src"
    if src.exists():
        import shutil

        shutil.rmtree(src)
    (src / "scatter").mkdir(parents=True)
    (src / "shards").mkdir(parents=True)
    (src / "dcp").mkdir(parents=True)
    (src / "small").mkdir(parents=True)

    (src / "small" / "hello.txt").write_text("hello arcstore\n")
    for idx in range(3):
        torch.save(
            {"idx": idx, "x": torch.tensor([idx])},
            src / "scatter" / f"sample_{idx:03d}.pt",
        )

    with tarfile.open(src / "shards" / "shard-000.tar", "w") as tf:
        data = b"sample zero\n"
        info = tarfile.TarInfo("000.txt")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))

    (src / "dcp" / ".metadata").write_bytes(b"metadata")
    (src / "dcp" / "shard.bin").write_bytes(b"shard")
    return src


def _retry_exists(path: str, timeout_s: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if os.path.exists(path):
            return True
        time.sleep(1.0)
    return os.path.exists(path)


class FakeAccelerator:
    is_main_process = True
    is_local_main_process = True

    def __init__(self):
        self.loaded = None

    def save_state(self, path):
        import torch

        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save({"state": 1}, str(Path(path) / "state.pt"))

    def load_state(self, path):
        self.loaded = path

    def wait_for_everyone(self):
        pass


def _decode(raw: bytes):
    import torch

    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)


def _first_dataset_sample(uri: str, **kwargs) -> int:
    ds = ScatterPtDataset(uri, transform=_decode, shuffle_buffer=1, **kwargs)
    sample = next(iter(ds))
    return int(sample["idx"])


def main() -> int:
    mode = os.environ.get("ARCSTORE_SMOKE_MODE", "nomount")
    local = _base_local() / mode
    local.mkdir(parents=True, exist_ok=True)
    src = _write_inputs(local)
    s3 = _s3_root()

    arcstore.upload_dir(str(src), s3)

    results: dict[str, object] = {
        "mode": mode,
        "s3": s3,
        "hostname": socket.gethostname(),
    }

    small_s3 = f"{s3}/small/hello.txt"
    scatter_s3 = f"{s3}/scatter"
    shards_s3 = f"{s3}/shards"
    dcp_s3 = f"{s3}/dcp"

    results["direct_read_bytes"] = arcstore.read_bytes(small_s3).decode().strip()
    results["wds_default_url"] = tar_url(shards_s3, "shard-000.tar")
    assert str(results["wds_default_url"]).startswith("pipe:s5cmd cat s3://")
    results["wds_expand"] = expand_urls(f"{shards_s3}/shard-*.tar")
    assert len(results["wds_expand"]) == 1
    assert str(results["wds_expand"][0]).startswith("pipe:")

    results["scatter_direct_idx"] = _first_dataset_sample(
        scatter_s3,
        read_policy="direct_s3",
    )

    dcp_stage = local / "dcp-stage"
    arcstore.download_dir(dcp_s3, str(dcp_stage), required_files=(".metadata",))
    assert (dcp_stage / ".metadata").is_file()
    assert dcp_dir_exists(dcp_s3)
    try:
        arcstore.download_dir(
            scatter_s3,
            str(local / "bad-dcp"),
            required_files=(".metadata",),
            retries=1,
        )
    except FileNotFoundError:
        results["missing_marker_failed"] = True
    else:
        raise AssertionError("download_dir should fail when required marker is absent")

    acc = FakeAccelerator()
    acc_local = local / "checkpoint-7"
    acc_s3 = f"{s3}/accelerate/checkpoint-7"
    save_accelerate_state(acc, str(acc_local), acc_s3)
    acc2 = FakeAccelerator()
    step = load_accelerate_state(acc2, acc_s3, local_dir=str(local / "acc-stage"))
    assert step == 7 and acc2.loaded == str(local / "acc-stage")
    results["accelerate_step"] = step

    results["local_wds_url"] = tar_url(str(src / "shards"), "shard-000.tar")
    assert results["local_wds_url"] == str(src / "shards" / "shard-000.tar")
    results["scatter_local_idx"] = _first_dataset_sample(str(src / "scatter"))

    if mode == "mount":
        mount = os.environ.get("ARCSTORE_MOUNT_ROOT", "/threed-code")
        os.environ["ARCSTORE_S3_MOUNTS"] = f"arcwm-code-us-west-2={mount}"
        arcstore.refresh_mounts()
        mounted_path = arcstore.resolve(small_s3).read_path()
        assert mounted_path and mounted_path.startswith(mount)
        assert _retry_exists(mounted_path), mounted_path
        results["mounted_path"] = mounted_path
        mount_url = tar_url(shards_s3, "shard-000.tar", read_policy="mount")
        assert mount_url.startswith(mount), mount_url
        results["wds_mount_url"] = mount_url
        results["scatter_mount_idx"] = _first_dataset_sample(
            scatter_s3,
            read_policy="mount",
        )

    result_path = local / "result.json"
    result_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    arcstore.upload_file(str(result_path), f"{s3}/result-{mode}.json")
    print("ARCSTORE_SMOKE_RESULT " + json.dumps(results, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    # s3torchconnector may leave non-daemon native worker threads alive after a
    # short one-shot smoke. The result has already been uploaded and stdout is
    # flushed, so exit the process explicitly to let Koala mark the job done.
    os._exit(main())
