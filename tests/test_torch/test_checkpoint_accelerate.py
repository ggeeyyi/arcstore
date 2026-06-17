import os

import pytest

torch = pytest.importorskip("torch")

from arcstore import load_checkpoint, save_checkpoint  # noqa: E402


class FakeAccelerator:
    is_main_process = True
    is_local_main_process = True

    def __init__(self):
        self.loaded = None
        self.waits = 0

    def save_state(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save({"state": 1}, os.path.join(path, "state.pt"))

    def load_state(self, path):
        self.loaded = path

    def wait_for_everyone(self):
        self.waits += 1


def test_save_to_s3_requires_local_dir(fake_s5cmd):
    acc = FakeAccelerator()
    with pytest.raises(ValueError) as ei:
        save_checkpoint("s3://bkt/run/checkpoint-5", "accelerate", accelerator=acc)
    assert "local_dir" in str(ei.value)


def test_accelerate_save_and_load_roundtrip(fake_s5cmd, tmp_path):
    acc = FakeAccelerator()
    save_checkpoint(
        "s3://bkt/run/checkpoint-12",
        "accelerate",
        accelerator=acc,
        local_dir=str(tmp_path / "checkpoint-12"),
    )
    assert (fake_s5cmd / "bkt" / "run" / "checkpoint-12" / "state.pt").is_file()

    step = load_checkpoint(
        "s3://bkt/run/checkpoint-12",
        "accelerate",
        accelerator=acc,
        local_dir=str(tmp_path / "stage"),
    )
    assert step == 12
    assert acc.loaded == str(tmp_path / "stage")


def test_deepspeed_kind_removed(tmp_path):
    # DeepSpeed is now served only by CheckpointManager; the registry no longer
    # exposes a "deepspeed" kind.
    acc = FakeAccelerator()
    with pytest.raises(ValueError) as ei:
        save_checkpoint(str(tmp_path / "checkpoint-3"), "deepspeed", accelerator=acc)
    assert "deepspeed" in str(ei.value)
