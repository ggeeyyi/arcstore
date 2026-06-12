import os

import pytest

torch = pytest.importorskip("torch")

from arcstore.torch import load_accelerate_state, save_accelerate_state  # noqa: E402


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


class FakeEMA:
    def __init__(self):
        self.loaded = None

    def state_dict(self):
        return {"ema": torch.ones(1)}

    def load_state_dict(self, state):
        self.loaded = state


def test_save_accelerate_state_uploads(fake_s5cmd, tmp_path):
    acc = FakeAccelerator()
    local = tmp_path / "checkpoint-12"

    save_accelerate_state(acc, str(local), "s3://bkt/run/checkpoint-12")

    assert (fake_s5cmd / "bkt" / "run" / "checkpoint-12" / "state.pt").is_file()


def test_load_accelerate_state_downloads_and_loads_ema(fake_s5cmd, tmp_path):
    src = fake_s5cmd / "bkt" / "run" / "checkpoint-12"
    src.mkdir(parents=True)
    torch.save({"state": 1}, src / "state.pt")
    torch.save({"ema": torch.ones(1)}, src / "ema.pt")

    acc = FakeAccelerator()
    ema = FakeEMA()
    step = load_accelerate_state(
        acc,
        "s3://bkt/run/checkpoint-12",
        local_dir=str(tmp_path / "stage"),
        ema=ema,
    )

    assert step == 12
    assert acc.loaded == str(tmp_path / "stage")
    assert torch.equal(ema.loaded["ema"], torch.ones(1))
