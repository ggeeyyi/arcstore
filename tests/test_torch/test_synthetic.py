"""SyntheticDataset + open_dataset(format="synthetic") + build_dataloader."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import arcstore  # noqa: E402
from arcstore.torch import SyntheticDataset  # noqa: E402


def test_synthetic_dataset_deterministic():
    ds = SyntheticDataset(num_samples=4, sample_shape=(2, 3), feature_dim=8, feature_len=5)
    assert len(ds) == 4
    a = ds[0]
    b = ds[0]
    assert torch.equal(a["sample"], b["sample"])  # per-index seed -> deterministic
    assert a["sample"].shape == (2, 3)
    assert a["feature"].shape == (5, 8)


def test_open_dataset_synthetic_length_and_decode():
    seen = {}

    def decode(s):
        seen["called"] = True
        return s["sample"].sum()

    ds = arcstore.open_dataset(
        "", format="synthetic", length=3, sample_shape=(2,), feature_dim=2, feature_len=2,
        decode=decode,
    )
    assert len(ds) == 3
    _ = ds[0]
    assert seen.get("called")


def test_build_dataloader_synthetic():
    dl = arcstore.build_dataloader(
        "", format="synthetic", length=8, num_workers=0, batch_size=2,
        sample_shape=(2,), feature_dim=2, feature_len=2,
    )
    batch = next(iter(dl))
    assert batch["sample"].shape[0] == 2
