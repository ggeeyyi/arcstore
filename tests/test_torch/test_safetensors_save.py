"""save_safetensors_weights + load_pretrained round-trip (single process, local)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors.torch")
import torch.nn as nn  # noqa: E402

from arcstore.torch import load_pretrained, save_safetensors_weights  # noqa: E402


def test_save_then_load_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    # pass an explicit state_dict to avoid the collective DCP gather (no dist here)
    out = save_safetensors_weights(model, str(tmp_path), state_dict=model.state_dict())
    assert out is not None and out.endswith("model.safetensors")

    target = nn.Linear(4, 3)
    # weights differ before load
    assert not torch.equal(target.weight, model.weight)
    stats = load_pretrained(target, out, strict=True)
    assert stats["missing_keys"] == 0 and stats["unexpected_keys"] == 0
    assert torch.equal(target.weight, model.weight)
    assert torch.equal(target.bias, model.bias)


def test_save_to_dir_uses_weights_name(tmp_path):
    model = nn.Linear(2, 2)
    out = save_safetensors_weights(model, str(tmp_path), state_dict=model.state_dict())
    assert (tmp_path / "model.safetensors").is_file()
    assert out == str(tmp_path / "model.safetensors")
