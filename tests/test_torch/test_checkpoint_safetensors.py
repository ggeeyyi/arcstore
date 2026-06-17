import pytest

pytest.importorskip("torch")

from arcstore import load_checkpoint, save_checkpoint  # noqa: E402


def test_save_forwards_to_save_safetensors_weights(monkeypatch):
    import arcstore.torch.ckpt_backends.safetensors_backend as be

    captured = {}

    def fake_save(model, out_dir, *, state_dict=None, s3_prefix=None, workers=None):
        captured["model"] = model
        captured["out_dir"] = out_dir
        captured["state_dict"] = state_dict
        captured["s3_prefix"] = s3_prefix
        captured["workers"] = workers
        return s3_prefix or out_dir

    monkeypatch.setattr(be, "save_safetensors_weights", fake_save)

    model = object()
    out = save_checkpoint(
        "s3://bkt/model.safetensors",
        "safetensors",
        model=model,
        state_dict={"w": 1},
    )
    # s3 dest: the backend stages to a local cache dir, then has
    # save_safetensors_weights upload under s3_prefix and returns that URI.
    assert out == "s3://bkt/model.safetensors"
    assert captured["model"] is model
    assert captured["state_dict"] == {"w": 1}
    assert captured["s3_prefix"] == "s3://bkt/model.safetensors"
    assert captured["out_dir"]  # a local stage dir was passed through


def test_load_forwards_to_load_safetensors_auto(monkeypatch):
    import arcstore.torch.ckpt_backends.safetensors_backend as be

    captured = {}

    def fake_load(uri_or_dir, **kw):
        captured["uri"] = uri_or_dir
        captured["kw"] = kw
        return {"w": 1}

    monkeypatch.setattr(be, "load_safetensors_auto", fake_load)

    out = load_checkpoint(
        "s3://bkt/model.safetensors", "safetensors", concurrency=4
    )
    assert out == {"w": 1}
    assert captured["uri"] == "s3://bkt/model.safetensors"
    assert captured["kw"]["concurrency"] == 4
