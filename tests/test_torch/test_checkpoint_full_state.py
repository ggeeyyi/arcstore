import pytest

pytest.importorskip("torch")

from arcstore import load_checkpoint, save_checkpoint  # noqa: E402


def test_save_forwards_to_save_full_state(monkeypatch):
    import arcstore.torch.ckpt_backends.full_state_backend as be

    captured = {}

    def fake_save(dest, models, optimizers, **kw):
        captured["args"] = (dest, models, optimizers)
        captured["kw"] = kw

    monkeypatch.setattr(be, "save_full_state", fake_save)

    models = object()
    optimizers = object()
    save_checkpoint(
        "s3://bkt/run/ckpt",
        "full_state",
        models=models,
        optimizers=optimizers,
        step=42,
    )
    assert captured["args"] == ("s3://bkt/run/ckpt", models, optimizers)
    assert captured["kw"]["step"] == 42


def test_load_forwards_to_load_full_state(monkeypatch):
    import arcstore.torch.ckpt_backends.full_state_backend as be

    def fake_load(src, models, optimizers, **kw):
        return 99

    monkeypatch.setattr(be, "load_full_state", fake_load)

    step = load_checkpoint(
        "s3://bkt/run/ckpt",
        "full_state",
        models=object(),
        optimizers=object(),
    )
    assert step == 99
