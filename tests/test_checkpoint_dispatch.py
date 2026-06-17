import pytest

from arcstore import (
    available_checkpoint_kinds,
    load_checkpoint,
    register_checkpoint_backend,
    save_checkpoint,
)


def test_unknown_kind_save_raises_value_error():
    with pytest.raises(ValueError) as ei:
        save_checkpoint("/data/ckpt", "bogus", obj={})
    assert "no save backend" in str(ei.value)
    assert "bogus" in str(ei.value)


def test_unknown_kind_load_raises_value_error():
    with pytest.raises(ValueError) as ei:
        load_checkpoint("/data/ckpt", "bogus")
    assert "no load backend" in str(ei.value)
    assert "bogus" in str(ei.value)


def test_available_checkpoint_kinds_callable():
    kinds = available_checkpoint_kinds()
    assert isinstance(kinds, list)


def test_register_checkpoint_backend_dispatches_custom_kind():
    import arcstore.checkpoint.registry as reg

    save_sentinel = object()
    load_sentinel = object()
    register_checkpoint_backend(
        "custom_ckpt",
        save=lambda path, **kw: save_sentinel,
        load=lambda path, **kw: load_sentinel,
    )
    try:
        assert save_checkpoint("/data/x", "custom_ckpt", a=1) is save_sentinel
        assert load_checkpoint("/data/x", "custom_ckpt") is load_sentinel
        assert "custom_ckpt" in available_checkpoint_kinds()
    finally:
        reg._SAVE.pop("custom_ckpt", None)
        reg._LOAD.pop("custom_ckpt", None)
