import pytest

from arcstore import available_backends, open_dataset, register_backend


def test_placeholder_formats_registered():
    backends = available_backends()
    for fmt in ("jsonl", "lmdb", "mds"):
        assert fmt in backends


def test_unsupported_jsonl_raises_with_guidance():
    with pytest.raises(NotImplementedError) as ei:
        open_dataset("s3://bkt/meta/manifest.jsonl")
    msg = str(ei.value)
    assert "jsonl" in msg
    assert "ensure_local_file" in msg


def test_unsupported_lmdb_points_to_scatter(monkeypatch, tmp_path):
    (tmp_path / "data.mdb").write_bytes(b"")
    with pytest.raises(NotImplementedError) as ei:
        open_dataset(str(tmp_path))
    assert "scatter" in str(ei.value)


def test_unknown_format_raises_value_error():
    with pytest.raises(ValueError) as ei:
        open_dataset("/data/whatever", format="bogus")
    assert "no backend" in str(ei.value)


def test_format_override_dispatches_by_given_format():
    # A .jsonl path would auto-detect as jsonl, but format= overrides it.
    # Override to lmdb -- a reserved placeholder no torch backend shadows -- so
    # this holds whether or not arcstore[torch] is installed (the real "mds"
    # backend registered by arcstore.torch.backends would otherwise shadow that
    # placeholder and not raise NotImplementedError).
    with pytest.raises(NotImplementedError) as ei:
        open_dataset("/data/x.jsonl", format="lmdb")
    msg = str(ei.value)
    assert "lmdb" in msg  # dispatched by the override...
    assert "ensure_local_file" not in msg  # ...not the auto-detected jsonl backend


def test_register_backend_dispatches_custom_format():
    import arcstore.data.registry as reg

    sentinel = object()
    register_backend("custom_fmt", lambda path, **kw: sentinel)
    try:
        assert open_dataset("/data/dir", format="custom_fmt") is sentinel
    finally:
        reg._BACKENDS.pop("custom_fmt", None)
