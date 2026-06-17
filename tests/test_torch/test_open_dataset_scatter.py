import io

import pytest

torch = pytest.importorskip("torch")

import arcstore  # noqa: E402


@pytest.fixture
def pt_dir(tmp_path):
    for i in range(8):
        torch.save({"x": torch.full((2,), float(i)), "name": f"s{i}"}, tmp_path / f"{i:03d}.pt")
    return tmp_path


def test_raw_dict_samples_without_decode(pt_dir):
    ds = arcstore.open_dataset(str(pt_dir), shuffle_buffer=1)
    samples = list(ds)
    assert len(samples) == 8
    assert all(set(s) == {"pt"} for s in samples)
    assert all(isinstance(s["pt"], (bytes, bytearray)) for s in samples)


def test_decode_applied(pt_dir):
    def decode(sample):
        return torch.load(io.BytesIO(sample["pt"]), weights_only=False)

    ds = arcstore.open_dataset(str(pt_dir), decode=decode, shuffle_buffer=1)
    out = list(ds)
    assert len(out) == 8
    assert {s["name"] for s in out} == {f"s{i}" for i in range(8)}


def test_legacy_transform_bridge(pt_dir):
    # Old single-arg transform(raw_bytes) is bridged to decode.
    ds = arcstore.open_dataset(
        str(pt_dir), transform=lambda raw: len(raw), shuffle_buffer=1
    )
    out = list(ds)
    assert len(out) == 8
    assert all(isinstance(n, int) and n > 0 for n in out)


def test_format_override(pt_dir):
    ds = arcstore.open_dataset(str(pt_dir), format="scatter", shuffle_buffer=1)
    assert len(list(ds)) == 8


def test_length_for_len(pt_dir):
    ds = arcstore.open_dataset(str(pt_dir), length=8, shuffle_buffer=1)
    assert len(ds) == 8
    ds_nolen = arcstore.open_dataset(str(pt_dir), shuffle_buffer=1)
    with pytest.raises(TypeError):
        len(ds_nolen)
