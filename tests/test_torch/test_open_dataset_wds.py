import io
import tarfile

import pytest

pytest.importorskip("torch")
pytest.importorskip("webdataset")

import arcstore  # noqa: E402


@pytest.fixture
def tar_shard(tmp_path):
    path = tmp_path / "shard-000.tar"
    with tarfile.open(path, "w") as tf:
        for i in range(4):
            data = f"sample-{i}".encode()
            info = tarfile.TarInfo(name=f"{i:03d}.txt")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


def test_wds_auto_detect_and_decode(tar_shard):
    ds = arcstore.open_dataset(
        str(tar_shard),
        decode=lambda s: s["txt"].decode(),
        shuffle_buffer=0,
        shuffle_shards=False,
    )
    out = sorted(list(ds))
    assert out == [f"sample-{i}" for i in range(4)]


def test_wds_raw_samples_have_key(tar_shard):
    ds = arcstore.open_dataset(str(tar_shard), shuffle_buffer=0, shuffle_shards=False)
    samples = list(ds)
    assert len(samples) == 4
    assert all("__key__" in s and "txt" in s for s in samples)


def test_wds_format_detected_as_wds(tar_shard):
    assert arcstore.detect_format(str(tar_shard)) == "wds"
