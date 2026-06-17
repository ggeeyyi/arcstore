"""reservoir_shuffle is pure-python (torch-free), so it runs without torch."""
from arcstore.data.shuffle import reservoir_shuffle


def test_passthrough_when_buffer_le_one():
    assert list(reservoir_shuffle(iter(range(5)), buffer_size=1)) == list(range(5))


def test_reservoir_shuffle_preserves_multiset():
    out = list(reservoir_shuffle(iter(range(100)), buffer_size=10, seed=1))
    assert sorted(out) == list(range(100))


def test_reservoir_shuffle_actually_reorders():
    out = list(reservoir_shuffle(iter(range(100)), buffer_size=10, seed=1))
    assert out != list(range(100))  # a real shuffle, not a passthrough


def test_reservoir_shuffle_deterministic_for_seed():
    a = list(reservoir_shuffle(iter(range(100)), buffer_size=10, seed=1))
    b = list(reservoir_shuffle(iter(range(100)), buffer_size=10, seed=1))
    assert a == b
