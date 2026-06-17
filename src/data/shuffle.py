"""Sample-level streaming shuffle (pure-python, no torch).

Lives in the ``data`` layer so both the torch-free ``_DecodedView`` and the
torch ``ScatterPtDataset`` can share it without the data layer importing
upward into :mod:`arcstore.torch`.
"""
from __future__ import annotations

import os
import random
from typing import Iterable, Iterator

__all__ = ["reservoir_shuffle"]


def reservoir_shuffle(it: Iterable, buffer_size: int, seed: int = 0) -> Iterator:
    """Streaming reservoir shuffle for IterableDatasets (sample-level)."""
    if buffer_size <= 1:
        yield from it
        return
    rng = random.Random(seed + int(os.environ.get("RANK", "0")))
    buf: list = []
    for x in it:
        buf.append(x)
        if len(buf) >= buffer_size:
            j = rng.randrange(len(buf))
            buf[j], buf[-1] = buf[-1], buf[j]
            yield buf.pop()
    rng.shuffle(buf)
    yield from buf
