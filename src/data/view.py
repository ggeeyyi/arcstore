"""``_DecodedView``: wrap a stream of WebDataset-style ``dict`` samples with a
unified ``decode`` callback, optional sample-level reservoir shuffle, and an
optional artificial length.

Backends that yield raw ``dict`` samples (e.g. the scatter ``.pt`` backend)
wrap their iterator in this view so every format exposes the same
``IterableDataset`` surface. WebDataset already does its own
sharding/shuffle/decode inside its pipeline, so the wds backend does not use
this view.

Requires torch (imported lazily-safe: this module is only imported by the
torch backends).
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator, Optional

from torch.utils.data import IterableDataset

from .shuffle import reservoir_shuffle


class _DecodedView(IterableDataset):
    """Apply ``decode`` (+ optional shuffle) to a raw dict-sample iterable.

    ``source`` must be an iterable (or a zero-arg callable returning a fresh
    iterator on each epoch) of ``dict`` samples. ``decode`` maps each sample
    dict to the final training sample; ``decode=None`` yields the raw dicts.
    """

    def __init__(
        self,
        source: Iterable[dict] | Callable[[], Iterator[dict]],
        *,
        decode: Optional[Callable[[dict], Any]] = None,
        shuffle_buffer: int = 0,
        length: Optional[int] = None,
    ):
        super().__init__()
        self._source = source
        self._decode = decode
        self._shuffle_buffer = shuffle_buffer
        self._length = length

    def _raw_iter(self) -> Iterator[dict]:
        src = self._source
        return iter(src() if callable(src) else src)

    def __iter__(self) -> Iterator[Any]:
        it: Iterator[Any] = self._raw_iter()
        if self._shuffle_buffer and self._shuffle_buffer > 1:
            it = reservoir_shuffle(it, self._shuffle_buffer)
        if self._decode is None:
            yield from it
        else:
            for sample in it:
                yield self._decode(sample)

    def __len__(self) -> int:
        if self._length is not None:
            return self._length
        raise TypeError(
            "open_dataset returns an iterable dataset with no fixed length; "
            "pass length= to define an artificial epoch size."
        )


__all__ = ["_DecodedView"]
