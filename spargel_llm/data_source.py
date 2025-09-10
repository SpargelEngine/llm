from abc import ABC, abstractmethod
from collections.abc import Iterable
from random import Random
from typing import Callable, override


class DataSource[T](ABC):
    """Base class for data sources."""

    @abstractmethod
    def sample(self) -> T:
        pass


class PlainTextSource(DataSource[str]):
    """Data source that samples from a given text.

    Sampled text will have a length equally distributed between [min_len, max_len],
    and a position equally distributed in the possible positions.
    """

    _text: str
    _min_len: int
    _max_len: int
    _random: Random

    def __init__(
        self, text: str, min_len: int, max_len: int | None, *, random: Random = Random()
    ):
        """
        Args:
            text (str): the text to sample from
            min_len (int): minimum length for sampled text
            max_len (int | None): maximum length for sampled text; equal to min_len if not provided
        """

        if max_len is None:
            max_len = min_len

        assert min_len >= 0 and max_len >= min_len and max_len <= len(text)

        self._text = text
        self._random = random
        self._min_len, self._max_len = min_len, max_len

    @override
    def sample(self) -> str:
        if self._min_len == self._max_len:
            length = self._random.randint(self._min_len, self._max_len)
        else:
            length = self._min_len

        start = self._random.randint(0, len(self._text) - length)
        return self._text[start : start + length]


class GeneratedDataSource[T](DataSource[T]):
    """Data source that samples by calling a generator function.

    The generator function can use a Random instance to probuce random results.
    """

    _func: Callable[[Random], T]
    _random: Random

    def __init__(self, func: Callable[[Random], T], *, random: Random = Random()):
        self._func = func
        self._random = random

    @override
    def sample(self) -> T:
        return self._func(self._random)


class WeightedDataSource[T](DataSource[T]):
    """Data source that samples from multiple sources randomly.

    Each time, one of the sources is randomly chosen according to the provided weights.
    """

    _weights: list[float]
    _sources: list[DataSource[T]]
    _random: Random

    def __init__(
        self,
        sources: Iterable[tuple[float, DataSource[T]]],
        *,
        random: Random = Random(),
    ):
        self._weights = []
        self._sources = []

        sum_of_weights = 0.0
        for weight, source in sources:
            assert weight >= 0.0
            sum_of_weights += weight

            self._weights.append(weight)
            self._sources.append(source)

        assert sum_of_weights > 0.0

        self._random = random

    @override
    def sample(self) -> T:
        source = self._random.choices(self._sources, weights=self._weights)[0]
        return source.sample()
