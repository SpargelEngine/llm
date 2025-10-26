from abc import ABC, abstractmethod
from random import Random
from typing import Callable, Iterable, Iterator, Sequence, override

from .typing import Sliceable


class DataSource[T](ABC):
    """Base class for data sources."""

    @abstractmethod
    def sample(self) -> T: ...

    def sample_multiple(self, count: int) -> Iterable[T]:
        assert count >= 0
        for _ in range(count):
            yield self.sample()


class SeqDataSource[T](DataSource[T]):
    data: Sequence[T]
    random: Random

    def __init__(self, data: Sequence[T], *, random: Random = Random()):
        assert len(data) > 0

        self.data = data
        self.random = random

    @override
    def sample(self) -> T:
        index = self.random.randint(0, len(self.data) - 1)
        return self.data[index]


class GeneratedDataSource[T](DataSource[T]):
    """Data source that generates data by calling a generator function.

    The generator function can use a Random instance to probuce random results.
    """

    func: Callable[[Random], T]
    random: Random

    def __init__(self, func: Callable[[Random], T], *, random: Random = Random()):
        self.func = func
        self.random = random

    @override
    def sample(self) -> T:
        return self.func(self.random)


class WeightedDataSource[T](DataSource[T]):
    """Data source that samples from multiple sources randomly.

    Each time, one of the sources is randomly chosen according to the provided weights.
    """

    sources: Sequence[DataSource[T]]
    weights: Sequence[float]
    random: Random

    def __init__(
        self,
        sources: Sequence[DataSource[T]],
        weights: Sequence[float],
        *,
        random: Random = Random(),
    ):
        self.sources = sources
        self.weights = weights
        self.random = random

        assert all(weight >= 0 for weight in weights) and sum(weights) > 0.0

    @override
    def sample(self) -> T:
        source = self.random.choices(self.sources, weights=self.weights)[0]
        return source.sample()

    @override
    def sample_multiple(self, count: int) -> Iterable[T]:
        assert count >= 0
        for source in self.random.choices(self.sources, weights=self.weights, k=count):
            yield source.sample()


class SliceDataSource[S: Sliceable](DataSource[S]):
    """Data source that gives slices of length in the given range"""

    seq: S
    min_len: int
    max_len: int
    random: Random

    def __init__(
        self,
        seq: S,
        min_len: int,
        max_len: int = 0,
        *,
        random: Random = Random(),
    ):
        if max_len == 0:
            max_len = min_len

        assert 0 <= min_len <= max_len <= len(seq)

        self.seq = seq
        self.min_len, self.max_len = min_len, max_len
        self.random = random

    @override
    def sample(self) -> S:
        if self.min_len == self.max_len:
            length = self.min_len
        else:
            length = self.random.randint(self.min_len, self.max_len)

        index = self.random.randint(0, len(self.seq) - length)

        return self.seq[index : index + length]


# deprecated
class Dataset[T](ABC):
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, index: int) -> T: ...


class ListDataset[T](Dataset[T]):
    data: list[T]

    def __init__(self, data: list[T]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> T:
        if index < 0 or index >= len(self.data):
            raise IndexError

        return self.data[index]


# deprecated
class DataLoader[T](Iterator[T]):
    datasets: list[Dataset[T]]
    shuffle: bool
    random: Random

    _len: int
    _indices: list[tuple[int, int]]
    _current: int

    def __init__(
        self,
        datasets: Sequence[Dataset[T]],
        shuffle: bool = False,
        *,
        random: Random = Random(),
    ):
        self.datasets = list(datasets)
        self.shuffle = shuffle
        self.random = random

        self._len = sum(len(dataset) for dataset in datasets)
        self._indices = []
        for i, dataset in enumerate(datasets):
            for j in range(len(dataset)):
                self._indices.append((i, j))

    @override
    def __iter__(self) -> Iterator[T]:
        if self.shuffle:
            self.random.shuffle(self._indices)

        self._current = 0

        return self

    @override
    def __next__(self) -> T:
        if self._current >= self._len:
            raise StopIteration
        else:
            i, j = self._indices[self._current]
            result = self.datasets[i][j]

            self._current += 1

            return result
