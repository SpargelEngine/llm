from abc import ABC, abstractmethod
from random import Random
from typing import Callable, Iterable, Iterator, override


class DataSource[T](ABC):
    """Base class for data sources."""

    @abstractmethod
    def sample(self) -> T:
        pass

    def sample_multiple(self, count: int) -> Iterable[T]:
        assert count >= 0
        for _ in range(count):
            yield self.sample()


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

    @override
    def sample_multiple(self, count: int) -> Iterable[T]:
        assert count >= 0
        for source in self._random.choices(
            self._sources, weights=self._weights, k=count
        ):
            yield source.sample()


class Dataset[T](ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> T:
        pass


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


class DataLoader[T](Iterator[T]):
    dataset: Dataset[T]
    shuffle: bool
    random: Random

    _len: int
    _indices: list[int]
    _current: int

    def __init__(
        self,
        dataset: Dataset[T],
        shuffle: bool = False,
        *,
        random: Random = Random(),
    ):
        self.dataset = dataset
        self.shuffle = shuffle
        self.random = random

        self._len = len(dataset)
        self._indices = []

    @override
    def __iter__(self) -> Iterator[T]:
        if self.shuffle:
            self._indices = list(range(self._len))
            self.random.shuffle(self._indices)

        self._current = 0

        return self

    @override
    def __next__(self) -> T:
        if self._current >= self._len:
            raise StopIteration
        else:
            if self.shuffle:
                result = self.dataset[self._indices[self._current]]
            else:
                result = self.dataset[self._current]

            self._current += 1

            return result
