from random import Random
from typing import Callable, Optional, Sequence, override
from warnings import deprecated

from spargel_llm.data import DataSource

from .data import Dataset
from .tokenizer import Tokenizer


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
        self,
        text: str,
        min_len: int,
        max_len: int | None = None,
        *,
        random: Random = Random(),
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
        if self._min_len != self._max_len:
            length = self._random.randint(self._min_len, self._max_len)
        else:
            length = self._min_len

        start = self._random.randint(0, len(self._text) - length)
        return self._text[start : start + length]


class FixedLengthDataset[T](Dataset[list[T]]):
    seq: list[T]
    length: int
    stride: int
    offset: int

    _count: int

    def __init__(self, seq: Sequence[T], length: int, stride: int = 1, offset: int = 0):
        assert length > 0 and length <= len(seq)

        self.seq = list(seq)
        self.length = length
        self.stride = stride
        self.offset = offset

        self._count = (len(seq) - offset - length) // stride + 1

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index: int) -> list[T]:
        if index < 0 or index >= self._count:
            raise IndexError

        start = self.offset + index * self.stride
        return self.seq[start : start + self.length]


@deprecated("should use TokenizedTextSource instead")
class TokenizedDataset(Dataset[list[int]]):
    text_dataset: Dataset[str]
    tokenizer: Tokenizer
    processor: Optional[Callable[[list[int]], list[int]]]

    def __init__(
        self,
        text_dataset: Dataset[str],
        tokenizer: Tokenizer,
        *,
        processor: Optional[Callable[[list[int]], list[int]]] = None,
    ):
        self.text_dataset = text_dataset
        self.tokenizer = tokenizer

        self.processor = processor

    def __len__(self) -> int:
        return len(self.text_dataset)

    def __getitem__(self, index: int) -> list[int]:
        tokens = self.tokenizer.encode(self.text_dataset[index])

        if self.processor is not None:
            tokens = self.processor(tokens)

        return tokens
