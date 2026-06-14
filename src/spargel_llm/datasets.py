from typing import Callable, Optional, Sequence
from warnings import deprecated

from .data import Dataset
from .meta import ai_marker
from .tokenizer import Tokenizer

type PadFunc[T] = Optional[Callable[[list[T], int], None]]


@ai_marker(human_checked=True)
class FixedLengthDataset[T](Dataset[list[T]]):
    seq: list[T]
    length: int
    stride: int
    offset: int
    allow_incomplete: bool
    pad_func: PadFunc[T]

    _count: int

    def __init__(
        self,
        seq: Sequence[T],
        length: int,
        stride: int = 1,
        offset: int = 0,
        allow_incomplete: bool = False,
        pad_func: PadFunc[T] = None,
    ):
        assert length > 0
        if not allow_incomplete:
            assert length <= len(seq)

        self.seq = list(seq)
        self.length = length
        self.stride = stride
        self.offset = offset
        self.allow_incomplete = allow_incomplete
        self.pad_func = pad_func

        if allow_incomplete:
            self._count = max(0, (len(seq) - offset + stride - 1) // stride)
        else:
            self._count = max(0, (len(seq) - offset - length) // stride + 1)

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index: int) -> list[T]:
        if index < 0 or index >= self._count:
            raise IndexError

        start = self.offset + index * self.stride
        end = start + self.length

        subseq = self.seq[start:end]

        if (
            self.allow_incomplete
            and self.pad_func is not None
            and len(subseq) < self.length
        ):
            self.pad_func(subseq, self.length)

        return subseq


def pad_simple(
    seq: list[int], length: int, pad_index: int, end_index: Optional[int] = None
):
    if len(seq) < length:
        if end_index is not None:
            seq.append(end_index)
        seq.extend(pad_index for _ in range(length - len(seq)))


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
