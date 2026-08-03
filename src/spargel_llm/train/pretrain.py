# Utilities for pre-training

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from pyarrow.parquet import ParquetFile

from .basic import Item


class SlidingWindowIterator(Iterator[Item]):
    """Iterate over a token sequence with a sliding window."""

    def __init__(
        self,
        seq: np.ndarray,
        seq_len: int,
        offset: int = 0,
        *,
        pad_id: int,
        stride: int = 0,
    ):
        self.seq = seq
        self.seq_len = seq_len
        self.offset = offset
        self.pad_id = pad_id
        self.stride = stride if stride > 0 else seq_len + 1
        self._full_mask = np.ones(seq_len, dtype=np.bool)

    def __iter__(self) -> SlidingWindowIterator:
        return self

    def __next__(self) -> Item:
        if self.offset > len(self.seq) - 2:
            raise StopIteration

        start = self.offset
        self.offset += self.stride

        L = min(len(self.seq) - 1 - start, self.seq_len)  # L >= 2

        if L == self.seq_len:
            return Item(
                seq_len=self.seq_len,
                input_ids=self.seq[start : start + self.seq_len],
                target_ids=self.seq[start + 1 : start + 1 + self.seq_len],
                mask=self._full_mask,
                tokens_valid=self.seq_len,
            )

        input_ids = np.full(self.seq_len, self.pad_id, dtype=np.int64)
        target_ids = np.full(self.seq_len, self.pad_id, dtype=np.int64)
        mask = np.zeros(self.seq_len, dtype=np.bool)

        input_ids[:L] = self.seq[start : start + L]
        target_ids[:L] = self.seq[start + 1 : start + 1 + L]
        mask[:L] = True

        return Item(
            seq_len=self.seq_len,
            input_ids=input_ids,
            target_ids=target_ids,
            mask=mask,
            tokens_valid=L,
        )


class ParquetConcatIterator(Iterator[Item]):
    """Iterate over a Parquet file in concat mode.

    Within each row group, rows are joined with a separator token,
    then a sliding window produces samples.
    """

    def __init__(
        self,
        pf: ParquetFile,
        seq_len: int,
        group_index: int = 0,
        offset: int = 0,
        *,
        pad_id: int,
        sep_id: int,
        column_name: str = "tokens",
        stride: int = 0,
        loop: bool = False,
    ):
        self.pf = pf
        self.seq_len = seq_len
        self.group_index = group_index
        self._offset = offset
        self.pad_id = pad_id
        self.sep_id = sep_id
        self.column_name = column_name
        self.stride = stride
        self.loop = loop

        self._num_row_groups = pf.metadata.num_row_groups
        self._inner: SlidingWindowIterator | None = None

    @property
    def offset(self) -> int:
        """Current window position within the active row group."""
        if self._inner is not None:
            return self._inner.offset
        return self._offset

    def __iter__(self) -> ParquetConcatIterator:
        return self

    def __next__(self) -> Item:
        while True:
            if self._inner is not None:
                try:
                    return next(self._inner)
                except StopIteration:
                    self._inner = None
                    self.group_index += 1
                    self._offset = 0

            if self.group_index >= self._num_row_groups:
                if self.loop:
                    self.group_index = 0
                else:
                    raise StopIteration

            table = self.pf.read_row_group(self.group_index, columns=[self.column_name])
            column = table.column(self.column_name).combine_chunks()

            offsets = column.offsets.to_numpy()
            values = column.values.to_numpy()

            n_rows = len(offsets) - 1
            total_len = values.size + n_rows - 1  # values + SEP between each pair

            seq_concat = np.empty(total_len, dtype=np.int64)

            pos = 0
            for i in range(n_rows):
                start = int(offsets[i])
                end = int(offsets[i + 1])
                L = end - start
                seq_concat[pos : pos + L] = values[start:end]
                pos += L
                if i < n_rows - 1:
                    seq_concat[pos] = self.sep_id
                    pos += 1

            self._inner = SlidingWindowIterator(
                seq_concat,
                self.seq_len,
                self.offset,
                pad_id=self.pad_id,
                stride=self.stride,
            )
