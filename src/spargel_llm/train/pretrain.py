# Utilities for pre-training

from collections.abc import Iterator

import numpy as np
from pyarrow.parquet import ParquetFile

from .basic import Item


def iterate_sliding_window(
    seq: np.ndarray,
    seq_len: int,
    offset: int = 0,
    *,
    pad_id: int,
    stride: int = 0,
) -> Iterator[Item]:
    if stride <= 0:
        stride = seq_len + 1

    while offset <= len(seq) - 2:
        L = min(len(seq) - 1 - offset, seq_len)  # L >= 2

        if L == seq_len:
            yield Item(
                seq_len=seq_len,
                input_ids=seq[offset : offset + seq_len],
                target_ids=seq[offset + 1 : offset + 1 + seq_len],
                mask=np.ones(seq_len, dtype=np.bool),
                tokens_valid=seq_len,
            )
        else:
            input_ids = np.full(seq_len, pad_id, dtype=np.int64)
            target_ids = np.full(seq_len, pad_id, dtype=np.int64)
            mask = np.zeros(seq_len, dtype=np.bool)

            input_ids[:L] = seq[offset : offset + L]
            target_ids[:L] = seq[offset + 1 : offset + 1 + L]
            mask[:L] = True

            yield Item(
                seq_len=seq_len,
                input_ids=input_ids,
                target_ids=target_ids,
                mask=mask,
                tokens_valid=L,
            )

        offset += stride


def iterate_parquet_concat(
    pf: ParquetFile,
    seq_len: int,
    start_rg_idx: int = 0,
    offset: int = 0,
    *,
    pad_id: int,
    sep_id: int,
    column_name: str = "tokens",
    stride: int = 0,
) -> Iterator[Item]:
    """
    Get data from a parquet file using concat mode.

    All sequences inside each row group are joined with a separator token,
    and data are sampled by sliding window over the joined sequence.
    """

    num_row_groups = pf.metadata.num_row_groups

    for rg_idx in range(start_rg_idx, num_row_groups):
        table = pf.read_row_group(rg_idx, columns=[column_name])
        column = table.column(column_name)

        total_len = len(column) - 1  # space for SEP token
        for seq in column:
            total_len += len(seq)

        seq_concat = np.empty(total_len, dtype=np.int64)

        pos = 0
        for seq in column:
            L = len(seq)
            seq_concat[pos : pos + L] = seq
            if pos + L < total_len:
                seq_concat[pos + L] = sep_id
            pos += L + 1

        yield from iterate_sliding_window(
            seq_concat, seq_len, offset, pad_id=pad_id, stride=stride
        )

        # Offset is only applied to the first starting row group.
        offset = 0
