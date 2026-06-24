"""Shared Parquet utilities used by CLI scripts and library code."""

import random
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq


def resolve_index(idx: int, total: int) -> int:
    """Resolve a possibly-negative index into a non-negative one."""
    if idx < 0:
        idx += total
    return idx


def iter_row_groups(pf: pq.ParquetFile):
    """Yield ``(rg_idx, offset, rg_rows)`` for each row group in *pf*."""
    offset = 0
    for rg_idx in range(pf.metadata.num_row_groups):
        rg_rows = pf.metadata.row_group(rg_idx).num_rows
        yield rg_idx, offset, rg_rows
        offset += rg_rows


def resolve_row(pf: pq.ParquetFile, row: int | None = None) -> int:
    """Resolve a row index for *pf*: negative → from end, None → random.

    Validates that the result is within ``[0, pf.metadata.num_rows)``.
    """
    if row is None:
        row = random.randrange(pf.metadata.num_rows)
    row = resolve_index(row, pf.metadata.num_rows)
    if row < 0 or row >= pf.metadata.num_rows:
        raise IndexError(
            f"row index out of range (file has {pf.metadata.num_rows} rows)"
        )
    return row


def read_row(pf: pq.ParquetFile, row: int, columns: list[str]) -> tuple[int, pa.Table]:
    """Read a single *row* from *pf*, only touching its row group.

    Returns ``(local_index, table)`` where *table* is the row group table
    and *local_index* is the row's position within it.
    """
    for rg_idx, offset, rg_rows in iter_row_groups(pf):
        if offset + rg_rows > row:
            table = pf.read_row_group(rg_idx, columns=columns)
            return row - offset, table
    raise IndexError(f"row {row} not found (file has {pf.metadata.num_rows} rows)")


def get_dataset_id(pf: pq.ParquetFile) -> str:
    """Extract the ``dataset_id`` from a Parquet file's schema metadata."""
    metadata = pf.schema_arrow.metadata
    if metadata is None:
        return ""
    return metadata.get(b"dataset_id", b"").decode()


def make_text_schema() -> pa.Schema:
    """Create the standard schema for a texts Parquet file."""
    schema = pa.schema([pa.field("text", pa.string())])
    return schema.with_metadata({"dataset_id": uuid4().hex})


def make_tokens_schema() -> pa.Schema:
    """Create the standard schema for a tokens Parquet file."""
    schema = pa.schema([pa.field("tokens", pa.list_(pa.uint32()))])
    return schema.with_metadata({"dataset_id": uuid4().hex})
