"""Inspect and display contents of data Parquet files (texts and tokens)."""

import sys
from argparse import ArgumentParser

import numpy as np
import pyarrow.parquet as pq

from spargel_llm.logging import log_info
from spargel_llm.utils import PromptAbortError

#### actions ####


def action_info(path: str, row_group: int = 0):
    """Show metadata about a tokens.parquet file."""
    pf = pq.ParquetFile(path)

    print(f"File:             {path}")
    print(f"Rows:             {pf.metadata.num_rows:,}")
    print(f"Row groups:       {pf.metadata.num_row_groups}")
    print(f"Schema:           {pf.schema_arrow}")

    metadata = pf.schema_arrow.metadata
    if metadata:
        dataset_id = metadata.get(b"dataset_id", b"").decode()
        if dataset_id:
            print(f"Dataset ID:       {dataset_id}")

    # Sample one row group for length statistics
    table = pf.read_row_group(row_group, columns=["tokens"])
    col = table.column("tokens").combine_chunks()
    offsets = col.offsets.to_numpy()
    lengths = offsets[1:] - offsets[:-1]

    print(f"Length statistics from row group {row_group}:")
    print(f"  Total tokens:     {lengths.sum():,}")
    print(f"  Average length:   {lengths.mean():.1f}")
    print(f"  Min length:       {lengths.min()}")
    print(f"  Max length:       {lengths.max()}")


def action_text_info(path: str, row_group: int = 0):
    """Show metadata about a texts.parquet file."""
    pf = pq.ParquetFile(path)

    print(f"File:             {path}")
    print(f"Rows:             {pf.metadata.num_rows:,}")
    print(f"Row groups:       {pf.metadata.num_row_groups}")
    print(f"Schema:           {pf.schema_arrow}")

    metadata = pf.schema_arrow.metadata
    if metadata:
        dataset_id = metadata.get(b"dataset_id", b"").decode()
        if dataset_id:
            print(f"Dataset ID:       {dataset_id}")

    # Sample one row group for length statistics
    table = pf.read_row_group(row_group, columns=["text"])
    col = table.column("text").combine_chunks()
    offsets = np.frombuffer(col.buffers()[1], dtype=np.int32)
    lengths = offsets[1:] - offsets[:-1]

    print(f"Length statistics from row group {row_group}:")
    print(f"  Total characters: {lengths.sum():,}")
    print(f"  Average length:   {lengths.mean():.1f}")
    print(f"  Min length:       {lengths.min()}")
    print(f"  Max length:       {lengths.max()}")


def action_text_show(path: str, row: int | None = None):
    """Print the text content of a specific row from a texts.parquet file.

    Only reads the row group containing the target row, skipping all others.
    If row is not specified, a random row is selected.
    """
    import random

    pf = pq.ParquetFile(path)

    if row is None:
        row = random.randrange(pf.metadata.num_rows)

    if row < 0:
        row += pf.metadata.num_rows

    if row < 0 or row >= pf.metadata.num_rows:
        raise IndexError(
            f"row index out of range (file has {pf.metadata.num_rows} rows)"
        )

    # Find which row group contains the target row
    offset = 0
    for rg_idx in range(pf.metadata.num_row_groups):
        rg = pf.metadata.row_group(rg_idx)
        if offset + rg.num_rows > row:
            # Read only this row group and extract the target row
            table = pf.read_row_group(rg_idx, columns=["text"])
            local_idx = row - offset
            text = table.column("text")[local_idx].as_py()
            print(f"[{row}/{pf.metadata.num_rows}] length={len(text)}")
            print(text)
            return
        offset += rg.num_rows


#### main ####


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Data inspection CLI for texts.parquet and tokens.parquet files.",
    )

    subparsers = parser.add_subparsers(dest="action", help="actions", required=True)

    # info
    info_parser = subparsers.add_parser("info", help="show tokens.parquet metadata")
    info_parser.add_argument("path", help="path to tokens.parquet file")
    info_parser.add_argument(
        "-rg",
        "--row-group",
        type=int,
        default=0,
        help="row group index to sample for length statistics (default: 0)",
    )

    # text_info
    text_info_parser = subparsers.add_parser(
        "text_info", help="show texts.parquet metadata"
    )
    text_info_parser.add_argument("path", help="path to texts.parquet file")
    text_info_parser.add_argument(
        "-rg",
        "--row-group",
        type=int,
        default=0,
        help="row group index to sample for length statistics (default: 0)",
    )

    # text_show
    text_show_parser = subparsers.add_parser(
        "text_show", help="print text at a specific row from a texts.parquet file"
    )
    text_show_parser.add_argument("path", help="path to texts.parquet file")
    text_show_parser.add_argument(
        "row",
        nargs="?",
        type=int,
        default=None,
        help="zero-based row index to print (negative indices count from the end; "
        "random if omitted)",
    )

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    match args.action:
        case "info":
            action_info(args.path, row_group=args.row_group)
        case "text_info":
            action_text_info(args.path, row_group=args.row_group)
        case "text_show":
            action_text_show(args.path, args.row)
        case _:
            raise ValueError(f"unrecognized action: {args.action}")


if __name__ == "__main__":
    try:
        main()
    except PromptAbortError:
        log_info("Aborting.")
        sys.exit(1)
