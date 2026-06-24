"""Inspect and display contents of data Parquet files (texts and tokens)."""

from argparse import ArgumentParser

import numpy as np
import pyarrow.parquet as pq

from spargel_llm.parquet_utils import get_dataset_id, read_row, resolve_row

#### actions ####


def _open_and_print_parquet_metadata(path: str):
    """Open a Parquet file, print its metadata, and return it."""
    pf = pq.ParquetFile(path)
    print(f"File:             {path}")
    print(f"Rows:             {pf.metadata.num_rows:,}")
    print(f"Row groups:       {pf.metadata.num_row_groups}")
    print(f"Schema:           {pf.schema_arrow}")

    dataset_id = get_dataset_id(pf)
    if dataset_id:
        print(f"Dataset ID:       {dataset_id}")

    return pf


def _lengths_from_column(table, col_name: str) -> np.ndarray:
    """Extract per-row lengths from a column in a row group table."""
    col = table.column(col_name).combine_chunks()
    if col_name == "text":
        offsets = np.frombuffer(col.buffers()[1], dtype=np.int32)
    else:
        offsets = col.offsets.to_numpy()
    return offsets[1:] - offsets[:-1]


def action_info(path: str, col: str, row_group: int = 0):
    """Show metadata and length statistics for a data Parquet file."""
    pf = _open_and_print_parquet_metadata(path)
    table = pf.read_row_group(row_group, columns=[col])
    lengths = _lengths_from_column(table, col)

    label = "characters" if col == "text" else "tokens"
    print(f"Length statistics from row group {row_group}:")
    print(f"  Total {label}:     {lengths.sum():,}")
    print(f"  Average length:   {lengths.mean():.1f}")
    print(f"  Min length:       {lengths.min()}")
    print(f"  Max length:       {lengths.max()}")


def action_text_show(path: str, row: int | None = None):
    """Print the text content of a specific row from a texts.parquet file.

    Only reads the row group containing the target row, skipping all others.
    If row is not specified, a random row is selected.
    """
    pf = pq.ParquetFile(path)
    row = resolve_row(pf, row)
    local_idx, table = read_row(pf, row, ["text"])
    text = table.column("text")[local_idx].as_py()
    print(f"[{row}/{pf.metadata.num_rows}] length={len(text)}")
    print(text)


#### main ####


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Data inspection CLI for texts.parquet and tokens.parquet files.",
    )

    subparsers = parser.add_subparsers(dest="action", help="actions", required=True)

    # info
    info_parser = subparsers.add_parser(
        "info", help="show metadata and length statistics for a data Parquet file"
    )
    info_parser.add_argument(
        "path", help="path to a tokens.parquet or texts.parquet file"
    )
    info_parser.add_argument(
        "-c",
        "--col",
        default="tokens",
        help="column to sample for length statistics (default: tokens)",
    )
    info_parser.add_argument(
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
            action_info(args.path, col=args.col, row_group=args.row_group)
        case "text_show":
            action_text_show(args.path, args.row)
        case _:
            raise ValueError(f"unrecognized action: {args.action}")


if __name__ == "__main__":
    main()
