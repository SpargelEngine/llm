"""Print the text content of a specific row from a process_texts.py output parquet file.

Only reads the row group containing the target row, skipping all others.
"""

from argparse import ArgumentParser

import pyarrow.parquet as pq


def main():
    parser = ArgumentParser(description="Print text at row N from a Parquet file.")
    parser.add_argument("path", help="path to the Parquet file")
    parser.add_argument(
        "row",
        type=int,
        help="zero-based row index to print (negative indices count from the end)",
    )
    args = parser.parse_args()

    pf = pq.ParquetFile(args.path)

    if args.row < 0:
        args.row += pf.metadata.num_rows

    if args.row < 0 or args.row >= pf.metadata.num_rows:
        raise IndexError(
            f"row index out of range (file has {pf.metadata.num_rows} rows)"
        )

    # Find which row group contains the target row
    offset = 0
    for rg_idx in range(pf.metadata.num_row_groups):
        rg = pf.metadata.row_group(rg_idx)
        if offset + rg.num_rows > args.row:
            # Read only this row group and extract the target row
            table = pf.read_row_group(rg_idx, columns=["text"])
            local_idx = args.row - offset
            print(table.column("text")[local_idx].as_py())
            return
        offset += rg.num_rows


if __name__ == "__main__":
    main()
