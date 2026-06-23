"""Parquet file utilities."""

from argparse import ArgumentParser

from tqdm import tqdm
import pyarrow.parquet as pq


def action_info(input_path: str):
    """Display basic information about a Parquet file."""
    pf = pq.ParquetFile(input_path)
    meta = pf.metadata
    schema = pf.schema_arrow

    print(f"File:            {input_path}")
    print(f"Format version:  {meta.format_version}")
    print(f"Created by:      {meta.created_by}")
    print(f"Num rows:        {meta.num_rows:,}")
    print(f"Num row groups:  {meta.num_row_groups}")
    print(f"Num columns:     {meta.num_columns}")
    print(f"Serialized size: {meta.serialized_size:,} bytes")
    print()
    print("Schema:")
    for i, field in enumerate(schema):
        print(f"  {i}: {field.name}  {field.type}")

    total_rg_size = sum(
        meta.row_group(i).total_byte_size for i in range(meta.num_row_groups)
    )
    avg_rows = meta.num_rows / meta.num_row_groups
    avg_size = total_rg_size / meta.num_row_groups
    print()
    print(
        f"Row groups:      {meta.num_row_groups} "
        f"(avg {avg_rows:,.0f} rows, {avg_size:,.0f} bytes each)"
    )


def action_split(input_path: str, pos: int, output1: str, output2: str):
    """Split a Parquet file into two at the given row position.

    Rows ``0..pos-1`` go to *output1*, rows ``pos..end`` go to *output2*.
    """
    pf = pq.ParquetFile(input_path)
    total_rows = pf.metadata.num_rows

    if pos < 0:
        pos += total_rows
    if pos < 0 or pos > total_rows:
        raise ValueError(
            f"split position {pos} out of range (file has {total_rows} rows)"
        )

    schema = pf.schema_arrow

    writer1 = pq.ParquetWriter(output1, schema, compression="zstd")
    writer2 = pq.ParquetWriter(output2, schema, compression="zstd")

    offset = 0
    for rg_idx in tqdm(range(pf.metadata.num_row_groups), desc="row group"):
        rg = pf.metadata.row_group(rg_idx)
        rg_rows = rg.num_rows
        rg_end = offset + rg_rows

        if rg_end <= pos:
            # Entirely in part 1.
            table = pf.read_row_group(rg_idx)
            writer1.write_table(table)
        elif offset >= pos:
            # Entirely in part 2.
            table = pf.read_row_group(rg_idx)
            writer2.write_table(table)
        else:
            # Straddles the boundary – slice the row group.
            table = pf.read_row_group(rg_idx)
            split_idx = pos - offset
            writer1.write_table(table.slice(0, split_idx))
            writer2.write_table(table.slice(split_idx))

        offset = rg_end

    writer1.close()
    writer2.close()

    size1 = pq.ParquetFile(output1).metadata.num_rows
    size2 = pq.ParquetFile(output2).metadata.num_rows
    print(f"{input_path} -> {output1} ({size1:,} rows) + {output2} ({size2:,} rows)")


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Parquet file utilities")

    subparsers = parser.add_subparsers(dest="action", required=True)

    info_parser = subparsers.add_parser(
        "info", help="display basic information about a Parquet file"
    )
    info_parser.add_argument("input", help="Parquet file to inspect")

    split_parser = subparsers.add_parser("split", help="split a Parquet file into two")
    split_parser.add_argument("input", help="input Parquet file")
    split_parser.add_argument(
        "pos",
        type=int,
        help="split position (rows 0..pos-1 → output1, pos..end → output2)",
    )
    split_parser.add_argument("output1", help="first output file")
    split_parser.add_argument("output2", help="second output file")

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    match args.action:
        case "info":
            action_info(args.input)
        case "split":
            action_split(args.input, args.pos, args.output1, args.output2)
        case _:
            raise ValueError(f"unrecognized action: {args.action}")


if __name__ == "__main__":
    main()
