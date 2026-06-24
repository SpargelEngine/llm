"""Parquet file utilities."""

from argparse import ArgumentParser

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from spargel_llm.parquet_utils import iter_row_groups, resolve_index


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


def _parse_slice(slice_spec: str, total: int) -> tuple[int, int]:
    """Parse a ``[start]:[end]`` string into absolute row indices."""
    parts = slice_spec.split(":")
    if len(parts) != 2:
        raise ValueError(f"slice must be in the form [start]:[end], got {slice_spec!r}")
    start_str, end_str = parts
    start = int(start_str) if start_str else 0
    end = int(end_str) if end_str else total

    start = max(0, min(resolve_index(start, total), total))
    end = max(0, min(resolve_index(end, total), total))

    if start >= end:
        raise ValueError(
            f"empty slice: start={start} >= end={end} (file has {total} rows)"
        )
    return start, end


def action_copy(
    input_path: str,
    output: str,
    row_group_size: int | None = None,
):
    """Copy a Parquet file, optionally controlling row group size.

    Reads row groups from *input_path* and writes them to *output* with the
    same schema.  When *row_group_size* is given, every row group in the output
    except possibly the last will contain exactly that many rows.
    """
    pf = pq.ParquetFile(input_path)
    schema = pf.schema_arrow

    writer = pq.ParquetWriter(
        output, schema, compression="zstd", row_group_size=row_group_size
    )

    pending: list[pa.Table] = []
    pending_rows = 0

    for rg_idx in tqdm(range(pf.metadata.num_row_groups), desc="copy"):
        table = pf.read_row_group(rg_idx)

        if not row_group_size:
            writer.write_table(table)
            continue

        pending.append(table)
        pending_rows += table.num_rows

        while pending_rows >= row_group_size:
            combined = pa.concat_tables(pending)
            writer.write_table(combined.slice(0, row_group_size))
            remainder = combined.slice(row_group_size)
            pending = [remainder] if remainder.num_rows > 0 else []
            pending_rows = remainder.num_rows

    if pending:
        writer.write_table(pa.concat_tables(pending))

    writer.close()

    out_rows = pq.ParquetFile(output).metadata.num_rows
    out_rgs = pq.ParquetFile(output).metadata.num_row_groups
    print(f"{input_path} -> {output} ({out_rows:,} rows, {out_rgs} row groups)")


def action_split(input_path: str, pos: int, output1: str, output2: str):
    """Split a Parquet file into two at the given row position.

    Rows ``0..pos-1`` go to *output1*, rows ``pos..end`` go to *output2*.
    """
    pf = pq.ParquetFile(input_path)
    pos = resolve_index(pos, pf.metadata.num_rows)

    if pos < 0 or pos > pf.metadata.num_rows:
        raise ValueError(
            f"split position {pos} out of range (file has {pf.metadata.num_rows} rows)"
        )

    schema = pf.schema_arrow
    w1 = pq.ParquetWriter(output1, schema, compression="zstd")
    w2 = pq.ParquetWriter(output2, schema, compression="zstd")

    for rg_idx, offset, rg_rows in tqdm(
        iter_row_groups(pf), desc="row group", total=pf.metadata.num_row_groups
    ):
        rg_end = offset + rg_rows

        if rg_end <= pos:
            w1.write_table(pf.read_row_group(rg_idx))
        elif offset >= pos:
            w2.write_table(pf.read_row_group(rg_idx))
        else:
            table = pf.read_row_group(rg_idx)
            split_idx = pos - offset
            w1.write_table(table.slice(0, split_idx))
            w2.write_table(table.slice(split_idx))

    w1.close()
    w2.close()

    n1 = pq.ParquetFile(output1).metadata.num_rows
    n2 = pq.ParquetFile(output2).metadata.num_rows
    print(f"{input_path} -> {output1} ({n1:,} rows) + {output2} ({n2:,} rows)")


def action_slice(input_path: str, slice_spec: str, output: str):
    """Extract a slice of rows from a Parquet file.

    *slice_spec* is a string like ``[start]:[end]`` where both *start* and
    *end* may be empty to mean the beginning/end of the file.  Negative
    values count from the end, just like Python slices.
    """
    pf = pq.ParquetFile(input_path)
    start, end = _parse_slice(slice_spec, pf.metadata.num_rows)

    schema = pf.schema_arrow
    writer = pq.ParquetWriter(output, schema, compression="zstd")

    for rg_idx, offset, rg_rows in iter_row_groups(pf):
        rg_end = offset + rg_rows

        if rg_end <= start:
            continue
        if offset >= end:
            break
        if offset >= start and rg_end <= end:
            writer.write_table(pf.read_row_group(rg_idx))
        else:
            table = pf.read_row_group(rg_idx)
            local_start = max(0, start - offset)
            local_end = min(rg_rows, end - offset)
            writer.write_table(table.slice(local_start, local_end - local_start))

    writer.close()

    out_rows = pq.ParquetFile(output).metadata.num_rows
    print(f"{input_path}[{start}:{end}] -> {output} ({out_rows:,} rows)")


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

    copy_parser = subparsers.add_parser(
        "copy", help="copy a Parquet file, optionally re-chunking row groups"
    )
    copy_parser.add_argument("input", help="input Parquet file")
    copy_parser.add_argument("output", help="output Parquet file")
    copy_parser.add_argument(
        "-g",
        "--row-group-size",
        type=int,
        default=None,
        help="target rows per output row group",
    )

    slice_parser = subparsers.add_parser(
        "slice", help="extract a slice of rows from a Parquet file"
    )
    slice_parser.add_argument("input", help="input Parquet file")
    slice_parser.add_argument(
        "slice",
        help="slice specification as [start]:[end], e.g. :100, 50:, 50:100",
    )
    slice_parser.add_argument("output", help="output Parquet file")

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    match args.action:
        case "info":
            action_info(args.input)
        case "copy":
            action_copy(args.input, args.output, args.row_group_size)
        case "split":
            action_split(args.input, args.pos, args.output1, args.output2)
        case "slice":
            action_slice(args.input, args.slice, args.output)
        case _:
            raise ValueError(f"unrecognized action: {args.action}")


if __name__ == "__main__":
    main()
