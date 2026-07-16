"""CLI for data files."""

import os
import random
from argparse import ArgumentParser

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from spargel_llm.parquet_utils import (
    concat_parquet_files,
    get_dataset_id,
    iter_row_groups,
    read_row,
    resolve_index,
    resolve_row,
    with_new_dataset_id,
)

#### shared helpers ####


def _distribute_rows(pf, output_paths, schema, rng, show_progress=True):
    """Distribute rows randomly across output files.

    Parameters
    ----------
    pf : pq.ParquetFile
        Input Parquet file.
    output_paths : list[str]
        Paths of the output files, one per bucket.
    schema : pa.Schema
        Schema for the output files (caller decides whether to set a dataset ID).
    rng : random.Random
        Random number generator.
    show_progress : bool
        Whether to show a tqdm progress bar.
    """
    writers: list[pq.ParquetWriter] = []
    try:
        for p in output_paths:
            writers.append(pq.ParquetWriter(p, schema, compression="zstd"))
        num_files = len(output_paths)
        iterator = range(pf.metadata.num_row_groups)
        if show_progress:
            iterator = tqdm(iterator, desc="distribute")
        for rg_idx in iterator:
            table = pf.read_row_group(rg_idx)
            n = table.num_rows
            assignments = [rng.randrange(num_files) for _ in range(n)]
            buckets: dict[int, list[int]] = {i: [] for i in range(num_files)}
            for row_i, target in enumerate(assignments):
                buckets[target].append(row_i)
            for target, indices in buckets.items():
                if indices:
                    chunk = table.take(pa.array(indices))
                    writers[target].write_table(chunk)
    finally:
        for w in writers:
            w.close()


#### actions ####
def action_concat(output: str, inputs: list[str]):
    """Concatenate multiple Parquet files into one."""
    total = concat_parquet_files(inputs, output)
    out_rgs = pq.ParquetFile(output).metadata.num_row_groups
    print(f"{len(inputs)} files -> {output} ({total:,} rows, {out_rgs} row groups)")


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
    schema = with_new_dataset_id(pf.schema_arrow)
    written_rows = 0

    with pq.ParquetWriter(
        output, schema, compression="zstd", row_group_size=row_group_size
    ) as writer:
        pending: list[pa.Table] = []
        pending_rows = 0

        for rg_idx in tqdm(range(pf.metadata.num_row_groups), desc="copy"):
            table = pf.read_row_group(rg_idx)

            if not row_group_size:
                writer.write_table(table)
                written_rows += table.num_rows
                continue

            pending.append(table)
            pending_rows += table.num_rows

            while pending_rows >= row_group_size:
                combined = pa.concat_tables(pending)
                pending.clear()
                chunk = combined.slice(0, row_group_size)
                writer.write_table(chunk)
                written_rows += chunk.num_rows
                remainder = combined.slice(row_group_size)
                pending = [remainder] if remainder.num_rows > 0 else []
                pending_rows = remainder.num_rows

        if pending:
            combined = pending[0] if len(pending) == 1 else pa.concat_tables(pending)
            writer.write_table(combined)
            written_rows += combined.num_rows

    out_rgs = pq.ParquetFile(output).metadata.num_row_groups
    print(f"{input_path} -> {output} ({written_rows:,} rows, {out_rgs} row groups)")


def action_dist(
    input_path: str,
    output_format: str,
    num_files: int,
    random_seed: int | None = None,
):
    """Distribute rows evenly across N output Parquet files.

    Each row is assigned to one of *num_files* outputs with equal probability.
    *output_format* is a printf-style pattern, e.g. ``output-%04d.parquet``.
    """
    if num_files < 2:
        raise ValueError(f"num_files must be >= 2, got {num_files}")

    pf = pq.ParquetFile(input_path)
    output_paths = [output_format % i for i in range(num_files)]
    rng = random.Random(random_seed)
    _distribute_rows(pf, output_paths, with_new_dataset_id(pf.schema_arrow), rng)
    print(f"{input_path} distributed across {num_files} files.")


def _open_and_print_parquet_metadata(path: str):
    """Open a Parquet file, print its metadata, and return it."""
    pf = pq.ParquetFile(path)
    print(f"File:             {path}")
    print(f"Rows:             {pf.metadata.num_rows:,}")
    print(f"Row groups:       {pf.metadata.num_row_groups}")
    print("Schema:")
    print(pf.schema_arrow)
    print("=" * 16)

    dataset_id = get_dataset_id(pf)
    if dataset_id:
        print(f"Dataset ID:       {dataset_id}")

    return pf


def _lengths_from_column(table, col_name: str) -> np.ndarray:
    """Extract per-row lengths from a column in a row group table."""
    col = table.column(col_name)
    chunk = col.combine_chunks() if col.num_chunks > 1 else col.chunk(0)
    if col_name == "text":
        offsets = np.frombuffer(chunk.buffers()[1], dtype=np.int32)
    else:
        offsets = chunk.offsets.to_numpy()
    return offsets[1:] - offsets[:-1]


def _resolve_col(pf: pq.ParquetFile, col: str | None) -> str:
    """Resolve the column to use for length statistics.

    If *col* is given, use it directly.  Otherwise try ``"text"`` first,
    then ``"tokens"``.
    """
    if col is not None:
        return col
    schema_names = pf.schema_arrow.names
    for candidate in ("text", "tokens"):
        if candidate in schema_names:
            return candidate
    raise ValueError(
        f"no column specified and neither 'text' nor 'tokens' found in {schema_names}"
    )


def action_info(path: str, col: str | None = None, row_group: int = 0):
    """Show metadata and length statistics for a data Parquet file."""
    pf = _open_and_print_parquet_metadata(path)
    col = _resolve_col(pf, col)
    table = pf.read_row_group(row_group, columns=[col])
    lengths = _lengths_from_column(table, col)

    label = "characters" if col == "text" else "tokens"
    print(f"Length statistics from row group {row_group}:")
    print(f"  Number of rows:   {table.num_rows}")
    print(f"  Total {label}:     {lengths.sum():,}")
    print(f"  Average length:   {lengths.mean():.1f}")
    print(f"  Min length:       {lengths.min()}")
    print(f"  Max length:       {lengths.max()}")


def action_shuffle(
    input_path: str,
    output: str,
    random_seed: int | None = None,
    buckets: int | None = None,
    temp_format: str | None = None,
):
    """Shuffle all rows of a Parquet file and write to *output*.

    When *buckets* is given, rows are first randomly distributed into
    *buckets* temporary files, each bucket is shuffled independently, and
    the results are concatenated.  This limits peak memory usage to
    approximately 1/*buckets* of the full dataset while producing a
    uniformly random permutation.
    """
    rng = random.Random(random_seed)
    pf = pq.ParquetFile(input_path)

    if buckets is None or buckets <= 1:
        # All-in-memory shuffle
        print(f"Reading {pf.metadata.num_rows:,} rows from {input_path} ...")
        table = pf.read()
        indices = list(range(table.num_rows))
        rng.shuffle(indices)
        shuffled = table.take(pa.array(indices))
        del table, indices
        schema = with_new_dataset_id(pf.schema_arrow)
        with pq.ParquetWriter(output, schema, compression="zstd") as writer:
            writer.write_table(shuffled)
        out_rgs = pq.ParquetFile(output).metadata.num_row_groups
        print(
            f"{input_path} -> {output} ({shuffled.num_rows:,} rows, {out_rgs} row groups)"
        )
        return

    # Bucketed shuffle
    total_rows = pf.metadata.num_rows
    print(f"Bucketed shuffle: {total_rows:,} rows across {buckets} buckets")

    # Step 1: distribute rows randomly into temp files
    if temp_format is None:
        uid = os.urandom(4).hex()
        temp_format = f".shuffle-bucket-%04d-{uid}.parquet"
    temp_paths = [temp_format % i for i in range(buckets)]
    _distribute_rows(pf, temp_paths, pf.schema_arrow, rng)

    # Step 2: shuffle each bucket independently
    for temp_path in tqdm(temp_paths, desc="shuffle buckets"):
        pf_temp = pq.ParquetFile(temp_path)
        table = pf_temp.read()
        indices = list(range(table.num_rows))
        rng.shuffle(indices)
        shuffled = table.take(pa.array(indices))
        del table, indices
        schema = pf_temp.schema_arrow
        with pq.ParquetWriter(temp_path, schema, compression="zstd") as writer:
            writer.write_table(shuffled)

    # Step 3: concatenate into final output
    total = concat_parquet_files(temp_paths, output)

    # Step 4: clean up temp files
    for temp_path in temp_paths:
        try:
            os.remove(temp_path)
        except FileNotFoundError:
            pass

    out_rgs = pq.ParquetFile(output).metadata.num_row_groups
    print(f"{input_path} -> {output} ({total:,} rows, {out_rgs} row groups)")


def _parse_slice(slice_spec: str, total: int) -> tuple[int, int]:
    """Parse a ``[start]:[end]`` string into absolute row indices."""
    parts = slice_spec.split(":")
    if len(parts) != 2:
        raise ValueError(f"slice must be in the form [start]:[end], got {slice_spec!r}")
    start_str, end_str = parts
    start = int(start_str) if start_str else 0
    end = int(end_str) if end_str else total

    start = min(max(0, resolve_index(start, total)), total)
    end = min(max(0, resolve_index(end, total)), total)

    if start >= end:
        raise ValueError(
            f"empty slice: start={start} >= end={end} (file has {total} rows)"
        )
    return start, end


def action_slice(input_path: str, slice_spec: str, output: str):
    """Extract a slice of rows from a Parquet file.

    *slice_spec* is a string like ``[start]:[end]`` where both *start* and
    *end* may be empty to mean the beginning/end of the file.  Negative
    values count from the end, just like Python slices.
    """
    pf = pq.ParquetFile(input_path)
    start, end = _parse_slice(slice_spec, pf.metadata.num_rows)

    schema = with_new_dataset_id(pf.schema_arrow)
    written_rows = 0

    print(f"Slicing {input_path}[{start}:{end}] ...")

    with pq.ParquetWriter(output, schema, compression="zstd") as writer:
        for rg_idx, offset, rg_rows in iter_row_groups(pf):
            rg_end = offset + rg_rows

            if rg_end <= start:
                continue
            if offset >= end:
                break
            if offset >= start and rg_end <= end:
                table = pf.read_row_group(rg_idx)
                writer.write_table(table)
                written_rows += table.num_rows
            else:
                table = pf.read_row_group(rg_idx)
                local_start = max(0, start - offset)
                local_end = min(rg_rows, end - offset)
                chunk = table.slice(local_start, local_end - local_start)
                writer.write_table(chunk)
                written_rows += chunk.num_rows

    print(f"{input_path}[{start}:{end}] -> {output} ({written_rows:,} rows)")


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

    schema = with_new_dataset_id(pf.schema_arrow)
    n1 = n2 = 0

    with (
        pq.ParquetWriter(output1, schema, compression="zstd") as w1,
        pq.ParquetWriter(output2, schema, compression="zstd") as w2,
    ):
        for rg_idx, offset, rg_rows in tqdm(
            iter_row_groups(pf), desc="row group", total=pf.metadata.num_row_groups
        ):
            rg_end = offset + rg_rows

            if rg_end <= pos:
                table = pf.read_row_group(rg_idx)
                w1.write_table(table)
                n1 += table.num_rows
            elif offset >= pos:
                table = pf.read_row_group(rg_idx)
                w2.write_table(table)
                n2 += table.num_rows
            else:
                table = pf.read_row_group(rg_idx)
                split_idx = pos - offset
                first = table.slice(0, split_idx)
                second = table.slice(split_idx)
                w1.write_table(first)
                w2.write_table(second)
                n1 += first.num_rows
                n2 += second.num_rows

    print(f"{input_path} -> {output1} ({n1:,} rows) + {output2} ({n2:,} rows)")


def action_text_show(path: str, row: int | None = None, field: str = "text"):
    """Print the text content of a specific row from a texts.parquet file.

    Only reads the row group containing the target row, skipping all others.
    If row is not specified, a random row is selected.
    """
    pf = pq.ParquetFile(path)
    row = resolve_row(pf, row)
    local_idx, table = read_row(pf, row, [field])
    text = table.column(field)[local_idx].as_py()
    print(f"[{row}/{pf.metadata.num_rows}] length={len(text)}")
    print(text)


#### main ####


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Data inspection and manipulation CLI for data files.",
    )

    subparsers = parser.add_subparsers(dest="action", help="actions", required=True)

    # concat
    concat_parser = subparsers.add_parser(
        "concat", help="concatenate multiple Parquet files into one"
    )
    concat_parser.add_argument("output", help="output Parquet file")
    concat_parser.add_argument(
        "inputs", nargs="+", help="input Parquet files to concatenate"
    )

    # copy
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

    # dist
    dist_parser = subparsers.add_parser(
        "dist", help="distribute rows evenly across N output Parquet files"
    )
    dist_parser.add_argument("input", help="input Parquet file")
    dist_parser.add_argument(
        "output_format",
        help="printf-style output path pattern, e.g. output-%%04d.parquet",
    )
    dist_parser.add_argument(
        "num_files", type=int, help="number of output files to distribute across"
    )
    dist_parser.add_argument(
        "-rs", "--random-seed", type=int, default=None, help="random seed"
    )

    # info
    info_parser = subparsers.add_parser(
        "info", help="show metadata and length statistics for a data Parquet file"
    )
    info_parser.add_argument(
        "path", help="path to a tokens.parquet or texts.parquet file"
    )
    info_parser.add_argument(
        "-f",
        "--field",
        default=None,
        help="column to sample for length statistics (default: try 'text', then 'tokens')",
    )
    info_parser.add_argument(
        "-rg",
        "--row-group",
        type=int,
        default=0,
        help="row group index to sample for length statistics (default: 0)",
    )

    # shuffle
    shuffle_parser = subparsers.add_parser(
        "shuffle", help="randomly shuffle all rows of a Parquet file"
    )
    shuffle_parser.add_argument("input", help="input Parquet file")
    shuffle_parser.add_argument("output", help="output Parquet file")
    shuffle_parser.add_argument(
        "-rs", "--random-seed", type=int, default=None, help="random seed"
    )
    shuffle_parser.add_argument(
        "-b",
        "--buckets",
        type=int,
        default=None,
        help="number of buckets for memory-efficient shuffle",
    )
    shuffle_parser.add_argument(
        "--temp-format",
        default=None,
        help="printf-style pattern for temp files, e.g. /tmp/shuf-%%04d.parquet",
    )

    # slice
    slice_parser = subparsers.add_parser(
        "slice", help="extract a slice of rows from a Parquet file"
    )
    slice_parser.add_argument("input", help="input Parquet file")
    slice_parser.add_argument(
        "slice",
        help="slice specification as [start]:[end], e.g. :100, 50:, 50:100",
    )
    slice_parser.add_argument("output", help="output Parquet file")

    # split
    split_parser = subparsers.add_parser("split", help="split a Parquet file into two")
    split_parser.add_argument("input", help="input Parquet file")
    split_parser.add_argument(
        "pos",
        type=int,
        help="split position (rows 0..pos-1 → output1, pos..end → output2)",
    )
    split_parser.add_argument("output1", help="first output file")
    split_parser.add_argument("output2", help="second output file")

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
    text_show_parser.add_argument(
        "--field",
        "-f",
        default="text",
        help="column name to read from the Parquet file (default: text)",
    )

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    match args.action:
        case "concat":
            action_concat(args.output, args.inputs)
        case "copy":
            action_copy(args.input, args.output, args.row_group_size)
        case "dist":
            action_dist(
                args.input, args.output_format, args.num_files, args.random_seed
            )
        case "info":
            action_info(args.path, col=args.field, row_group=args.row_group)
        case "shuffle":
            action_shuffle(
                args.input,
                args.output,
                args.random_seed,
                args.buckets,
                args.temp_format,
            )
        case "slice":
            action_slice(args.input, args.slice, args.output)
        case "split":
            action_split(args.input, args.pos, args.output1, args.output2)
        case "text_show":
            action_text_show(args.path, args.row, field=args.field)
        case _:
            raise ValueError(f"unrecognized action: {args.action}")


if __name__ == "__main__":
    main()
