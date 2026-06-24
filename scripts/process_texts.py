from argparse import ArgumentParser
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from spargel_llm.parquet_utils import make_text_schema
from spargel_llm.text_pass import load_texts


def main():
    parser = ArgumentParser(
        "Text Processing CLI Tool",
        description="Load a text pass config and write results to a Parquet file.",
    )

    parser.add_argument(
        "config",
        help="path to text pass JSON config",
    )
    parser.add_argument(
        "output",
        help="output Parquet file path",
    )
    parser.add_argument(
        "--batch-size",
        "-bs",
        type=int,
        default=1000,
        help="row group size for Parquet writing (default: 1000)",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    schema = make_text_schema()

    with pq.ParquetWriter(output_path, schema, compression="zstd") as writer:
        batch: list[str] = []
        for text in tqdm(load_texts(args.config)):
            batch.append(text)
            if len(batch) >= args.batch_size:
                writer.write_table(pa.table({"text": batch}, schema=schema))
                batch = []
        if batch:
            writer.write_table(pa.table({"text": batch}, schema=schema))

    print(f"Parquet saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
