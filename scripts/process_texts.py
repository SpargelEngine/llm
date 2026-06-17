from argparse import ArgumentParser
from pathlib import Path
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

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
    args = parser.parse_args()

    output_path = Path(args.output)
    schema = pa.schema([pa.field("text", pa.string())])
    dataset_id = uuid4().hex
    schema = schema.with_metadata({"dataset_id": dataset_id})

    WRITE_BATCH = 1000

    with pq.ParquetWriter(output_path, schema, compression="zstd") as writer:
        batch: list[str] = []
        for text in tqdm(load_texts(args.config)):
            batch.append(text)
            if len(batch) >= WRITE_BATCH:
                writer.write_table(pa.table({"text": batch}, schema=schema))
                batch = []
        if batch:
            writer.write_table(pa.table({"text": batch}, schema=schema))

    print(f"Parquet saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
