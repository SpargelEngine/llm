"""Performance benchmarks for iter_batches."""

from __future__ import annotations

import time
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from spargel_llm.train import iter_batches, iter_batches_indep

RNG = np.random.default_rng(42)


def _make_parquet(rows: list[list[int]], dataset_id: str = "bench") -> pq.ParquetFile:
    schema = pa.schema([pa.field("tokens", pa.list_(pa.uint32()))])
    schema = schema.with_metadata({"dataset_id": dataset_id})
    table = pa.table(
        {"tokens": [pa.array(r, type=pa.uint32()) for r in rows]}, schema=schema
    )
    buf = pa.BufferOutputStream()
    pq.write_table(table, buf)
    return pq.ParquetFile(pa.BufferReader(buf.getvalue()))


def gen_rows(n_rows: int, mean_len: int, min_len: int = 2) -> list[list[int]]:
    lengths = RNG.poisson(mean_len, size=n_rows).clip(min_len)
    return [RNG.integers(0, 32000, size=n, dtype=np.int32).tolist() for n in lengths]


def _consume(iterator: Iterator) -> tuple[int, float]:
    count = 0
    t0 = time.perf_counter()
    for batch in iterator:
        _ = batch[0].sum(), batch[1].sum(), batch[2].sum()
        count += 1
    return count, time.perf_counter() - t0


_BASE: list[tuple[str, int, int, int | None]] = [
    # (label, seq_len, batch_size, stride)
    ("seq=128  bs=128", 128, 128, None),
    ("seq=512  bs=32", 512, 32, None),
    ("seq=512  bs=64", 512, 64, None),
    ("seq=2048 bs=16", 2048, 16, None),
    ("seq=128  bs=128  stride=64", 128, 128, 64),
]
_BOUNDARY: list[tuple[str, int, int, int | None, int | None, int | None]] = [
    ("seq=512  bs=32   +bdy", 512, 32, None, 1, 2),
]

CONFIGS: list[tuple[str, int, int, int | None, int | None, int | None, str]] = [
    # (label, seq_len, batch_size, stride, sot_index, eot_index, mode)
    *((f"{lbl}  concat", sl, bs, st, None, None, "concat") for lbl, sl, bs, st in _BASE),
    *((f"{lbl}  indep", sl, bs, st, None, None, "indep") for lbl, sl, bs, st in _BASE),
    *((f"{lbl}  indep", sl, bs, st, sot, eot, "indep") for lbl, sl, bs, st, sot, eot in _BOUNDARY),
]


def main():

    print("Preparing Datasets...")
    datasets = [
        ("short-many (50Kx~200)", _make_parquet(gen_rows(50_000, 200))),
        ("medium    (5Kx~5000)", _make_parquet(gen_rows(5_000, 5000))),
        ("long-few  (200x~50000)", _make_parquet(gen_rows(200, 50000))),
    ]
    print("=" * 24)

    header = f"{'Dataset':<28s} {'Config':<26s} {'Batches':>7s}  {'Time':>6s}"
    print(header)
    print("-" * 72)

    total_t0 = time.perf_counter()

    for ds_label, pf in datasets:
        for cfg_label, sl, bs, stride, sot, eot, mode in CONFIGS:

            def iterator():
                if mode == "indep":
                    return iter_batches_indep(
                        pf,
                        seq_len=sl,
                        batch_size=bs,
                        pad_index=0,
                        stride=stride,
                        sot_index=sot,
                        eot_index=eot,
                    )
                else:
                    return iter_batches(
                        pf,
                        seq_len=sl,
                        batch_size=bs,
                        pad_index=0,
                        sep_index=4,
                        stride=stride,
                    )

            _consume(iterator())  # warm-up
            count, sec = _consume(iterator())
            print(f"{ds_label:<28s} {cfg_label:<26s} {count:>7d}  {sec:>5.1f}s")

    total_sec = time.perf_counter() - total_t0
    print(f"\nTotal: {total_sec:.1f}s")


if __name__ == "__main__":
    main()
