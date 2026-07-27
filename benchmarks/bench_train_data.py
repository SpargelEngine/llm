"""Performance benchmarks for ParquetConcatIterator."""

from __future__ import annotations

import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from spargel_llm.train.pretrain import ParquetConcatIterator

RNG = np.random.default_rng(42)
PAD = 0
SEP = 4


def _make_parquet(
    rows: list[list[int]], row_group_size: int = 0
) -> pq.ParquetFile:
    """Create an in-memory ParquetFile from *rows*.

    If *row_group_size* is 0 (default), all rows go into a single row group.
    """
    schema = pa.schema([pa.field("tokens", pa.list_(pa.uint32()))])
    table = pa.table(
        {"tokens": [pa.array(r, type=pa.uint32()) for r in rows]}, schema=schema
    )
    buf = pa.BufferOutputStream()
    kw = {}
    if row_group_size > 0:
        kw["row_group_size"] = row_group_size
    pq.write_table(table, buf, **kw)
    return pq.ParquetFile(pa.BufferReader(buf.getvalue()))


def gen_rows(n_rows: int, mean_len: int, min_len: int = 2) -> list[list[int]]:
    """Generate *n_rows* random token sequences with Poisson-distributed lengths."""
    lengths = RNG.poisson(mean_len, size=n_rows).clip(min_len)
    return [RNG.integers(0, 32000, size=int(n), dtype=np.int32).tolist() for n in lengths]


def _bench_iteration(
    pf: pq.ParquetFile, seq_len: int, stride: int
) -> tuple[int, float, list[float]]:
    """Consume the full iterator, recording per-row-group-switch latency.

    Returns ``(total_items, total_seconds, switch_latencies_ms)``.
    """
    it = ParquetConcatIterator(pf, seq_len, pad_id=PAD, sep_id=SEP, stride=stride)
    count = 0
    switches: list[float] = []
    prev_rg = 0
    t0 = time.perf_counter()

    for _item in it:
        count += 1
        if it.group_index != prev_rg:
            t_now = time.perf_counter()
            # The latency attributable to this switch is dominated by the
            # read_row_group call inside __next__.  We record wall-clock
            # since the last switch as an upper bound.
            switches.append((t_now - t0) * 1000)
            prev_rg = it.group_index
            t0 = t_now  # reset so the next measurement is a clean interval

    elapsed = time.perf_counter() - t0
    if count > 0:
        # re-add the time since the last switch for the total
        total = sum(switches) / 1000 + elapsed
    else:
        total = 0.0
    return count, total, switches


# ── dataset shapes ──────────────────────────────────────────────────

DATASETS: list[tuple[str, int, int, int]] = [
    # (label, n_rows, mean_len, row_group_size)
    ("short-many (50K×~200,  RG=2000)", 50_000, 200, 2000),
    ("medium    (5K×~5000,  RG=500)",   5_000, 5000, 500),
    ("long-few  (200×~50000, RG=20)",     200, 50000, 20),
]

# ── configs ─────────────────────────────────────────────────────────

CONFIGS: list[tuple[str, int, int]] = [
    # (label, seq_len, stride)
    ("seq=128  stride=129", 128, 0),       # 0 → default (seq_len+1)
    ("seq=512  stride=513", 512, 0),
    ("seq=2048 stride=2049", 2048, 0),
    ("seq=128  stride=64", 128, 64),       # overlapping
    ("seq=512  stride=256", 512, 256),
]

# ── main ────────────────────────────────────────────────────────────


def main() -> None:
    print("Preparing datasets ...")
    datasets: list[tuple[str, pq.ParquetFile]] = []
    for label, n_rows, mean_len, rg_size in DATASETS:
        rows = gen_rows(n_rows, mean_len)
        pf = _make_parquet(rows, row_group_size=rg_size)
        n_rg = pf.metadata.num_row_groups
        datasets.append((f"{label}  ({n_rg} RGs)", pf))

    header = (
        f"{'Dataset':<40s} {'Config':<24s}"
        f" {'Items':>8s}  {'Time':>7s}  {'items/s':>9s}"
        f"  {'RG switch':>10s}"
    )
    print()
    print(header)
    print("-" * 110)

    for ds_label, pf in datasets:
        for cfg_label, sl, stride in CONFIGS:
            # warm-up
            _bench_iteration(pf, sl, stride)

            count, sec, switches = _bench_iteration(pf, sl, stride)
            ips = count / sec if sec > 0 else 0

            if switches:
                sw_avg = np.mean(switches)
                sw_max = np.max(switches)
                sw_str = f"avg={sw_avg:.0f}  max={sw_max:.0f} ms"
            else:
                sw_str = "—"

            print(
                f"{ds_label:<40s} {cfg_label:<24s}"
                f" {count:>8d}  {sec:>6.2f}s  {ips:>8.0f}"
                f"  {sw_str:>10s}"
            )

    print()


if __name__ == "__main__":
    main()
