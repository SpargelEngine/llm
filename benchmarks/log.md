# iter_batches benchmark log

## 2025-06-25 — initial

**Changes from previous**:

- `_augment_row`: return zero-copy view when no SOT/EOT tokens (skip allocation + copy)
- buffer swap: only reset mask — stale input/target data is harmless since mask=True hides all unwritten positions

**Micro-benchmarks**: `_augment_row` 1.32× faster, fill op 8–10× faster.

**End-to-end** (synthetic data, average of 5 runs):

```
Dataset                      Config                     Batches    Time
------------------------------------------------------------------------
short-many (50K×~200)        seq=128  bs=128                781    0.3s
short-many (50K×~200)        seq=512  bs=32                1562    0.2s
short-many (50K×~200)        seq=512  bs=64                 781    0.2s
short-many (50K×~200)        seq=2048 bs=16                3125    0.3s
short-many (50K×~200)        seq=512  bs=32  +bdy          1562    0.3s
short-many (50K×~200)        seq=128  bs=128 stride=64     1435    0.4s
medium    (5K×~5000)         seq=128  bs=128               1544    0.5s
medium    (5K×~5000)         seq=512  bs=32                1569    0.3s
medium    (5K×~5000)         seq=512  bs=64                 784    0.3s
medium    (5K×~5000)         seq=2048 bs=16                 937    0.3s
medium    (5K×~5000)         seq=512  bs=32  +bdy          1569    0.3s
medium    (5K×~5000)         seq=128  bs=128 stride=64     3069    0.9s
long-few  (200×~50000)       seq=128  bs=128                611    0.2s
long-few  (200×~50000)       seq=512  bs=32                 613    0.1s
long-few  (200×~50000)       seq=512  bs=64                 306    0.1s
long-few  (200×~50000)       seq=2048 bs=16                 312    0.1s
long-few  (200×~50000)       seq=512  bs=32  +bdy           613    0.1s
long-few  (200×~50000)       seq=128  bs=128 stride=64     1221    0.3s
```

Total runtime: ~11s.

**Summary**: Combined optimizations target ~6.5% of profile (fill ~2%, augment ~4%). Real bottlenecks remain Parquet I/O (33%) and Python iteration overhead (35%), which require structural changes.
