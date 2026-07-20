"""Unit tests for spargel_llm.train."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from spargel_llm.train import (
    TrainTracker,
    _augment_row,
    _yield_batches_from_arrays,
    iter_batches,
    iter_batches_indep,
)

SEP = 4


def _make_parquet(rows: list[list[int]], dataset_id: str = "test") -> pq.ParquetFile:
    schema = pa.schema([pa.field("tokens", pa.list_(pa.uint32()))])
    schema = schema.with_metadata({"dataset_id": dataset_id})
    table = pa.table(
        {"tokens": [pa.array(r, type=pa.uint32()) for r in rows]}, schema=schema
    )
    buf = pa.BufferOutputStream()
    pq.write_table(table, buf)
    return pq.ParquetFile(pa.BufferReader(buf.getvalue()))


def _make_parquet_multi_rg(
    rows: list[list[int]], row_group_size: int, dataset_id: str = "test"
) -> pq.ParquetFile:
    schema = pa.schema([pa.field("tokens", pa.list_(pa.uint32()))])
    schema = schema.with_metadata({"dataset_id": dataset_id})
    table = pa.table(
        {"tokens": [pa.array(r, type=pa.uint32()) for r in rows]}, schema=schema
    )
    buf = pa.BufferOutputStream()
    pq.write_table(table, buf, row_group_size=row_group_size)
    return pq.ParquetFile(pa.BufferReader(buf.getvalue()))


# ═══════════════════════════════════════════════════════════════════
# _augment_row
# ═══════════════════════════════════════════════════════════════════


class TestAugmentRow:
    def test_no_boundary(self):
        v = np.array([10, 20, 30, 40], dtype=np.uint32)
        r = _augment_row(v, 1, 3)
        np.testing.assert_array_equal(r, [20, 30])

    def test_sot(self):
        v = np.array([10, 20, 30], dtype=np.uint32)
        r = _augment_row(v, 0, 3, sot_index=2)
        np.testing.assert_array_equal(r, [2, 10, 20, 30])

    def test_eot(self):
        v = np.array([10, 20, 30], dtype=np.uint32)
        r = _augment_row(v, 0, 3, eot_index=3)
        np.testing.assert_array_equal(r, [10, 20, 30, 3])

    def test_both(self):
        v = np.array([10, 20, 30], dtype=np.uint32)
        r = _augment_row(v, 0, 3, sot_index=2, eot_index=3)
        np.testing.assert_array_equal(r, [2, 10, 20, 30, 3])

    def test_empty_with_boundaries(self):
        v = np.array([], dtype=np.uint32)
        r = _augment_row(v, 0, 0, sot_index=2, eot_index=3)
        np.testing.assert_array_equal(r, [2, 3])

    def test_slice(self):
        v = np.array([5, 10, 20, 30, 40], dtype=np.uint32)
        r = _augment_row(v, 1, 4, sot_index=1, eot_index=2)
        np.testing.assert_array_equal(r, [1, 10, 20, 30, 2])


# ═══════════════════════════════════════════════════════════════════
# _yield_batches_from_arrays  (unit tests on plain numpy arrays)
# ═══════════════════════════════════════════════════════════════════


def _collect_from_arrays(seq_len, batch_size, arrays, *, stride=None):
    """Run _yield_batches_from_arrays and return list of (inputs, targets, non_pad)."""
    if stride is None:
        stride = seq_len + 1
    result: list[tuple[list[list[int]], list[list[int]], int]] = []
    it = _yield_batches_from_arrays(
        iter(arrays),
        seq_len,
        batch_size,
        pad_index=0,
        stride=stride,
    )
    for batch in it:
        result.append(
            (
                batch.input_ids.tolist(),
                batch.target_ids.tolist(),
                batch.tokens_non_pad,
            )
        )
    return result


class TestYieldBatchesFromArrays:
    def test_basic(self):
        """seq_len=3, stride=4: two non-overlapping windows."""
        src = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
        batches = _collect_from_arrays(3, 2, [(src, 0, None)], stride=4)
        assert len(batches) == 1
        inputs, targets, non_pad = batches[0]
        assert inputs == [[1, 2, 3], [5, 6, 0]]
        assert targets == [[2, 3, 4], [6, 7, 0]]
        assert non_pad == 5

    def test_overlapping_stride(self):
        """stride=1 produces overlapping windows."""
        src = np.array([1, 2, 3, 4], dtype=np.int32)
        batches = _collect_from_arrays(2, 4, [(src, 0, None)], stride=1)
        assert len(batches) == 1
        inputs, targets, non_pad = batches[0]
        assert inputs == [[1, 2], [2, 3], [3, 0], [0, 0]]
        assert targets == [[2, 3], [3, 4], [4, 0], [0, 0]]
        assert non_pad == 5

    def test_start_pos(self):
        """start_pos skips initial windows."""
        src = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
        batches = _collect_from_arrays(2, 2, [(src, 2, None)], stride=2)
        assert batches[0][0] == [[3, 4], [5, 0]]

    def test_on_sample(self):
        """on_sample callback is invoked after each sample."""
        src = np.array([10, 20, 30, 40], dtype=np.int32)
        positions: list[int] = []

        def on_sample(pos: int) -> None:
            positions.append(pos)

        _collect_from_arrays(2, 8, [(src, 0, on_sample)], stride=2)
        # len=4, max_pos=2, stride=2. pos=0→2→4(stop). 2 samples.
        assert positions == [2, 4]

    def test_on_sample_with_tracker(self):
        """Tracker is updated via closure — caller owns the mapping."""
        src = np.array([1, 2, 3, 9, 4, 5], dtype=np.int32)
        bp = np.array([0, 4], dtype=np.int64)
        ri = np.array([0, 1], dtype=np.int64)
        tracker = TrainTracker(0, 0)
        cursor = 0
        n = len(bp)

        def on_sample(pos: int) -> None:
            nonlocal cursor
            while cursor + 1 < n and bp[cursor + 1] <= pos:
                cursor += 1
            tracker.index = int(ri[cursor])
            tracker.offset = pos - int(bp[cursor])

        _collect_from_arrays(2, 8, [(src, 0, on_sample)], stride=2)
        assert tracker.index == 1
        assert tracker.offset == 2


# ═══════════════════════════════════════════════════════════════════
# iter_batches  (end-to-end)
# ═══════════════════════════════════════════════════════════════════


def _first_batch(iter_fn, *args, **kwargs):
    """Return (inputs, masks, targets) from the first batch of *iter_fn*."""
    inputs, masks, targets, _, _, _ = next(iter(iter_fn(*args, **kwargs)))
    return inputs, masks, targets


def _assert_tracker_resume(iter_fn, pf, **iter_kw):
    """Resume from tracker produces a valid suffix with no overlap."""

    def samples(t: TrainTracker):
        out = []
        for inputs, _, targets, sv, _, _ in iter_fn(
            pf,
            seq_len=2,
            batch_size=2,
            pad_index=0,
            stride=2,
            tracker=t,
            **iter_kw,
        ):
            for s in range(2):
                if sv[s]:
                    out.append((inputs[s].tolist(), targets[s].tolist()))
        return out

    tracker = TrainTracker(0, 0)
    it = iter_fn(
        pf,
        seq_len=2,
        batch_size=2,
        pad_index=0,
        stride=2,
        tracker=tracker,
        **iter_kw,
    )
    inputs1, _, targets1, sv1, _, _ = next(it)
    seg1 = [(inputs1[s].tolist(), targets1[s].tolist()) for s in range(2) if sv1[s]]

    resume_tracker = TrainTracker(tracker.index, tracker.offset)
    seg2 = samples(resume_tracker)

    if seg1 and seg2:
        assert seg1[-1] != seg2[0], "resume should not duplicate"

    combined = seg1 + seg2
    assert len(combined) == len({(tuple(a), tuple(b)) for a, b in combined})


class TestIterBatches:
    def test_two_rows(self):
        pf = _make_parquet([[1, 2, 3], [10, 20, 30]])
        inputs, _, targets = _first_batch(
            iter_batches,
            pf,
            seq_len=4,
            batch_size=1,
            pad_index=0,
            stride=4,
            sep_index=SEP,
        )
        # flat = [1,2,3,4,10,20,30], len=7. max_pos=5.
        # pos=0 L=4: [1,2,3,4], [2,3,4,10]
        np.testing.assert_array_equal(inputs[0], [1, 2, 3, 4])
        np.testing.assert_array_equal(targets[0], [2, 3, 4, 10])

    def test_sep_between_rows(self):
        pf = _make_parquet([[1, 2], [5, 6], [9, 10]])
        inputs, _, targets = _first_batch(
            iter_batches,
            pf,
            seq_len=2,
            batch_size=4,
            pad_index=0,
            stride=2,
            sep_index=SEP,
        )
        # flat = [1,2,4,5,6,4,9,10], len=8.
        # pos=0: [1,2], pos=2: [4,5], pos=4: [6,4], pos=6: [9,0]
        np.testing.assert_array_equal(inputs, [[1, 2], [4, 5], [6, 4], [9, 0]])

    def test_single_row_no_trailing_sep(self):
        pf = _make_parquet([[1, 2, 3, 4, 5]])
        inputs, _, targets = _first_batch(
            iter_batches,
            pf,
            seq_len=3,
            batch_size=2,
            pad_index=0,
            stride=3,
            sep_index=SEP,
        )
        # flat = [1,2,3,4,5], len=5. pos=0: [1,2,3], pos=3 L=1: [4,0,0]
        np.testing.assert_array_equal(inputs, [[1, 2, 3], [4, 0, 0]])
        np.testing.assert_array_equal(targets, [[2, 3, 4], [5, 0, 0]])

    def test_partial_batch(self):
        pf = _make_parquet([[1, 2, 3, 4]])
        batches = list(
            iter_batches(
                pf,
                seq_len=2,
                batch_size=4,
                pad_index=0,
                stride=2,
                sep_index=SEP,
            )
        )
        assert len(batches) == 1
        inputs, masks, targets, _, _, non_pad = batches[0]
        # flat=[1,2,3,4], len=4. pos=0 L=2: [1,2], pos=2 L=1: [3,0]. 2 windows.
        np.testing.assert_array_equal(inputs, [[1, 2], [3, 0], [0, 0], [0, 0]])
        np.testing.assert_array_equal(
            masks, [[False, False], [False, True], [False, False], [False, False]]
        )
        assert non_pad == 3

    def test_start_index(self):
        pf = _make_parquet([[10, 20], [30, 40], [1, 2, 3], [4, 5, 6]])
        inputs, _, _ = _first_batch(
            iter_batches,
            pf,
            seq_len=2,
            batch_size=3,
            pad_index=0,
            stride=2,
            tracker=TrainTracker(index=2, offset=0),
            sep_index=SEP,
        )
        np.testing.assert_array_equal(inputs, [[1, 2], [3, 4], [4, 5]])

    def test_start_offset(self):
        pf = _make_parquet([[1, 2, 3, 4, 5, 6]])
        inputs, _, _ = _first_batch(
            iter_batches,
            pf,
            seq_len=2,
            batch_size=2,
            pad_index=0,
            stride=2,
            tracker=TrainTracker(index=0, offset=2),
            sep_index=SEP,
        )
        # offset=2: row becomes [3,4,5,6]. flat=[3,4,5,6].
        # pos=0 L=2: [3,4]→[4,5], pos=2 L=1: [5,0]→[6,0]
        np.testing.assert_array_equal(inputs, [[3, 4], [5, 0]])

    def test_empty_rows_skipped(self):
        pf = _make_parquet([[], [1, 2], [], [3, 4]])
        inputs, _, _ = _first_batch(
            iter_batches,
            pf,
            seq_len=2,
            batch_size=2,
            pad_index=0,
            stride=2,
            sep_index=SEP,
        )
        np.testing.assert_array_equal(inputs, [[1, 2], [4, 3]])

    def test_mask_consistency(self):
        pf = _make_parquet([[1, 2, 3, 4, 5], [6, 7], [8, 9, 10, 11, 12, 13, 14]])
        for _, masks, targets, slot_valid, _, _ in iter_batches(
            pf,
            seq_len=3,
            batch_size=3,
            pad_index=0,
            stride=2,
            sep_index=SEP,
        ):
            m = masks.numpy()
            t = targets.numpy()
            sv = slot_valid.numpy()
            for s in range(len(sv)):
                if sv[s]:
                    np.testing.assert_array_equal(m[s], t[s] == 0)
                else:
                    np.testing.assert_array_equal(m[s], np.zeros_like(m[s], dtype=bool))

    def test_token_count_consistency(self):
        pf = _make_parquet([[1, 2, 3, 4, 5], [6, 7], [8, 9, 10, 11, 12, 13, 14]])
        for _, _, targets, _, tokens, non_pad in iter_batches(
            pf,
            seq_len=3,
            batch_size=3,
            pad_index=0,
            stride=2,
            sep_index=SEP,
        ):
            assert tokens == 9  # batch_size * seq_len
            assert non_pad == (targets != 0).sum().item()

    def test_tracker_resume(self):
        pf = _make_parquet([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        _assert_tracker_resume(iter_batches, pf, sep_index=SEP)


# ═══════════════════════════════════════════════════════════════════
# iter_batches_indep  (end-to-end)
# ═══════════════════════════════════════════════════════════════════


class TestIterBatchesIndep:
    def test_single_row(self):
        pf = _make_parquet([[1, 2, 3, 4, 5]])
        inputs, _, targets = _first_batch(
            iter_batches_indep,
            pf,
            seq_len=3,
            batch_size=2,
            pad_index=0,
            stride=3,
        )
        np.testing.assert_array_equal(inputs, [[1, 2, 3], [4, 0, 0]])
        np.testing.assert_array_equal(targets, [[2, 3, 4], [5, 0, 0]])

    def test_custom_stride(self):
        pf = _make_parquet([[1, 2, 3, 4, 5]])
        inputs, _, _ = _first_batch(
            iter_batches_indep,
            pf,
            seq_len=2,
            batch_size=4,
            pad_index=0,
            stride=1,
        )
        np.testing.assert_array_equal(inputs, [[1, 2], [2, 3], [3, 4], [4, 0]])

    def test_sot_eot(self):
        pf = _make_parquet([[10, 20]])
        inputs, _, targets = _first_batch(
            iter_batches_indep,
            pf,
            seq_len=3,
            batch_size=1,
            pad_index=0,
            sot_index=1,
            eot_index=2,
        )
        np.testing.assert_array_equal(inputs[0], [1, 10, 20])
        np.testing.assert_array_equal(targets[0], [10, 20, 2])

    def test_short_row_skipped(self):
        pf = _make_parquet([[5]])
        batches = list(iter_batches_indep(pf, seq_len=2, batch_size=4, pad_index=0))
        assert len(batches) == 0

    def test_sot_eot_empty_row(self):
        pf = _make_parquet([[]])
        inputs, _, targets = _first_batch(
            iter_batches_indep,
            pf,
            seq_len=2,
            batch_size=1,
            pad_index=0,
            sot_index=7,
            eot_index=8,
        )
        np.testing.assert_array_equal(inputs[0], [7, 0])
        np.testing.assert_array_equal(targets[0], [8, 0])

    def test_start_index(self):
        pf = _make_parquet([[10, 20], [30, 40], [1, 2, 3], [4, 5, 6], [7, 8, 9]])
        inputs, _, _ = _first_batch(
            iter_batches_indep,
            pf,
            seq_len=2,
            batch_size=2,
            pad_index=0,
            tracker=TrainTracker(index=2, offset=0),
        )
        np.testing.assert_array_equal(inputs, [[1, 2], [4, 5]])

    def test_start_offset(self):
        pf = _make_parquet([[1, 2, 3, 4, 5, 6]])
        inputs, _, _ = _first_batch(
            iter_batches_indep,
            pf,
            seq_len=2,
            batch_size=2,
            pad_index=0,
            stride=2,
            tracker=TrainTracker(index=0, offset=2),
        )
        np.testing.assert_array_equal(inputs, [[3, 4], [5, 0]])

    def test_start_index_mid_row_group(self):
        rows = [[1, 2], [3, 4], [5, 6], [7, 8], [10, 20], [30, 40]]
        pf = _make_parquet_multi_rg(rows, row_group_size=3)
        inputs, _, _ = _first_batch(
            iter_batches_indep,
            pf,
            seq_len=2,
            batch_size=2,
            pad_index=0,
            tracker=TrainTracker(index=4, offset=0),
        )
        np.testing.assert_array_equal(inputs, [[10, 0], [30, 0]])

    def test_tracker_resume(self):
        pf = _make_parquet([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12, 13, 14]])
        _assert_tracker_resume(iter_batches_indep, pf)

    def test_partial_batch(self):
        pf = _make_parquet([[1, 2, 3, 4], [5, 6, 7, 8]])
        batches = list(
            iter_batches_indep(
                pf,
                seq_len=2,
                batch_size=3,
                pad_index=0,
                stride=2,
            )
        )
        assert len(batches) == 2
        # Batch 1 (full): row0 pos=0, row0 pos=2, row1 pos=0
        i1, m1, t1, _, _, np1 = batches[0]
        np.testing.assert_array_equal(i1, [[1, 2], [3, 0], [5, 6]])
        np.testing.assert_array_equal(t1, [[2, 3], [4, 0], [6, 7]])
        np.testing.assert_array_equal(
            m1, [[False, False], [False, True], [False, False]]
        )
        assert np1 == 5
        # Batch 2 (partial): row1 pos=2
        i2, m2, t2, _, _, np2 = batches[1]
        np.testing.assert_array_equal(i2, [[7, 0], [0, 0], [0, 0]])
        np.testing.assert_array_equal(t2, [[8, 0], [0, 0], [0, 0]])
        np.testing.assert_array_equal(
            m2, [[False, True], [False, False], [False, False]]
        )
        assert np2 == 1

    def test_buffer_staleness(self):
        """After buffer cycle, unwritten positions remain pad_index."""
        pf = _make_parquet([[1, 2, 3, 4, 5, 6, 7, 8], [10, 20, 30, 40, 50, 60, 70, 80]])
        it = iter_batches_indep(pf, seq_len=4, batch_size=2, pad_index=0, stride=3)

        b1 = next(it)
        np.testing.assert_array_equal(b1.input_ids[1], [4, 5, 6, 7])
        np.testing.assert_array_equal(b1.target_ids[1], [5, 6, 7, 8])

        b2 = next(it)
        np.testing.assert_array_equal(b2.input_ids[0], [7, 0, 0, 0])
        np.testing.assert_array_equal(b2.target_ids[0], [8, 0, 0, 0])
        np.testing.assert_array_equal(b2.input_ids[1], [10, 20, 30, 40])

        b3 = next(it)
        # Slot reused — must not have stale data from batch 1 slot 1.
        np.testing.assert_array_equal(b3.input_ids[0], [40, 50, 60, 70])
        np.testing.assert_array_equal(b3.target_ids[1], [80, 0, 0, 0])
        assert b3.tokens_non_pad == (b3.target_ids != 0).sum().item()

    def test_mask_and_token_invariants(self):
        """Every batch: mask == (targets == pad) and non_pad == count(targets != pad)."""
        pf = _make_parquet([[1, 2, 3, 4, 5], [6, 7], [8, 9, 10, 11, 12, 13, 14]])
        for _, masks, targets, _, tokens, non_pad in iter_batches_indep(
            pf,
            seq_len=3,
            batch_size=3,
            pad_index=0,
            stride=2,
        ):
            assert tokens == 9
            actual = (targets != 0).sum().item()
            assert non_pad == actual
            np.testing.assert_array_equal(masks.numpy(), targets.numpy() == 0)
