"""Unit tests for data preparation in spargel_llm.train."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from spargel_llm.train.basic import Item, compile_batch_data
from spargel_llm.train.pretrain import ParquetConcatIterator, SlidingWindowIterator

PAD = 0
SEP = 4


# ═══════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════


def _make_parquet(rows: list[list[int]]) -> pq.ParquetFile:
    schema = pa.schema([pa.field("tokens", pa.list_(pa.uint32()))])
    table = pa.table(
        {"tokens": [pa.array(r, type=pa.uint32()) for r in rows]}, schema=schema
    )
    buf = pa.BufferOutputStream()
    pq.write_table(table, buf)
    return pq.ParquetFile(pa.BufferReader(buf.getvalue()))


def _make_parquet_multi_rg(
    rows: list[list[int]], row_group_size: int
) -> pq.ParquetFile:
    schema = pa.schema([pa.field("tokens", pa.list_(pa.uint32()))])
    table = pa.table(
        {"tokens": [pa.array(r, type=pa.uint32()) for r in rows]}, schema=schema
    )
    buf = pa.BufferOutputStream()
    pq.write_table(table, buf, row_group_size=row_group_size)
    return pq.ParquetFile(pa.BufferReader(buf.getvalue()))



# ═══════════════════════════════════════════════════════════════════
# iterate_sliding_window
# ═══════════════════════════════════════════════════════════════════


class TestIterateSlidingWindow:
    def test_non_overlapping(self):
        """stride defaults to seq_len+1 → non-overlapping windows."""
        seq = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
        items = list(SlidingWindowIterator(seq, seq_len=3, pad_id=PAD))

        assert len(items) == 2
        np.testing.assert_array_equal(items[0].input_ids, [1, 2, 3])
        np.testing.assert_array_equal(items[0].target_ids, [2, 3, 4])
        np.testing.assert_array_equal(items[0].mask, [True, True, True])
        assert items[0].num_non_pad == 3
        assert items[0].num_valid == 3

        np.testing.assert_array_equal(items[1].input_ids, [5, 6, 0])
        np.testing.assert_array_equal(items[1].target_ids, [6, 7, 0])
        np.testing.assert_array_equal(items[1].mask, [True, True, False])
        assert items[1].num_non_pad == 2
        assert items[1].num_valid == 2

    def test_overlapping_stride(self):
        seq = np.array([1, 2, 3, 4], dtype=np.int64)
        items = list(
            SlidingWindowIterator(seq, seq_len=2, pad_id=PAD, stride=1)
        )

        assert len(items) == 3
        np.testing.assert_array_equal(items[0].input_ids, [1, 2])
        np.testing.assert_array_equal(items[0].target_ids, [2, 3])
        np.testing.assert_array_equal(items[1].input_ids, [2, 3])
        np.testing.assert_array_equal(items[1].target_ids, [3, 4])
        np.testing.assert_array_equal(items[2].input_ids, [3, 0])
        np.testing.assert_array_equal(items[2].target_ids, [4, 0])
        assert items[2].num_non_pad == 1
        assert items[2].num_valid == 1

    def test_stride_zero_means_non_overlapping(self):
        seq = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        items = list(
            SlidingWindowIterator(seq, seq_len=3, pad_id=PAD, stride=0)
        )
        assert len(items) == 1
        np.testing.assert_array_equal(items[0].input_ids, [1, 2, 3])

    def test_offset_skips_initial_positions(self):
        seq = np.array([10, 20, 30, 40, 50], dtype=np.int64)
        items = list(
            SlidingWindowIterator(seq, seq_len=2, offset=2, pad_id=PAD, stride=2)
        )
        assert len(items) == 1
        np.testing.assert_array_equal(items[0].input_ids, [30, 40])

    def test_empty_sequence(self):
        seq = np.array([], dtype=np.int64)
        items = list(SlidingWindowIterator(seq, seq_len=3, pad_id=PAD))
        assert len(items) == 0

    def test_single_token_yields_nothing(self):
        seq = np.array([42], dtype=np.int64)
        items = list(SlidingWindowIterator(seq, seq_len=3, pad_id=PAD))
        assert len(items) == 0

    def test_two_tokens(self):
        seq = np.array([7, 8], dtype=np.int64)
        items = list(SlidingWindowIterator(seq, seq_len=3, pad_id=PAD))

        assert len(items) == 1
        np.testing.assert_array_equal(items[0].input_ids, [7, 0, 0])
        np.testing.assert_array_equal(items[0].target_ids, [8, 0, 0])
        np.testing.assert_array_equal(items[0].mask, [True, False, False])
        assert items[0].num_non_pad == 1
        assert items[0].num_valid == 1

    def test_partial_tail_padded(self):
        seq = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        items = list(
            SlidingWindowIterator(seq, seq_len=3, pad_id=7, stride=3)
        )
        assert len(items) == 2
        # offset=3: L=1 → input=[4,7,7], target=[5,7,7]
        np.testing.assert_array_equal(items[1].input_ids, [4, 7, 7])
        np.testing.assert_array_equal(items[1].target_ids, [5, 7, 7])
        np.testing.assert_array_equal(items[1].mask, [True, False, False])
        assert items[1].num_non_pad == 1
        assert items[1].num_valid == 1

    def test_input_target_offset_by_one(self):
        seq = np.array([5, 10, 15, 20, 25], dtype=np.int64)
        items = list(
            SlidingWindowIterator(seq, seq_len=3, pad_id=PAD, stride=2)
        )
        np.testing.assert_array_equal(items[0].input_ids, [5, 10, 15])
        np.testing.assert_array_equal(items[0].target_ids, [10, 15, 20])

    def test_offset_beyond_sequence_yields_nothing(self):
        seq = np.array([1, 2, 3], dtype=np.int64)
        items = list(
            SlidingWindowIterator(seq, seq_len=2, offset=2, pad_id=PAD)
        )
        assert len(items) == 0

    def test_long_sequence(self):
        seq = np.arange(100, dtype=np.int64)
        items = list(
            SlidingWindowIterator(seq, seq_len=10, pad_id=PAD, stride=5)
        )
        assert len(items) == 20
        assert items[0].num_non_pad == 10
        assert items[-1].num_non_pad == 4
        assert items[0].num_valid == 10
        assert items[-1].num_valid == 4

    def test_full_window_mask_always_true(self):
        seq = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        items = list(
            SlidingWindowIterator(seq, seq_len=2, pad_id=PAD, stride=2)
        )
        for item in items:
            assert item.mask.all()

    def test_offset_tracks_position(self):
        """offset reflects the next window start position during iteration."""
        seq = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
        it = SlidingWindowIterator(seq, seq_len=2, pad_id=PAD, stride=2)

        assert it.offset == 0
        next(it)
        assert it.offset == 2
        next(it)
        assert it.offset == 4
        next(it)  # last (partial) window
        assert it.offset == 6  # past end, next call raises StopIteration

    def test_resume_from_saved_offset(self):
        """Resuming with a saved offset yields the same remaining items."""
        seq = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
        it = SlidingWindowIterator(seq, seq_len=2, pad_id=PAD, stride=2)

        # Consume first item, then save position
        next(it)
        saved_offset = it.offset

        # Full iteration from scratch
        all_items = list(
            SlidingWindowIterator(seq, seq_len=2, pad_id=PAD, stride=2)
        )

        # Resume from saved offset
        resumed_items = list(
            SlidingWindowIterator(
                seq, seq_len=2, offset=saved_offset, pad_id=PAD, stride=2
            )
        )

        # resumed_items should match all_items[1:]
        assert len(resumed_items) == len(all_items) - 1
        for a, b in zip(resumed_items, all_items[1:]):
            np.testing.assert_array_equal(a.input_ids, b.input_ids)
            np.testing.assert_array_equal(a.target_ids, b.target_ids)

    def test_offset_before_iteration(self):
        """offset equals the initial value before any iteration."""
        it = SlidingWindowIterator(
            np.array([1, 2, 3], dtype=np.int64),
            seq_len=2, offset=5, pad_id=PAD,
        )
        assert it.offset == 5


# ═══════════════════════════════════════════════════════════════════
# iterate_parquet_concat
# ═══════════════════════════════════════════════════════════════════


class TestIterateParquetConcat:
    def test_basic_concat(self):
        pf = _make_parquet([[1, 2], [10, 20, 30]])
        items = list(
            ParquetConcatIterator(pf, seq_len=3, pad_id=PAD, sep_id=SEP)
        )
        # flat = [1,2,4,10,20,30]
        assert len(items) == 2
        np.testing.assert_array_equal(items[0].input_ids, [1, 2, 4])
        np.testing.assert_array_equal(items[0].target_ids, [2, 4, 10])
        assert items[0].num_non_pad == 3
        assert items[0].num_valid == 3
        np.testing.assert_array_equal(items[1].input_ids, [20, 0, 0])
        np.testing.assert_array_equal(items[1].target_ids, [30, 0, 0])
        assert items[1].num_non_pad == 1
        assert items[1].num_valid == 1

    def test_sep_only_between_rows(self):
        pf = _make_parquet([[1], [5], [9]])
        items = list(
            ParquetConcatIterator(pf, seq_len=3, pad_id=PAD, sep_id=SEP)
        )
        # flat = [1,4,5,4,9], no trailing SEP
        np.testing.assert_array_equal(items[0].input_ids, [1, 4, 5])
        np.testing.assert_array_equal(items[0].target_ids, [4, 5, 4])

    def test_single_row_no_sep(self):
        pf = _make_parquet([[10, 20, 30]])
        items = list(
            ParquetConcatIterator(
                pf, seq_len=2, pad_id=PAD, sep_id=SEP, stride=2
            )
        )
        assert len(items) == 1
        np.testing.assert_array_equal(items[0].input_ids, [10, 20])
        np.testing.assert_array_equal(items[0].target_ids, [20, 30])

    def test_start_rg_idx(self):
        rows = [[1, 2], [3, 4], [5, 6]]
        pf = _make_parquet_multi_rg(rows, row_group_size=1)
        items = list(
            ParquetConcatIterator(
                pf, seq_len=2, group_index=1, pad_id=PAD, sep_id=SEP, stride=2
            )
        )
        # RG1 ([3, 4]) and RG2 ([5, 6]) each produce one partial window
        assert len(items) == 2
        np.testing.assert_array_equal(items[0].input_ids, [3, 0])
        np.testing.assert_array_equal(items[0].target_ids, [4, 0])
        np.testing.assert_array_equal(items[1].input_ids, [5, 0])
        np.testing.assert_array_equal(items[1].target_ids, [6, 0])

    def test_offset_first_rg_only(self):
        rows = [[1, 2, 3, 4], [5, 6, 7, 8]]
        pf = _make_parquet_multi_rg(rows, row_group_size=1)
        items = list(
            ParquetConcatIterator(
                pf, seq_len=2, group_index=0, offset=2,
                pad_id=PAD, sep_id=SEP, stride=2,
            )
        )
        # RG0 offset=2: starts mid-sequence, RG1 offset resets to 0
        assert len(items) == 3
        np.testing.assert_array_equal(items[0].input_ids, [3, 0])
        np.testing.assert_array_equal(items[1].input_ids, [5, 6])
        np.testing.assert_array_equal(items[2].input_ids, [7, 0])

    def test_custom_column_name(self):
        schema = pa.schema([pa.field("data", pa.list_(pa.uint32()))])
        table = pa.table(
            {"data": [pa.array([7, 8, 9], type=pa.uint32())]}, schema=schema
        )
        buf = pa.BufferOutputStream()
        pq.write_table(table, buf)
        pf = pq.ParquetFile(pa.BufferReader(buf.getvalue()))

        items = list(
            ParquetConcatIterator(
                pf, seq_len=2, pad_id=PAD, sep_id=SEP,
                column_name="data", stride=2,
            )
        )
        assert len(items) == 1
        np.testing.assert_array_equal(items[0].input_ids, [7, 8])

    def test_multiple_row_groups(self):
        rows = [[1, 2], [3, 4], [5, 6], [7, 8]]
        pf = _make_parquet_multi_rg(rows, row_group_size=2)
        items = list(
            ParquetConcatIterator(
                pf, seq_len=2, pad_id=PAD, sep_id=SEP, stride=2,
            )
        )
        assert len(items) == 4
        np.testing.assert_array_equal(items[0].input_ids, [1, 2])
        np.testing.assert_array_equal(items[2].input_ids, [5, 6])

    def test_all_items_have_same_seq_len(self):
        pf = _make_parquet([[1, 2, 3], [10, 20, 30, 40]])
        for item in ParquetConcatIterator(pf, seq_len=5, pad_id=PAD, sep_id=SEP):
            assert item.seq_len == 5

    def test_stride_forwarded(self):
        pf = _make_parquet([[1, 2, 3, 4, 5, 6]])
        items_default = list(
            ParquetConcatIterator(pf, seq_len=2, pad_id=PAD, sep_id=SEP)
        )
        items_stride1 = list(
            ParquetConcatIterator(pf, seq_len=2, pad_id=PAD, sep_id=SEP, stride=1)
        )
        assert len(items_stride1) > len(items_default)

    def test_initial_position(self):
        """Before iteration, rg_idx and offset reflect constructor args."""
        pf = _make_parquet_multi_rg(
            [[1, 2], [3, 4], [5, 6]], row_group_size=1
        )
        it = ParquetConcatIterator(
            pf, seq_len=2, group_index=1, offset=3, pad_id=PAD, sep_id=SEP,
        )
        assert it.group_index == 1
        assert it.offset == 3

    def test_position_tracks_during_iteration(self):
        """rg_idx and offset update correctly as items are consumed.

        rg_idx is advanced lazily: only when we try to read past the end
        of the current row group's inner iterator, not when the last item
        is consumed.  offset always reflects the inner iterator's position
        (or the initial offset if no inner is active).
        """
        rows = [[1, 2, 3, 4], [5, 6, 7, 8]]
        pf = _make_parquet_multi_rg(rows, row_group_size=1)
        it = ParquetConcatIterator(
            pf, seq_len=2, pad_id=PAD, sep_id=SEP, stride=2,
        )

        assert it.group_index == 0
        assert it.offset == 0

        next(it)  # first window of RG0: [1,2]/[2,3]
        assert it.group_index == 0
        assert it.offset == 2  # advanced by stride

        next(it)  # last window of RG0: [3,0]/[4,0]
        # Still RG0 — the inner hasn't been asked for the next item yet
        assert it.group_index == 0
        assert it.offset == 4  # inner's position past the end

        # The *next* call triggers the row group transition
        next(it)  # first window of RG1: [5,6]/[6,7]
        assert it.group_index == 1
        assert it.offset == 2  # inner for RG1 advanced by stride

    def test_resume_mid_row_group(self):
        """Resuming from a saved (rg_idx, offset) within a row group works."""
        rows = [[1, 2, 3, 4, 5, 6], [10, 20, 30]]
        pf = _make_parquet_multi_rg(rows, row_group_size=1)
        it = ParquetConcatIterator(
            pf, seq_len=2, pad_id=PAD, sep_id=SEP, stride=2,
        )

        # Consume first item, then checkpoint
        next(it)
        saved_rg, saved_offset = it.group_index, it.offset

        # Consume remaining items from the original iterator
        remaining_original = list(it)

        # Resume from checkpoint
        resumed_it = ParquetConcatIterator(
            pf, seq_len=2, group_index=saved_rg, offset=saved_offset,
            pad_id=PAD, sep_id=SEP, stride=2,
        )
        remaining_resumed = list(resumed_it)

        assert len(remaining_resumed) == len(remaining_original)
        for a, b in zip(remaining_resumed, remaining_original):
            np.testing.assert_array_equal(a.input_ids, b.input_ids)
            np.testing.assert_array_equal(a.target_ids, b.target_ids)

    def test_resume_at_row_group_boundary(self):
        """Resuming at a row group boundary (rg_idx=1, offset=0) works.

        Starting at rg_idx=1 with offset=0 should produce the same
        items as a full iteration with RG0 items skipped.
        """
        rows = [[1, 2], [3, 4], [5, 6]]
        pf = _make_parquet_multi_rg(rows, row_group_size=1)

        # Full iteration from the start
        all_items = list(ParquetConcatIterator(
            pf, seq_len=2, pad_id=PAD, sep_id=SEP, stride=2,
        ))

        # RG0 has 1 row with 2 tokens, stride=2 → exactly 1 item
        # So rg_idx=1, offset=0 should skip the first item
        resumed_items = list(ParquetConcatIterator(
            pf, seq_len=2, group_index=1, offset=0,
            pad_id=PAD, sep_id=SEP, stride=2,
        ))

        assert len(resumed_items) == len(all_items) - 1
        for a, b in zip(resumed_items, all_items[1:]):
            np.testing.assert_array_equal(a.input_ids, b.input_ids)
            np.testing.assert_array_equal(a.target_ids, b.target_ids)

    # ── loop tests ────────────────────────────────────────────────

    def test_loop_wraps_around(self):
        """loop=True wraps back to the first row group after exhaustion."""
        rows = [[1, 2, 3], [10, 20, 30]]
        pf = _make_parquet_multi_rg(rows, row_group_size=1)
        it = ParquetConcatIterator(
            pf, seq_len=2, pad_id=PAD, sep_id=SEP, stride=2, loop=True,
        )

        # Each RG has seq_len=3, stride=2 → 1 item per RG → 2 items per cycle
        cycle1 = [next(it) for _ in range(2)]
        cycle2 = [next(it) for _ in range(2)]

        assert len(cycle1) == 2
        assert len(cycle2) == 2
        for a, b in zip(cycle1, cycle2):
            np.testing.assert_array_equal(a.input_ids, b.input_ids)
            np.testing.assert_array_equal(a.target_ids, b.target_ids)

    def test_loop_resets_offset(self):
        """After looping, group_index wraps to 0; inner offset tracks normally."""
        rows = [[1, 2, 3, 4], [5, 6, 7, 8]]
        pf = _make_parquet_multi_rg(rows, row_group_size=1)
        it = ParquetConcatIterator(
            pf, seq_len=2, pad_id=PAD, sep_id=SEP, stride=2, loop=True,
        )

        # Each RG: len=4, seq_len=2, stride=2 → 2 items per RG → 4 per cycle.
        # After 4 items, RG1's inner is still active but not yet exhausted.
        # One more next() exhausts RG1, wraps to RG0, and yields RG0's first
        # window, advancing inner offset to 2.
        for _ in range(4):
            next(it)

        # The 5th next() triggers the wrap AND yields the first item of the new cycle
        next(it)
        assert it.group_index == 0
        assert it.offset == 2  # inner advanced by one window after wrap

    def test_loop_default_off(self):
        """Default loop=False raises StopIteration after exhaustion."""
        pf = _make_parquet([[1, 2, 3]])
        it = ParquetConcatIterator(
            pf, seq_len=2, pad_id=PAD, sep_id=SEP, stride=2,
        )
        list(it)  # exhaust
        with __import__("pytest").raises(StopIteration):
            next(it)

    def test_loop_with_single_row_group(self):
        """loop=True works even when there is only one row group."""
        pf = _make_parquet([[1, 2, 3, 4]])
        it = ParquetConcatIterator(
            pf, seq_len=2, pad_id=PAD, sep_id=SEP, stride=2, loop=True,
        )

        items1 = [next(it) for _ in range(2)]   # cycle 1
        items2 = [next(it) for _ in range(2)]   # cycle 2

        assert len(items1) == 2
        assert len(items2) == 2
        # Each cycle should produce the same windows
        np.testing.assert_array_equal(items1[0].input_ids, items2[0].input_ids)
        np.testing.assert_array_equal(items1[1].input_ids, items2[1].input_ids)

    def test_loop_position_tracks_correctly(self):
        """group_index and offset advance correctly across loop wraps."""
        rows = [[1, 2], [3, 4]]
        pf = _make_parquet_multi_rg(rows, row_group_size=1)
        it = ParquetConcatIterator(
            pf, seq_len=2, pad_id=PAD, sep_id=SEP, stride=2, loop=True,
        )

        # RG0, one partial item: [1,0]/[2,0]
        next(it)
        assert it.group_index == 0
        assert it.offset == 2  # past end of RG0

        # Triggers transition to RG1
        next(it)  # RG1: [3,0]/[4,0]
        assert it.group_index == 1
        assert it.offset == 2  # past end of RG1

        # Triggers wrap back to RG0
        next(it)  # RG0 again: [1,0]/[2,0]
        assert it.group_index == 0
        assert it.offset == 2


# ═══════════════════════════════════════════════════════════════════
# compile_batch_data
# ═══════════════════════════════════════════════════════════════════


class TestCompileBatchData:
    def test_exact_single_batch(self):
        items = [
            Item(seq_len=3,
                 input_ids=np.array([1, 2, 3], dtype=np.int64),
                 target_ids=np.array([2, 3, 4], dtype=np.int64),
                 mask=np.ones(3, dtype=np.bool), num_non_pad=3, num_valid=3),
            Item(seq_len=3,
                 input_ids=np.array([5, 6, 7], dtype=np.int64),
                 target_ids=np.array([6, 7, 8], dtype=np.int64),
                 mask=np.ones(3, dtype=np.bool), num_non_pad=3, num_valid=3),
        ]
        bd = compile_batch_data(items, batch_size=2, seq_len=3, pad_id=PAD)

        assert bd.batches == 1
        assert bd.batch_size == 2
        np.testing.assert_array_equal(bd.input_ids[0], [[1, 2, 3], [5, 6, 7]])
        np.testing.assert_array_equal(bd.target_ids[0], [[2, 3, 4], [6, 7, 8]])
        np.testing.assert_array_equal(bd.slot_mask[0], [True, True])
        assert bd.num_non_pad == [6]
        assert bd.num_valid == [6]

    def test_partial_last_batch(self):
        items = [
            Item(seq_len=2,
                 input_ids=np.array([1, 2], dtype=np.int64),
                 target_ids=np.array([2, 3], dtype=np.int64),
                 mask=np.ones(2, dtype=np.bool), num_non_pad=2, num_valid=2),
            Item(seq_len=2,
                 input_ids=np.array([4, 5], dtype=np.int64),
                 target_ids=np.array([5, 6], dtype=np.int64),
                 mask=np.ones(2, dtype=np.bool), num_non_pad=2, num_valid=2),
            Item(seq_len=2,
                 input_ids=np.array([7, 0], dtype=np.int64),
                 target_ids=np.array([8, 0], dtype=np.int64),
                 mask=np.array([True, False]), num_non_pad=1, num_valid=1),
        ]
        bd = compile_batch_data(items, batch_size=2, seq_len=2, pad_id=PAD)

        assert bd.batches == 2
        np.testing.assert_array_equal(bd.input_ids[1], [[7, 0], [0, 0]])
        np.testing.assert_array_equal(bd.slot_mask[0], [True, True])
        np.testing.assert_array_equal(bd.slot_mask[1], [True, False])
        assert bd.num_non_pad == [4, 1]
        assert bd.num_valid == [4, 1]

    def test_multiple_full_batches(self):
        items = [
            Item(seq_len=2,
                 input_ids=np.array([i, i + 1], dtype=np.int64),
                 target_ids=np.array([i + 1, i + 2], dtype=np.int64),
                 mask=np.ones(2, dtype=np.bool), num_non_pad=2, num_valid=2)
            for i in range(5)
        ]
        bd = compile_batch_data(items, batch_size=2, seq_len=2, pad_id=PAD)

        assert bd.batches == 3
        assert bd.num_non_pad == [4, 4, 2]
        assert bd.num_valid == [4, 4, 2]
        np.testing.assert_array_equal(bd.input_ids[2], [[4, 5], [0, 0]])

    def test_slot_mask_marks_unfilled(self):
        items = [
            Item(seq_len=3,
                 input_ids=np.array([1, 2, 3], dtype=np.int64),
                 target_ids=np.array([2, 3, 4], dtype=np.int64),
                 mask=np.ones(3, dtype=np.bool), num_non_pad=3, num_valid=3)
        ]
        bd = compile_batch_data(items, batch_size=3, seq_len=3, pad_id=PAD)

        np.testing.assert_array_equal(bd.slot_mask[0], [True, False, False])
        np.testing.assert_array_equal(bd.input_ids[0][1], [0, 0, 0])
        np.testing.assert_array_equal(bd.mask[0][1], [True, True, True])
        # unfilled slots contribute nothing to the totals
        assert bd.num_non_pad == [3]
        assert bd.num_valid == [3]

    def test_num_non_pad_sums_across_slots(self):
        """num_non_pad and num_valid are aggregated over the slots of a batch."""
        items = [
            Item(seq_len=3,
                 input_ids=np.array([1, 2, 3], dtype=np.int64),
                 target_ids=np.array([2, 3, 4], dtype=np.int64),
                 mask=np.ones(3, dtype=np.bool), num_non_pad=3, num_valid=3),
            Item(seq_len=3,
                 input_ids=np.array([5, 0, 0], dtype=np.int64),
                 target_ids=np.array([6, 0, 0], dtype=np.int64),
                 mask=np.array([True, False, False]), num_non_pad=1, num_valid=1),
        ]
        bd = compile_batch_data(items, batch_size=2, seq_len=3, pad_id=PAD)
        assert bd.num_non_pad == [4]
        assert bd.num_valid == [4]

    def test_num_valid_and_num_non_pad_are_independent(self):
        """num_valid counts visible positions even when targets are pad,
        so the two fields can differ."""
        items = [
            Item(seq_len=3,
                 input_ids=np.array([1, 2, 3], dtype=np.int64),
                 target_ids=np.array([2, 3, 4], dtype=np.int64),
                 mask=np.ones(3, dtype=np.bool), num_non_pad=3, num_valid=3),
            Item(seq_len=3,
                 input_ids=np.array([5, 0, 0], dtype=np.int64),
                 target_ids=np.array([6, 0, 0], dtype=np.int64),
                 mask=np.ones(3, dtype=np.bool), num_non_pad=1, num_valid=3),
        ]
        bd = compile_batch_data(items, batch_size=2, seq_len=3, pad_id=PAD)
        assert bd.num_non_pad == [4]
        assert bd.num_valid == [6]

    def test_asserts_equal_seq_len(self):
        import pytest
        items = [
            Item(seq_len=3,
                 input_ids=np.array([1, 2, 3], dtype=np.int64),
                 target_ids=np.array([2, 3, 4], dtype=np.int64),
                 mask=np.ones(3, dtype=np.bool), num_non_pad=3, num_valid=3),
            Item(seq_len=4,
                 input_ids=np.array([1, 2, 3, 4], dtype=np.int64),
                 target_ids=np.array([2, 3, 4, 5], dtype=np.int64),
                 mask=np.ones(4, dtype=np.bool), num_non_pad=4, num_valid=4),
        ]
        with pytest.raises(AssertionError):
            compile_batch_data(items, batch_size=2, seq_len=3, pad_id=PAD)

    def test_asserts_non_empty(self):
        import pytest
        with pytest.raises(AssertionError):
            compile_batch_data([], batch_size=2, seq_len=3, pad_id=PAD)

    def test_batch_size_larger_than_items(self):
        items = [
            Item(seq_len=2,
                 input_ids=np.array([1, 2], dtype=np.int64),
                 target_ids=np.array([2, 3], dtype=np.int64),
                 mask=np.ones(2, dtype=np.bool), num_non_pad=2, num_valid=2)
        ]
        bd = compile_batch_data(items, batch_size=4, seq_len=2, pad_id=PAD)
        assert bd.batches == 1
        assert bd.slot_mask[0].tolist() == [True, False, False, False]
