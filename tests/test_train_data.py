"""Unit tests for data preparation in spargel_llm.train."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from spargel_llm.train.basic import Item, compile_batch_data
from spargel_llm.train.pretrain import iterate_parquet_concat, iterate_sliding_window

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
        items = list(iterate_sliding_window(seq, seq_len=3, pad_id=PAD))

        assert len(items) == 2
        np.testing.assert_array_equal(items[0].input_ids, [1, 2, 3])
        np.testing.assert_array_equal(items[0].target_ids, [2, 3, 4])
        np.testing.assert_array_equal(items[0].mask, [True, True, True])
        assert items[0].tokens_valid == 3

        np.testing.assert_array_equal(items[1].input_ids, [5, 6, 0])
        np.testing.assert_array_equal(items[1].target_ids, [6, 7, 0])
        np.testing.assert_array_equal(items[1].mask, [True, True, False])
        assert items[1].tokens_valid == 2

    def test_overlapping_stride(self):
        seq = np.array([1, 2, 3, 4], dtype=np.int64)
        items = list(
            iterate_sliding_window(seq, seq_len=2, pad_id=PAD, stride=1)
        )

        assert len(items) == 3
        np.testing.assert_array_equal(items[0].input_ids, [1, 2])
        np.testing.assert_array_equal(items[0].target_ids, [2, 3])
        np.testing.assert_array_equal(items[1].input_ids, [2, 3])
        np.testing.assert_array_equal(items[1].target_ids, [3, 4])
        np.testing.assert_array_equal(items[2].input_ids, [3, 0])
        np.testing.assert_array_equal(items[2].target_ids, [4, 0])
        assert items[2].tokens_valid == 1

    def test_stride_zero_means_non_overlapping(self):
        seq = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        items = list(
            iterate_sliding_window(seq, seq_len=3, pad_id=PAD, stride=0)
        )
        assert len(items) == 1
        np.testing.assert_array_equal(items[0].input_ids, [1, 2, 3])

    def test_offset_skips_initial_positions(self):
        seq = np.array([10, 20, 30, 40, 50], dtype=np.int64)
        items = list(
            iterate_sliding_window(seq, seq_len=2, offset=2, pad_id=PAD, stride=2)
        )
        assert len(items) == 1
        np.testing.assert_array_equal(items[0].input_ids, [30, 40])

    def test_empty_sequence(self):
        seq = np.array([], dtype=np.int64)
        items = list(iterate_sliding_window(seq, seq_len=3, pad_id=PAD))
        assert len(items) == 0

    def test_single_token_yields_nothing(self):
        seq = np.array([42], dtype=np.int64)
        items = list(iterate_sliding_window(seq, seq_len=3, pad_id=PAD))
        assert len(items) == 0

    def test_two_tokens(self):
        seq = np.array([7, 8], dtype=np.int64)
        items = list(iterate_sliding_window(seq, seq_len=3, pad_id=PAD))

        assert len(items) == 1
        np.testing.assert_array_equal(items[0].input_ids, [7, 0, 0])
        np.testing.assert_array_equal(items[0].target_ids, [8, 0, 0])
        np.testing.assert_array_equal(items[0].mask, [True, False, False])
        assert items[0].tokens_valid == 1

    def test_partial_tail_padded(self):
        seq = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        items = list(
            iterate_sliding_window(seq, seq_len=3, pad_id=7, stride=3)
        )
        assert len(items) == 2
        # offset=3: L=1 → input=[4,7,7], target=[5,7,7]
        np.testing.assert_array_equal(items[1].input_ids, [4, 7, 7])
        np.testing.assert_array_equal(items[1].target_ids, [5, 7, 7])
        np.testing.assert_array_equal(items[1].mask, [True, False, False])
        assert items[1].tokens_valid == 1

    def test_input_target_offset_by_one(self):
        seq = np.array([5, 10, 15, 20, 25], dtype=np.int64)
        items = list(
            iterate_sliding_window(seq, seq_len=3, pad_id=PAD, stride=2)
        )
        np.testing.assert_array_equal(items[0].input_ids, [5, 10, 15])
        np.testing.assert_array_equal(items[0].target_ids, [10, 15, 20])

    def test_offset_beyond_sequence_yields_nothing(self):
        seq = np.array([1, 2, 3], dtype=np.int64)
        items = list(
            iterate_sliding_window(seq, seq_len=2, offset=2, pad_id=PAD)
        )
        assert len(items) == 0

    def test_long_sequence(self):
        seq = np.arange(100, dtype=np.int64)
        items = list(
            iterate_sliding_window(seq, seq_len=10, pad_id=PAD, stride=5)
        )
        assert len(items) == 20
        assert items[0].tokens_valid == 10
        assert items[-1].tokens_valid == 4

    def test_full_window_mask_always_true(self):
        seq = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        items = list(
            iterate_sliding_window(seq, seq_len=2, pad_id=PAD, stride=2)
        )
        for item in items:
            assert item.mask.all()


# ═══════════════════════════════════════════════════════════════════
# iterate_parquet_concat
# ═══════════════════════════════════════════════════════════════════


class TestIterateParquetConcat:
    def test_basic_concat(self):
        pf = _make_parquet([[1, 2], [10, 20, 30]])
        items = list(
            iterate_parquet_concat(pf, seq_len=3, pad_id=PAD, sep_id=SEP)
        )
        # flat = [1,2,4,10,20,30]
        assert len(items) == 2
        np.testing.assert_array_equal(items[0].input_ids, [1, 2, 4])
        np.testing.assert_array_equal(items[0].target_ids, [2, 4, 10])
        assert items[0].tokens_valid == 3
        np.testing.assert_array_equal(items[1].input_ids, [20, 0, 0])
        np.testing.assert_array_equal(items[1].target_ids, [30, 0, 0])
        assert items[1].tokens_valid == 1

    def test_sep_only_between_rows(self):
        pf = _make_parquet([[1], [5], [9]])
        items = list(
            iterate_parquet_concat(pf, seq_len=3, pad_id=PAD, sep_id=SEP)
        )
        # flat = [1,4,5,4,9], no trailing SEP
        np.testing.assert_array_equal(items[0].input_ids, [1, 4, 5])
        np.testing.assert_array_equal(items[0].target_ids, [4, 5, 4])

    def test_single_row_no_sep(self):
        pf = _make_parquet([[10, 20, 30]])
        items = list(
            iterate_parquet_concat(
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
            iterate_parquet_concat(
                pf, seq_len=2, start_rg_idx=1, pad_id=PAD, sep_id=SEP, stride=2
            )
        )
        np.testing.assert_array_equal(items[0].input_ids, [3, 0])

    def test_offset_first_rg_only(self):
        rows = [[1, 2, 3, 4], [5, 6, 7, 8]]
        pf = _make_parquet_multi_rg(rows, row_group_size=1)
        items = list(
            iterate_parquet_concat(
                pf, seq_len=2, start_rg_idx=0, offset=2,
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
            iterate_parquet_concat(
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
            iterate_parquet_concat(
                pf, seq_len=2, pad_id=PAD, sep_id=SEP, stride=2,
            )
        )
        assert len(items) == 4
        np.testing.assert_array_equal(items[0].input_ids, [1, 2])
        np.testing.assert_array_equal(items[2].input_ids, [5, 6])

    def test_all_items_have_same_seq_len(self):
        pf = _make_parquet([[1, 2, 3], [10, 20, 30, 40]])
        for item in iterate_parquet_concat(pf, seq_len=5, pad_id=PAD, sep_id=SEP):
            assert item.seq_len == 5

    def test_stride_forwarded(self):
        pf = _make_parquet([[1, 2, 3, 4, 5, 6]])
        items_default = list(
            iterate_parquet_concat(pf, seq_len=2, pad_id=PAD, sep_id=SEP)
        )
        items_stride1 = list(
            iterate_parquet_concat(pf, seq_len=2, pad_id=PAD, sep_id=SEP, stride=1)
        )
        assert len(items_stride1) > len(items_default)


# ═══════════════════════════════════════════════════════════════════
# compile_batch_data
# ═══════════════════════════════════════════════════════════════════


class TestCompileBatchData:
    def test_exact_single_batch(self):
        items = [
            Item(seq_len=3,
                 input_ids=np.array([1, 2, 3], dtype=np.int64),
                 target_ids=np.array([2, 3, 4], dtype=np.int64),
                 mask=np.ones(3, dtype=np.bool), tokens_valid=3),
            Item(seq_len=3,
                 input_ids=np.array([5, 6, 7], dtype=np.int64),
                 target_ids=np.array([6, 7, 8], dtype=np.int64),
                 mask=np.ones(3, dtype=np.bool), tokens_valid=3),
        ]
        bd = compile_batch_data(items, batch_size=2, seq_len=3, pad_id=PAD)

        assert bd.batches == 1
        assert bd.batch_size == 2
        np.testing.assert_array_equal(bd.input_ids[0], [[1, 2, 3], [5, 6, 7]])
        np.testing.assert_array_equal(bd.target_ids[0], [[2, 3, 4], [6, 7, 8]])
        np.testing.assert_array_equal(bd.slot_mask[0], [True, True])
        assert bd.tokens_valid == [6]

    def test_partial_last_batch(self):
        items = [
            Item(seq_len=2,
                 input_ids=np.array([1, 2], dtype=np.int64),
                 target_ids=np.array([2, 3], dtype=np.int64),
                 mask=np.ones(2, dtype=np.bool), tokens_valid=2),
            Item(seq_len=2,
                 input_ids=np.array([4, 5], dtype=np.int64),
                 target_ids=np.array([5, 6], dtype=np.int64),
                 mask=np.ones(2, dtype=np.bool), tokens_valid=2),
            Item(seq_len=2,
                 input_ids=np.array([7, 0], dtype=np.int64),
                 target_ids=np.array([8, 0], dtype=np.int64),
                 mask=np.array([True, False]), tokens_valid=1),
        ]
        bd = compile_batch_data(items, batch_size=2, seq_len=2, pad_id=PAD)

        assert bd.batches == 2
        np.testing.assert_array_equal(bd.input_ids[1], [[7, 0], [0, 0]])
        np.testing.assert_array_equal(bd.slot_mask[0], [True, True])
        np.testing.assert_array_equal(bd.slot_mask[1], [True, False])
        assert bd.tokens_valid == [4, 1]

    def test_multiple_full_batches(self):
        items = [
            Item(seq_len=2,
                 input_ids=np.array([i, i + 1], dtype=np.int64),
                 target_ids=np.array([i + 1, i + 2], dtype=np.int64),
                 mask=np.ones(2, dtype=np.bool), tokens_valid=2)
            for i in range(5)
        ]
        bd = compile_batch_data(items, batch_size=2, seq_len=2, pad_id=PAD)

        assert bd.batches == 3
        assert bd.tokens_valid == [4, 4, 2]
        np.testing.assert_array_equal(bd.input_ids[2], [[4, 5], [0, 0]])

    def test_slot_mask_marks_unfilled(self):
        items = [
            Item(seq_len=3,
                 input_ids=np.array([1, 2, 3], dtype=np.int64),
                 target_ids=np.array([2, 3, 4], dtype=np.int64),
                 mask=np.ones(3, dtype=np.bool), tokens_valid=3)
        ]
        bd = compile_batch_data(items, batch_size=3, seq_len=3, pad_id=PAD)

        np.testing.assert_array_equal(bd.slot_mask[0], [True, False, False])
        np.testing.assert_array_equal(bd.input_ids[0][1], [0, 0, 0])
        np.testing.assert_array_equal(bd.mask[0][1], [False, False, False])

    def test_tokens_valid_sums_across_slots(self):
        items = [
            Item(seq_len=3,
                 input_ids=np.array([1, 2, 3], dtype=np.int64),
                 target_ids=np.array([2, 3, 4], dtype=np.int64),
                 mask=np.ones(3, dtype=np.bool), tokens_valid=3),
            Item(seq_len=3,
                 input_ids=np.array([5, 0, 0], dtype=np.int64),
                 target_ids=np.array([6, 0, 0], dtype=np.int64),
                 mask=np.array([True, False, False]), tokens_valid=1),
        ]
        bd = compile_batch_data(items, batch_size=2, seq_len=3, pad_id=PAD)
        assert bd.tokens_valid == [4]

    def test_asserts_equal_seq_len(self):
        import pytest
        items = [
            Item(seq_len=3,
                 input_ids=np.array([1, 2, 3], dtype=np.int64),
                 target_ids=np.array([2, 3, 4], dtype=np.int64),
                 mask=np.ones(3, dtype=np.bool), tokens_valid=3),
            Item(seq_len=4,
                 input_ids=np.array([1, 2, 3, 4], dtype=np.int64),
                 target_ids=np.array([2, 3, 4, 5], dtype=np.int64),
                 mask=np.ones(4, dtype=np.bool), tokens_valid=4),
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
                 mask=np.ones(2, dtype=np.bool), tokens_valid=2)
        ]
        bd = compile_batch_data(items, batch_size=4, seq_len=2, pad_id=PAD)
        assert bd.batches == 1
        assert bd.slot_mask[0].tolist() == [True, False, False, False]
