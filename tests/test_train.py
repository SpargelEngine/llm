"""Unit tests for spargel_llm.train."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from spargel_llm.train import _augment_row, iter_batches


def _make_parquet(rows: list[list[int]], dataset_id: str = "test") -> pq.ParquetFile:
    """Create an in-memory ParquetFile from token lists."""
    schema = pa.schema([pa.field("tokens", pa.list_(pa.uint32()))])
    schema = schema.with_metadata({"dataset_id": dataset_id})
    table = pa.table(
        {"tokens": [pa.array(r, type=pa.uint32()) for r in rows]}, schema=schema
    )
    buf = pa.BufferOutputStream()
    pq.write_table(table, buf)
    return pq.ParquetFile(pa.BufferReader(buf.getvalue()))


def _first_batch(*args, **kwargs):
    """Convenience: return (inputs, masks, targets) of the first batch."""
    inputs, masks, targets, _ = next(iter(iter_batches(*args, **kwargs)))
    return inputs, masks, targets


class TestAugmentRow:
    def test_no_boundary_tokens(self):
        values = np.array([10, 20, 30, 40], dtype=np.int32)
        result = _augment_row(values, 1, 3)  # [20, 30]
        np.testing.assert_array_equal(result, [20, 30])

    def test_sot_only(self):
        values = np.array([10, 20, 30], dtype=np.int32)
        result = _augment_row(values, 0, 3, sot_index=2)
        np.testing.assert_array_equal(result, [2, 10, 20, 30])

    def test_eot_only(self):
        values = np.array([10, 20, 30], dtype=np.int32)
        result = _augment_row(values, 0, 3, eot_index=3)
        np.testing.assert_array_equal(result, [10, 20, 30, 3])

    def test_sot_and_eot(self):
        values = np.array([10, 20, 30], dtype=np.int32)
        result = _augment_row(values, 0, 3, sot_index=2, eot_index=3)
        np.testing.assert_array_equal(result, [2, 10, 20, 30, 3])

    def test_empty_row_with_sot_eot(self):
        values = np.array([], dtype=np.int32)
        result = _augment_row(values, 0, 0, sot_index=2, eot_index=3)
        np.testing.assert_array_equal(result, [2, 3])

    def test_partial_slice(self):
        values = np.array([5, 10, 20, 30, 40], dtype=np.int32)
        result = _augment_row(values, 1, 4, sot_index=1, eot_index=2)
        np.testing.assert_array_equal(result, [1, 10, 20, 30, 2])


class TestIterBatches:
    def test_single_row_basic(self):
        """Row [1,2,3,4,5], seq_len=3, stride=3 → 2 windows."""
        pf = _make_parquet([[1, 2, 3, 4, 5]])
        inputs, masks, targets = _first_batch(pf, seq_len=3, batch_size=2, pad_index=0)
        # pos=0: L=3, input=[1,2,3]       target=[2,3,4]
        # pos=3: L=1, input=[4,0,0]       target=[5,0,0]
        np.testing.assert_array_equal(inputs, [[1, 2, 3], [4, 0, 0]])
        np.testing.assert_array_equal(targets, [[2, 3, 4], [5, 0, 0]])

    def test_partial_batch_discarded(self):
        """4 windows, batch_size=3 → 1 full batch, last window discarded."""
        pf = _make_parquet([[1, 2, 3, 4], [5, 6, 7, 8]])
        batches = list(iter_batches(pf, seq_len=2, batch_size=3, pad_index=0))
        assert len(batches) == 1
        inputs, _, _, _ = batches[0]
        # Row 0: pos=0 L=2 [1,2], pos=2 L=1 [3,0]
        # Row 1: pos=0 L=2 [5,6], pos=2 L=1 [7,0] ← discarded
        np.testing.assert_array_equal(inputs, [[1, 2], [3, 0], [5, 6]])

    def test_sot_eot(self):
        """[10,20] + SOT=1,EOT=2 → augmented=[1,10,20,2], one window L=3."""
        pf = _make_parquet([[10, 20]])
        inputs, masks, targets = _first_batch(
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
        """Row with 1 token (no SOT/EOT) → augmented len=1, skipped."""
        pf = _make_parquet([[5]])
        batches = list(iter_batches(pf, seq_len=2, batch_size=4, pad_index=0))
        assert len(batches) == 0

    def test_sot_eot_empty_row(self):
        """Empty row + SOT/EOT → [SOT, EOT], len=2, one window L=1."""
        pf = _make_parquet([[]])
        inputs, masks, targets = _first_batch(
            pf,
            seq_len=2,
            batch_size=1,
            pad_index=0,
            sot_index=7,
            eot_index=8,
        )
        np.testing.assert_array_equal(inputs[0], [7, 0])
        np.testing.assert_array_equal(targets[0], [8, 0])

    def test_start_offset(self):
        """start_offset=2 skips first 2 positions of the first row."""
        pf = _make_parquet([[1, 2, 3, 4, 5, 6]])
        inputs, masks, targets = _first_batch(
            pf,
            seq_len=2,
            batch_size=2,
            pad_index=0,
            start_offset=2,
        )
        # pos=2: L=2, input=[3,4]  target=[4,5]
        # pos=4: L=1, input=[5,0]  target=[6,0]
        np.testing.assert_array_equal(inputs, [[3, 4], [5, 0]])
        np.testing.assert_array_equal(targets, [[4, 5], [6, 0]])

    def test_custom_stride(self):
        """stride=1 produces overlapping windows."""
        pf = _make_parquet([[1, 2, 3, 4, 5]])
        inputs, _, _ = _first_batch(
            pf,
            seq_len=2,
            batch_size=4,
            pad_index=0,
            stride=1,
        )
        # pos=0: L=2 [1,2]  pos=1: L=2 [2,3]
        # pos=2: L=2 [3,4]  pos=3: L=1 [4,0]
        np.testing.assert_array_equal(inputs, [[1, 2], [2, 3], [3, 4], [4, 0]])

    def test_tracker(self):
        pf = _make_parquet([[1, 2, 3, 4], [5, 6, 7, 8]])
        tracker = {}
        list(
            iter_batches(
                pf,
                seq_len=2,
                batch_size=2,
                pad_index=0,
                tracker=tracker,
                stride=2,
            )
        )
        # Row 0, pos=0; Row 0, pos=2 → batch 0 (yielded)
        # Row 1, pos=0; Row 1, pos=2 → batch 1 (yielded)
        # Tracker ends at last yielded sample: row 1, offset 2
        assert tracker["index"] == 1
        assert tracker["offset"] == 2
