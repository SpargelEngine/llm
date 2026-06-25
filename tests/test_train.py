"""Unit tests for spargel_llm.train."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

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
    inputs, masks, targets, _, _ = next(iter(iter_batches(*args, **kwargs)))
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
        inputs, _, _, _, _ = batches[0]
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

    def test_buffer_staleness_after_cycle(self):
        """Buffer unwritten positions must be pad_index after a full cycle.

        buffer 0 (batch 1) → buffer 1 (batch 2) → buffer 0 (batch 3).
        If only the mask is reset, batch 3's unwritten positions carry
        stale real token IDs from batch 1, which cross_entropy would
        treat as valid targets (it ignores only ``pad_index``, not the
        attention mask).

        IMPORTANT: ``torch.from_numpy`` tensors share memory with the
        numpy buffer.  Do **not** collect batches with ``list()`` (3+
        batches cause the double-buffer to cycle and overwrite earlier
        tensors).  Verify each batch immediately after yielding.
        """
        # batch_size=2, seq_len=4, stride=3, pad_index=0
        #
        # Row 0: 8 tokens → aug_len=8
        #   pos=0: L=min(4,7)=4 → sample 0 (buffer 0)
        #   pos=3: L=min(4,4)=4 → sample 1, batch 1 YIELD (buffer 0)
        #   pos=6: L=min(4,1)=1 → sample 0 (buffer 1)
        #
        # Row 1: 8 tokens → aug_len=8
        #   pos=0: L=min(4,7)=4 → sample 1, batch 2 YIELD (buffer 1)
        #   pos=3: L=min(4,4)=4 → sample 0 (buffer 0 — REUSED)
        #   pos=6: L=min(4,1)=1 → sample 1, batch 3 YIELD (buffer 0)
        #
        # Batch 3 sample 1 writes only 1 position (L=1).  Without the
        # fix targets[1, 1:] = [6, 7, 8] (stale from batch 1 sample 1).
        pf = _make_parquet(
            [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [10, 20, 30, 40, 50, 60, 70, 80],
            ]
        )

        it = iter_batches(pf, seq_len=4, batch_size=2, pad_index=0, stride=3)

        # Batch 1 (buffer 0): Row 0 pos=0 L=4, pos=3 L=4
        inputs_1, _, targets_1, _, n_non_pad_1 = next(it)
        np.testing.assert_array_equal(inputs_1[0], [1, 2, 3, 4])
        np.testing.assert_array_equal(targets_1[0], [2, 3, 4, 5])
        np.testing.assert_array_equal(inputs_1[1], [4, 5, 6, 7])
        np.testing.assert_array_equal(targets_1[1], [5, 6, 7, 8])
        assert n_non_pad_1 == 8

        # Batch 2 (buffer 1): Row 0 pos=6 L=1, Row 1 pos=0 L=4
        inputs_2, _, targets_2, _, n_non_pad_2 = next(it)
        np.testing.assert_array_equal(inputs_2[0], [7, 0, 0, 0])
        np.testing.assert_array_equal(targets_2[0], [8, 0, 0, 0])
        np.testing.assert_array_equal(inputs_2[1], [10, 20, 30, 40])
        np.testing.assert_array_equal(targets_2[1], [20, 30, 40, 50])
        assert n_non_pad_2 == 5

        # Batch 3 (buffer 0 reused): Row 1 pos=3 L=4, pos=6 L=1
        inputs_3, masks_3, targets_3, _, n_non_pad_3 = next(it)
        np.testing.assert_array_equal(inputs_3[0], [40, 50, 60, 70])
        np.testing.assert_array_equal(targets_3[0], [50, 60, 70, 80])
        np.testing.assert_array_equal(inputs_3[1], [70, 0, 0, 0])
        np.testing.assert_array_equal(targets_3[1], [80, 0, 0, 0])
        assert n_non_pad_3 == 5

        # non-pad count must match the actual count in targets
        non_pad_in_targets = (targets_3 != 0).sum().item()
        assert n_non_pad_3 == non_pad_in_targets, (
            f"n_non_pad ({n_non_pad_3}) != actual non-pad in targets "
            f"({non_pad_in_targets}) — stale buffer data leaked"
        )

        # Masks must be False exactly where data was written
        np.testing.assert_array_equal(masks_3[0], [False, False, False, False])
        np.testing.assert_array_equal(masks_3[1], [False, True, True, True])

        # No more batches expected
        with pytest.raises(StopIteration):
            next(it)

    def test_all_batches_have_consistent_token_counts(self):
        """Every yielded batch must have n_non_pad == count(targets != pad)."""
        pf = _make_parquet(
            [
                [1, 2, 3, 4, 5],
                [6, 7],
                [8, 9, 10, 11, 12, 13, 14],
            ]
        )
        for inputs, masks, targets, tokens, n_non_pad in iter_batches(
            pf, seq_len=3, batch_size=3, pad_index=0, stride=2
        ):
            assert tokens == inputs.numel(), "tokens must equal total elements"
            actual_non_pad = (targets != 0).sum().item()
            assert n_non_pad == actual_non_pad, (
                f"n_non_pad={n_non_pad} != actual={actual_non_pad}; "
                f"targets=\n{targets}"
            )

    def test_all_batches_have_consistent_masks(self):
        """Mask must be False exactly where targets != pad_index."""
        pf = _make_parquet(
            [
                [1, 2, 3, 4, 5],
                [6, 7],
                [8, 9, 10, 11, 12, 13, 14],
            ]
        )
        for _, masks, targets, _, _ in iter_batches(
            pf, seq_len=3, batch_size=3, pad_index=0, stride=2
        ):
            np.testing.assert_array_equal(
                masks.numpy(),
                targets.numpy() == 0,
                err_msg="mask must be True (ignored) where target is pad_index",
            )

    def test_many_buffer_cycles(self):
        """Exercise many buffer cycles to flush out any corner cases.

        Verifies that every yielded batch has consistent n_non_pad and
        mask-vs-targets values.  Processes batches as yielded — do not
        ``list()``-collect since tensors share memory with the buffer.
        """
        rows = [[i % 100 + 1] * 20 for i in range(100)]
        pf = _make_parquet(rows)
        count = 0
        for inputs, masks, targets, tokens, n_non_pad in iter_batches(
            pf, seq_len=4, batch_size=4, pad_index=0, stride=3
        ):
            assert tokens == inputs.numel()
            actual = (targets != 0).sum().item()
            assert n_non_pad == actual, f"n_non_pad={n_non_pad} != actual={actual}"
            np.testing.assert_array_equal(
                masks.numpy(),
                targets.numpy() == 0,
            )
            count += 1
        assert count > 10, f"expected >10 batches, got {count}"
