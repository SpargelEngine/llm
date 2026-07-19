import math
import time
from dataclasses import dataclass
from typing import Callable, Iterator, Literal, NamedTuple, Optional

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from pydantic import BaseModel, NonNegativeFloat, NonNegativeInt
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from spargel_llm.model import Model
from spargel_llm.parquet_utils import iter_row_groups

type Reduction = Literal["mean", "sum"]
type _FlatItem = tuple[np.ndarray, int, Callable[[int], None] | None]


class TrainInfo(BaseModel):
    time: NonNegativeFloat = 0
    token_count: NonNegativeInt = 0
    steps: NonNegativeInt = 0
    index: NonNegativeInt = 0
    offset: NonNegativeInt = 0
    dataset_id: str = ""


@dataclass
class TrainTracker:
    """Checkpoint token — records where the **next** sample will start.

    Mutated in-place by :func:`iter_batches` and :func:`iter_batches_indep`
    after each sample is added.  Save ``(index, offset)`` and pass them as
    *start_index* / *start_offset* to resume without sample duplication.
    """

    index: int
    offset: int


@torch.compile
def compute_loss_step(
    model: Model,
    input_ids: Tensor,
    mask: Tensor,
    target_ids: Tensor,
    slot_valid: Tensor,
    *,
    pad_index: int,
    reduction: Reduction = "mean",
):
    logits: Tensor = model(input_ids, mask)  # (batch, seq_len, vocab_size)

    # Zero out logits from invalid slots before cross_entropy.  Their
    # targets are all pad_index and would be ignored anyway; this is a
    # safety net so meaningless outputs from unfilled batch slots never
    # contribute to the loss.
    logits = torch.where(slot_valid[:, None, None], logits, torch.zeros_like(logits))

    flattened_logits = logits.flatten(0, -2).float()
    flattened_targets = target_ids.flatten(0, -1).to(torch.long)

    loss = nn.functional.cross_entropy(
        flattened_logits,
        flattened_targets,
        ignore_index=pad_index,
        reduction=reduction,
    )

    return loss


@torch.compile
def generate_step(model: Model, input: Tensor) -> Tensor:
    return model(input)


@dataclass
class StepInfo:
    step: int
    loss: float
    time: float
    tokens: int
    tokens_non_pad: int


class BatchData(NamedTuple):
    input_ids: Tensor
    mask: Tensor
    target_ids: Tensor
    slot_valid: Tensor  # (batch_size,) bool, False for unfilled slots in partial batch
    tokens: int
    tokens_non_pad: int


def train(
    info: TrainInfo,
    model: Model,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    batch_iterator: Iterator[BatchData],
    pad_index: int,
    steps: int,
    *,
    device: str = "cpu",
    step_callback: Optional[Callable[[StepInfo], None]] = None,
    micro_batches: int = 1,
    use_bf16: bool = True,
) -> int:
    """Run up to *steps* training steps.

    Each step consists of *micro_batches* forward/backward passes whose
    gradients are accumulated before a single ``optimizer.step()`` call.
    The effective batch size is ``micro_batch_size x micro_batches``.

    Returns the number of steps actually executed (may be less than *steps*
    if the data iterator is exhausted).
    """
    assert micro_batches >= 1

    torch_device = torch.device(device)
    device_type = torch_device.type

    for step in range(steps):
        t_start = time.perf_counter()

        # fetch micro batches
        cpu_batches: list[BatchData] = []
        step_tokens = 0
        step_tokens_non_pad = 0
        for _ in range(micro_batches):
            try:
                batch_data = next(batch_iterator)
            except StopIteration:
                if not cpu_batches:
                    return step
                break
            step_tokens += batch_data.tokens
            step_tokens_non_pad += batch_data.tokens_non_pad
            cpu_batches.append(batch_data)

        step_loss = torch.tensor(0.0, device=torch_device, dtype=torch.float32)

        model.train()

        # reset gradients
        optimizer.zero_grad()

        # forward / backward
        for batch_data in cpu_batches:
            inputs = batch_data.input_ids.to(torch_device)
            masks = batch_data.mask.to(torch_device)
            targets = batch_data.target_ids.to(torch_device)
            slot_valid = batch_data.slot_valid.to(torch_device)

            with torch.autocast(
                device_type=device_type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                loss = compute_loss_step(
                    model=model,
                    input_ids=inputs,
                    mask=masks,
                    target_ids=targets,
                    slot_valid=slot_valid,
                    pad_index=pad_index,
                    reduction="sum",
                )
            loss = loss / step_tokens_non_pad
            step_loss = step_loss + loss
            loss.backward()

        # update weights
        optimizer.step()

        # update learning rate
        lr_scheduler.step()

        if device == "cuda":
            torch.cuda.synchronize()
        t_end = time.perf_counter()

        step_time = t_end - t_start

        info.time += step_time
        info.token_count += step_tokens_non_pad
        info.steps += 1

        if step_callback is not None:
            step_callback(
                StepInfo(
                    step=info.steps,
                    loss=step_loss.item(),
                    time=step_time,
                    tokens=step_tokens,
                    tokens_non_pad=step_tokens_non_pad,
                )
            )

    return steps


def compute_validation_metrics(
    model: Model,
    dataset: pq.ParquetFile,
    seq_len: int,
    batch_size: int,
    pad_index: int,
    device: str,
    num_batches: int = 1,
    *,
    use_bf16: bool = True,
    indep: bool = False,
    sep_index: int | None = None,
    sot_index: int | None = None,
    eot_index: int | None = None,
) -> tuple[int, float, float]:
    assert num_batches > 0

    torch_device = torch.device(device)
    device_type = torch_device.type

    if indep:
        iterator = iter_batches_indep(
            dataset,
            seq_len,
            batch_size,
            pad_index,
            eot_index=eot_index,
            sot_index=sot_index,
        )
    else:
        assert sep_index is not None
        iterator = iter_batches(
            dataset,
            seq_len,
            batch_size,
            pad_index,
            sep_index,
        )

    total_loss = torch.tensor(0.0, device=torch_device, dtype=torch.float32)
    total_valid_tokens = 0
    batch_perplexities: list[float] = []

    model.eval()

    actual_batches = 0

    with torch.no_grad():
        for _ in range(num_batches):
            try:
                batch_data = next(iterator)
            except StopIteration:
                break

            inputs = batch_data.input_ids.to(torch_device)
            masks = batch_data.mask.to(torch_device)
            targets = batch_data.target_ids.to(torch_device)
            slot_valid = batch_data.slot_valid.to(torch_device)
            n_non_pad = batch_data.tokens_non_pad

            with torch.autocast(
                device_type=device_type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                loss = compute_loss_step(
                    model=model,
                    input_ids=inputs,
                    mask=masks,
                    target_ids=targets,
                    slot_valid=slot_valid,
                    pad_index=pad_index,
                    reduction="sum",
                )

            total_loss = total_loss + loss
            total_valid_tokens += n_non_pad
            per_batch_loss = loss.item() / n_non_pad
            batch_perplexities.append(math.exp(per_batch_loss))
            actual_batches += 1

    avg_loss = total_loss.item() / max(total_valid_tokens, 1)
    avg_perplexity = sum(batch_perplexities) / len(batch_perplexities)

    return actual_batches, avg_loss, avg_perplexity


def _augment_row(
    values: np.ndarray,
    row_start: int,
    row_end: int,
    sot_index: int | None = None,
    eot_index: int | None = None,
) -> np.ndarray:
    """Build a row with optional SOT/EOT boundary tokens.

    Returns an array of shape ``(row_len + extra_tokens,)`` where
    ``extra_tokens`` is 0, 1, or 2 depending on whether *sot_index*
    and/or *eot_index* are provided::

        [SOT?, values[row_start], ..., values[row_end-1], EOT?]

    When both *sot_index* and *eot_index* are ``None``, returns a
    zero-copy view into *values* instead of allocating a new array.
    """
    if sot_index is None and eot_index is None:
        return values[row_start:row_end]

    row_len = row_end - row_start
    has_sot = sot_index is not None
    has_eot = eot_index is not None
    extra = (1 if has_sot else 0) + (1 if has_eot else 0)
    augmented = np.empty(row_len + extra, dtype=np.int32)
    idx = 0
    if has_sot:
        augmented[0] = sot_index
        idx = 1
    augmented[idx : idx + row_len] = values[row_start:row_end]
    if has_eot:
        augmented[idx + row_len] = eot_index
    return augmented


def _resolve_start_row_group(
    dataset: pq.ParquetFile, start_index: int
) -> tuple[int, int]:
    """Return ``(first_rg, rows_skipped)`` for the given *start_index*."""
    for rg_idx, offset, rg_rows in iter_row_groups(dataset):
        if offset + rg_rows > start_index:
            return rg_idx, offset
    # start_index beyond all row groups → skip everything
    return dataset.metadata.num_row_groups, dataset.metadata.num_rows


def _read_row_group(
    dataset: pq.ParquetFile, rg_idx: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(offsets, values)`` numpy arrays for the tokens column."""
    table = dataset.read_row_group(rg_idx)
    col = table.column("tokens").combine_chunks()
    return col.offsets.to_numpy(), col.values.to_numpy()


def _yield_batches_from_arrays(
    arrays_iter: Iterator[_FlatItem],
    seq_len: int,
    batch_size: int,
    pad_index: int,
    *,
    stride: int | None = None,
) -> Iterator[BatchData]:
    """Slide windows over flat arrays and accumulate into batches.

    *arrays_iter* yields ``(flat, start_pos, on_sample)`` tuples:

    * *flat* — 1-D int32 token array.
    * *start_pos* — initial window position within *flat*.
    * *on_sample* — optional ``Callable[[int], None]`` called after each
      sample with the next position in *flat*.  Used by callers to update
      ``TrainTracker``.

    Yields :class:`BatchData` when the buffer fills.  A final partial
    batch is yielded after *arrays_iter* is exhausted.
    """
    if stride is None:
        stride = seq_len + 1

    bs, sl = batch_size, seq_len
    inputs = np.full((bs, sl), pad_index, dtype=np.int32)
    masks = np.ones((bs, sl), dtype=bool)
    targets = np.full((bs, sl), pad_index, dtype=np.int32)
    count = 0
    non_pad = 0

    def _emit(slot_valid: torch.Tensor) -> BatchData:
        return BatchData(
            torch.from_numpy(inputs.copy()),
            torch.from_numpy(masks.copy()),
            torch.from_numpy(targets.copy()),
            slot_valid,
            bs * sl,
            non_pad,
        )

    for flat, start_pos, on_sample in arrays_iter:
        max_pos = len(flat) - 2  # need ≥ 1 input + 1 target token
        pos = start_pos

        while pos <= max_pos:
            L = min(sl, len(flat) - pos - 1)

            inputs[count, :L] = flat[pos : pos + L]
            targets[count, :L] = flat[pos + 1 : pos + 1 + L]
            masks[count, :L] = False
            non_pad += L
            count += 1
            pos += stride

            if on_sample is not None:
                on_sample(pos)

            if count == bs:
                yield _emit(torch.ones(bs, dtype=torch.bool))
                inputs.fill(pad_index)
                masks.fill(True)
                targets.fill(pad_index)
                count = 0
                non_pad = 0

    if count > 0:
        if count < bs:
            masks[count:, :] = False
        slot_valid = torch.zeros(bs, dtype=torch.bool)
        slot_valid[:count] = True
        yield _emit(slot_valid)


def iter_batches(
    dataset: pq.ParquetFile,
    seq_len: int,
    batch_size: int,
    pad_index: int,
    sep_index: int,
    *,
    stride: int | None = None,
    tracker: TrainTracker | None = None,
) -> Iterator[BatchData]:
    """Yield batches with rows concatenated by *sep_index* within each row group.

    Rows in the same row group are joined with *sep_index* between non-last
    rows, then a sliding window produces samples.  Windows never cross row
    group boundaries.  Tails are padded with *pad_index*.

    If *tracker* is given, its ``index`` / ``offset`` are used as the
    start position and updated after each sample for checkpointing.
    """

    start_index = tracker.index if tracker is not None else 0
    start_offset = tracker.offset if tracker is not None else 0

    def _flat_iter():
        total_rows = dataset.metadata.num_rows
        first = True
        start_rg, row_idx = _resolve_start_row_group(dataset, start_index)

        for rg_idx in range(start_rg, dataset.metadata.num_row_groups):
            offsets, values = _read_row_group(dataset, rg_idx)
            n_col = len(offsets) - 1

            total_len = 0
            rows: list[tuple[int, int, bool, int]] = (
                []
            )  # (start, end, is_last, global_idx)

            for i in range(n_col):
                if row_idx < start_index:
                    row_idx += 1
                    continue

                rs = int(offsets[i])
                re = int(offsets[i + 1])

                if first and start_offset > 0:
                    rs = min(rs + start_offset, re)
                    first = False

                n = re - rs
                if n == 0:
                    row_idx += 1
                    continue

                is_last = row_idx >= total_rows - 1
                rows.append((rs, re, is_last, row_idx))
                total_len += n + (0 if is_last else 1)  # +1 for SEP
                row_idx += 1

            if not rows:
                continue

            flat = np.empty(total_len, dtype=np.int32)
            bp = np.empty(len(rows), dtype=np.int64)
            ri = np.empty(len(rows), dtype=np.int64)
            wp = 0

            for bi, (rs, re, is_last, gidx) in enumerate(rows):
                n = re - rs
                flat[wp : wp + n] = values[rs:re]
                bp[bi] = wp
                ri[bi] = gidx
                wp += n
                if not is_last:
                    flat[wp] = sep_index
                    wp += 1

            if tracker is not None:
                cursor = 0
                n_bp = len(bp)

                def on_sample(pos: int) -> None:
                    nonlocal cursor
                    while cursor + 1 < n_bp and bp[cursor + 1] <= pos:
                        cursor += 1
                    assert tracker is not None
                    tracker.index = int(ri[cursor])
                    tracker.offset = pos - int(bp[cursor])

                yield flat, 0, on_sample
            else:
                yield flat, 0, None

    yield from _yield_batches_from_arrays(
        _flat_iter(),
        seq_len,
        batch_size,
        pad_index,
        stride=stride,
    )


def iter_batches_indep(
    dataset: pq.ParquetFile,
    seq_len: int,
    batch_size: int,
    pad_index: int,
    *,
    stride: int | None = None,
    tracker: TrainTracker | None = None,
    sot_index: int | None = None,
    eot_index: int | None = None,
) -> Iterator[BatchData]:
    """Yield batches treating each row independently.

    For each row, optional SOT/EOT tokens are added via :func:`_augment_row`,
    then a sliding window produces samples.  Windows never cross row
    boundaries.  Rows shorter than 2 tokens are skipped.

    If *tracker* is given, its ``index`` / ``offset`` are used as the
    start position and updated after each sample for checkpointing.
    """

    start_index = tracker.index if tracker is not None else 0
    start_offset = tracker.offset if tracker is not None else 0

    def _flat_iter():
        first = True
        start_rg, row_idx = _resolve_start_row_group(dataset, start_index)

        for rg_idx in range(start_rg, dataset.metadata.num_row_groups):
            offsets, values = _read_row_group(dataset, rg_idx)

            for i in range(len(offsets) - 1):
                if row_idx < start_index:
                    row_idx += 1
                    continue

                src = _augment_row(
                    values,
                    int(offsets[i]),
                    int(offsets[i + 1]),
                    sot_index=sot_index,
                    eot_index=eot_index,
                )
                if len(src) <= 1:
                    row_idx += 1
                    continue

                start_pos = start_offset if first else 0
                first = False

                if tracker is not None:
                    rid = row_idx

                    def on_sample(pos: int) -> None:
                        assert tracker is not None
                        tracker.index = rid
                        tracker.offset = pos

                    yield src, start_pos, on_sample
                else:
                    yield src, start_pos, None

                row_idx += 1

    yield from _yield_batches_from_arrays(
        _flat_iter(),
        seq_len,
        batch_size,
        pad_index,
        stride=stride,
    )
