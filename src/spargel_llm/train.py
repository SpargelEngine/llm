import time
from dataclasses import dataclass
from typing import Callable, Iterator, Literal, Optional

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from pydantic import BaseModel, NonNegativeFloat, NonNegativeInt
from torch import Tensor
from torch.optim import Optimizer

from spargel_llm.model import Model

type Reduction = Literal["mean", "sum"]


class TrainInfo(BaseModel):
    time: NonNegativeFloat = 0
    token_count: NonNegativeInt = 0
    index: NonNegativeInt = 0
    offset: NonNegativeInt = 0
    dataset_id: str = ""


@torch.compile
def compute_loss_step(
    model: Model,
    input_ids: Tensor,
    mask: Tensor,
    target_ids: Tensor,
    *,
    pad_index: int,
    reduction: Reduction = "mean",
):
    logits: Tensor = model(input_ids, mask)  # (..., seq_len, vocab_size)
    flattened_logits = logits.flatten(0, -2)
    flattened_targets = target_ids.flatten(0, -1).to(torch.long)

    loss = nn.functional.cross_entropy(
        flattened_logits,
        flattened_targets,
        ignore_index=pad_index,
        reduction=reduction,
    )

    return loss


@torch.compile
def validation_metrics_step(
    model: Model,
    input_ids: Tensor,
    mask: Tensor,
    target_ids: Tensor,
    *,
    pad_index: int,
    reduction: Reduction = "mean",
):
    logits: Tensor = model(input_ids, mask)  # (..., seq_len, vocab_size)
    flattened_logits = logits.flatten(0, -2)
    flattened_targets = target_ids.flatten(0, -1).to(torch.long)

    loss = nn.functional.cross_entropy(
        flattened_logits,
        flattened_targets,
        ignore_index=pad_index,
        reduction=reduction,
    )

    valid_targets = flattened_targets != pad_index
    log_probs = torch.log_softmax(flattened_logits.float(), dim=-1)
    entropy_by_token = -(log_probs.exp() * log_probs).sum(dim=-1)
    valid_weights = valid_targets.to(entropy_by_token.dtype)
    entropy_sum = (entropy_by_token * valid_weights).sum()

    return loss, entropy_sum


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


def train(
    info: TrainInfo,
    model: Model,
    optimizer: Optimizer,
    batch_iterator: Iterator[tuple[Tensor, Tensor, Tensor, int, int]],
    pad_index: int,
    steps: int,
    *,
    device: str = "cpu",
    step_callback: Optional[Callable[[StepInfo], None]] = None,
    micro_batches: int = 1,
    step_offset: int = 0,
    use_bf16: bool = True,
) -> int:
    """Run up to *steps* training steps.

    Each step consists of *micro_batches* forward/backward passes whose
    gradients are accumulated before a single ``optimizer.step()`` call.
    The effective batch size is ``micro_batch_size x micro_batches``.

    *step_offset* is added to the step number reported to the callback,
    so callers that restart the data iterator (e.g. for looped datasets)
    can continue the step counter instead of resetting to 0.

    Returns the number of steps actually executed (may be less than *steps*
    if the data iterator is exhausted).
    """
    assert micro_batches >= 1

    torch_device = torch.device(device)
    device_type = torch_device.type

    for step in range(steps):
        t_start = time.perf_counter()

        # fetch micro batches
        cpu_batches: list[tuple[Tensor, Tensor, Tensor, int, int]] = []
        step_tokens = 0
        step_tokens_non_pad = 0
        for _ in range(micro_batches):
            try:
                inputs, masks, targets, tokens, tokens_non_pad = next(batch_iterator)
            except StopIteration:
                if not cpu_batches:
                    return step
                break
            step_tokens += tokens
            step_tokens_non_pad += tokens_non_pad
            cpu_batches.append((inputs, masks, targets, tokens, tokens_non_pad))

        step_loss = torch.tensor(0.0, device=torch_device, dtype=torch.float32)

        model.train()

        # reset gradients
        optimizer.zero_grad()

        # forward / backward
        for inputs, masks, targets, _, _ in cpu_batches:
            inputs = inputs.to(torch_device)
            masks = masks.to(torch_device)
            targets = targets.to(torch_device)

            with torch.autocast(
                device_type=device_type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                loss = compute_loss_step(
                    model=model,
                    input_ids=inputs,
                    mask=masks,
                    target_ids=targets,
                    pad_index=pad_index,
                    reduction="sum",
                )
            loss = loss / step_tokens_non_pad
            step_loss = step_loss + loss
            loss.backward()

        # update weights
        optimizer.step()

        if device == "cuda":
            torch.cuda.synchronize()
        t_end = time.perf_counter()

        step_time = t_end - t_start

        info.time += step_time
        info.token_count += step_tokens_non_pad

        if step_callback is not None:
            step_callback(
                StepInfo(
                    step=step_offset + step,
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
    eot_index: int | None = None,
    sot_index: int | None = None,
    *,
    start_index: int = 0,
    start_offset: int = 0,
    use_bf16: bool = True,
) -> tuple[float, float]:
    assert num_batches > 0

    torch_device = torch.device(device)
    device_type = torch_device.type

    def make_iterator():
        return iter_batches(
            dataset,
            seq_len,
            batch_size,
            pad_index,
            eot_index=eot_index,
            sot_index=sot_index,
            start_index=start_index,
            start_offset=start_offset,
        )

    iterator = make_iterator()

    total_loss = torch.tensor(0.0, device=torch_device, dtype=torch.float32)
    total_entropy = torch.tensor(0.0, device=torch_device, dtype=torch.float32)
    total_valid_tokens = 0

    model.eval()

    with torch.no_grad():
        for _ in range(num_batches):
            try:
                inputs, masks, targets, _, n_non_pad = next(iterator)
            except StopIteration:
                break

            inputs = inputs.to(torch_device)
            masks = masks.to(torch_device)
            targets = targets.to(torch_device)

            with torch.autocast(
                device_type=device_type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                loss, entropy_sum = validation_metrics_step(
                    model=model,
                    input_ids=inputs,
                    mask=masks,
                    target_ids=targets,
                    pad_index=pad_index,
                    reduction="sum",
                )

            total_loss = total_loss + loss
            total_entropy = total_entropy + entropy_sum
            total_valid_tokens += n_non_pad

    return (
        total_loss.item() / max(total_valid_tokens, 1),
        total_entropy.item() / max(total_valid_tokens, 1),
    )


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


def iter_batches(
    dataset: pq.ParquetFile,
    seq_len: int,
    batch_size: int,
    pad_index: int,
    stride: int | None = None,
    start_index: int = 0,
    start_offset: int = 0,
    tracker: dict | None = None,
    eot_index: int | None = None,
    sot_index: int | None = None,
) -> Iterator[tuple[Tensor, Tensor, Tensor, int, int]]:
    """Iterate through a pre-tokenized Parquet dataset and yield tensor batches.

    For each row, a window of ``seq_len + 1`` tokens slides with the given
    *stride* (default: ``seq_len``).  Each window produces one sample:
    ``(input_ids, mask, target_ids)``.  Short tails are padded with
    *pad_index*.

    Yields ``(inputs, masks, targets, tokens, tokens_non_pad)`` where
    *tokens* is the total number of elements in the batch tensors and
    *tokens_non_pad* is the number of valid (non-PAD) tokens — both
    computed during construction so the caller doesn't need a
    GPU-synchronising ``.item()`` call.

    Works directly on Arrow's underlying NumPy buffers to avoid the
    Python-object overhead of ``to_pylist()``.  Samples are accumulated
    into double-buffered batches so that yielded tensors are isolated from
    subsequent writes (including from partial batches that are never
    yielded).  Partial batches at the end are silently discarded.

    *start_index* skips the first N items of *dataset*.

    *start_offset* is used as the initial window position only for the
    first item (the one at *start_index*); all subsequent items start from
    the beginning.

    If *tracker* is a dict, it is updated before each sample with keys
    ``index`` (current dataset row) and ``offset`` (window position
    within that row).

    If *eot_index* is not ``None``, each row is treated as if it ends with
    an additional token of that value, so the effective row length becomes
    ``row_len + 1`` and the final window's target includes the EOT token.

    If *sot_index* is not ``None``, each row is treated as if it begins
    with an additional token of that value, so the effective row length
    becomes ``row_len + 1`` and the first window's input starts with SOT.
    """
    if stride is None:
        stride = seq_len

    # Double-buffer isolates yielded tensors from subsequent writes.
    # Even partial batches (which are never yielded) write to the buffer,
    # so without isolation a list(iter_batches(...)) would see aliased
    # memory in earlier tensors.
    bufs = [
        (
            np.full((batch_size, seq_len), pad_index, dtype=np.int32),
            np.ones((batch_size, seq_len), dtype=bool),
            np.full((batch_size, seq_len), pad_index, dtype=np.int32),
        )
        for _ in range(2)
    ]
    cur = 0
    batch_inputs, batch_masks, batch_targets = bufs[cur]
    sample_in_batch = 0
    batch_non_pad = 0

    first = True
    row_index = 0

    # Skip row groups that are entirely before start_index.
    start_rg = 0
    rg_start = 0
    while start_rg < dataset.metadata.num_row_groups:
        rg_rows = dataset.metadata.row_group(start_rg).num_rows
        if rg_start + rg_rows <= start_index:
            rg_start += rg_rows
            row_index += rg_rows
            start_rg += 1
            continue
        break

    for rg_idx in range(start_rg, dataset.metadata.num_row_groups):
        table = dataset.read_row_group(rg_idx)
        col = table.column("tokens").combine_chunks()
        offsets = col.offsets.to_numpy()
        values = col.values.to_numpy()

        n_rows = len(col)

        for i in range(n_rows):
            if row_index < start_index:
                row_index += 1
                continue

            row_start = int(offsets[i])
            row_end = int(offsets[i + 1])

            augmented = _augment_row(
                values,
                row_start,
                row_end,
                sot_index=sot_index,
                eot_index=eot_index,
            )
            aug_len = len(augmented)

            if aug_len <= 1:
                row_index += 1
                continue

            pos = start_offset if first else 0
            first = False

            while pos + 1 < aug_len:
                L = min(seq_len, aug_len - pos - 1)

                batch_inputs[sample_in_batch, :L] = augmented[pos : pos + L]
                batch_targets[sample_in_batch, :L] = augmented[pos + 1 : pos + 1 + L]
                batch_masks[sample_in_batch, :L] = False
                batch_non_pad += L

                if tracker is not None:
                    tracker["index"] = row_index
                    tracker["offset"] = pos

                sample_in_batch += 1

                if sample_in_batch == batch_size:
                    yield (
                        torch.from_numpy(batch_inputs),
                        torch.from_numpy(batch_masks),
                        torch.from_numpy(batch_targets),
                        batch_size * seq_len,
                        batch_non_pad,
                    )
                    cur = 1 - cur
                    # Reset all buffers: stale target data beyond the
                    # written L positions would be treated as valid by
                    # cross_entropy (which ignores only pad_index, not the
                    # attention mask), inflating the reported loss.
                    bufs[cur][0].fill(pad_index)
                    bufs[cur][1].fill(True)
                    bufs[cur][2].fill(pad_index)
                    batch_inputs, batch_masks, batch_targets = bufs[cur]
                    sample_in_batch = 0
                    batch_non_pad = 0

                pos += stride

            row_index += 1
