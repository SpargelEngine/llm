# Basic Training
# Can be used for pre-training and SFT.

from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer

from spargel_llm.model import Model
from spargel_llm.typing import CrossEntropyReduction


class Item(NamedTuple):
    """One piece of training data."""

    seq_len: int

    input_ids: np.ndarray  # (seq_len,) int64
    target_ids: np.ndarray  # (seq_len,) int64
    mask: np.ndarray  # (seq_len,) bool (True=valid)

    # number of valid (non-PAD) tokens in the target
    # which will participate in the computation of cross entropy
    tokens_valid: int


class BatchData(NamedTuple):
    """
    Data used for one forward/backward step (may compose of micro-batches).
    """

    batches: int  # number of batches
    batch_size: int
    seq_len: int

    input_ids: Tensor  # (batches, batch_size, seq_len) long
    target_ids: Tensor  # (batches, batch_size, seq_len) long
    mask: Tensor  # (batches, batch_size, seq_len) bool (True=valid)
    slot_mask: Tensor  # (batches, batch_size) bool (True=valid)

    # (batches) number of valid (non-PAD) tokens per batch in the target
    # which will participate in the computation of cross entropy
    tokens_valid: list[int]

    def to(self, device: torch.device | str | int):
        return BatchData(
            batches=self.batches,
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            input_ids=self.input_ids.to(device),
            target_ids=self.target_ids.to(device),
            mask=self.mask.to(device),
            slot_mask=self.slot_mask.to(device),
            tokens_valid=self.tokens_valid,
        )


def compile_batch_data(
    items: list[Item], batch_size: int, seq_len: int, *, pad_id: int
) -> BatchData:
    def ceil_div(a: int, b: int) -> int:
        return -(-a // b)

    assert len(items) > 0
    for item in items:
        assert item.seq_len == seq_len

    batches = ceil_div(len(items), batch_size)

    input_ids = torch.full((batches, batch_size, seq_len), pad_id, dtype=torch.long)
    target_ids = torch.full((batches, batch_size, seq_len), pad_id, dtype=torch.long)
    mask = torch.ones((batches, batch_size, seq_len), dtype=torch.bool)  # default True so unfilled slots don't cause all-masked NaN in attention
    slot_mask = torch.zeros((batches, batch_size), dtype=torch.bool)

    tokens_valid = [0] * batches

    for batch in range(batches):
        batch_tokens_valid = 0

        for offset in range(batch_size):
            i = batch * batch_size + offset
            if i >= len(items):  # last partial batch
                break
            item = items[i]

            input_ids[batch][offset] = torch.from_numpy(item.input_ids)
            target_ids[batch][offset] = torch.from_numpy(item.target_ids)
            mask[batch][offset] = torch.from_numpy(item.mask)
            slot_mask[batch][offset] = True

            batch_tokens_valid += item.tokens_valid

        tokens_valid[batch] = batch_tokens_valid

    return BatchData(
        batches=batches,
        batch_size=batch_size,
        seq_len=seq_len,
        input_ids=input_ids,
        target_ids=target_ids,
        mask=mask,
        slot_mask=slot_mask,
        tokens_valid=tokens_valid,
    )


@torch.compile
def compute_loss_step(
    model: Model,
    input_ids: Tensor,
    target_ids: Tensor,
    mask: Tensor,
    slot_mask: Tensor,
    *,
    pad_id: int,
    reduction: CrossEntropyReduction = "mean",
) -> Tensor:
    """
    Args:
        model: model instance
        target_ids: (..., seq_len) long
        input_ids:  (..., seq_len) long
        mask:       (..., seq_len) bool (True=valid)
        slot_mask:  (...) bool (True=valid)
        pad_id: token id for PAD

    Returns:
        loss
    """

    logits: Tensor = model(input_ids, ~mask)
    # (..., seq_len, vocab_size)

    # Zero out invalid slots, avoid NaN.
    logits = torch.where(slot_mask[:, None, None], logits, torch.zeros_like(logits))

    # flatten batch dimension to one
    flattened_logits = logits.flatten(0, -2)
    flattened_target = target_ids.flatten(0, -1)

    loss = F.cross_entropy(
        flattened_logits, flattened_target, ignore_index=pad_id, reduction=reduction
    )

    return loss


def train_step(
    model: Model,
    optimizer: Optimizer,
    batch_data: BatchData,
    *,
    pad_id: int,
    use_bf16: bool = True,
) -> Tensor:
    """
    Run one training step.

    `model` and `batch_data` should be on the same device.
    """

    device = batch_data.input_ids.device
    total_tokens_valid = sum(batch_data.tokens_valid)

    model.train()

    loss_all = torch.tensor(0.0, dtype=torch.float32, device=device)

    optimizer.zero_grad()

    # forward / backward
    for i in range(batch_data.batches):
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
        ):
            loss = compute_loss_step(
                model=model,
                input_ids=batch_data.input_ids[i],
                target_ids=batch_data.target_ids[i],
                mask=batch_data.mask[i],
                slot_mask=batch_data.slot_mask[i],
                pad_id=pad_id,
                reduction="sum",
            )

        loss /= total_tokens_valid
        loss_all += loss
        loss.backward()

    # update weights
    optimizer.step()

    return loss_all


def compute_validation_metrics(
    model: Model,
    batch_data: BatchData,
    *,
    pad_id: int,
    use_bf16: bool = True,
) -> tuple[Tensor, Tensor]:
    """
    Compute validation metrics on the given data.

    `model` and `batch_data` should be on the same device.

    Returns:
        avg_loss, avg_perplexity
    """

    device = batch_data.input_ids.device
    total_tokens_valid = sum(batch_data.tokens_valid)

    model.eval()

    sum_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
    sum_perplexities = torch.tensor(0.0, dtype=torch.float32, device=device)

    with torch.no_grad():
        for i in range(batch_data.batches):
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                loss = compute_loss_step(
                    model=model,
                    input_ids=batch_data.input_ids[i],
                    target_ids=batch_data.target_ids[i],
                    mask=batch_data.mask[i],
                    slot_mask=batch_data.slot_mask[i],
                    pad_id=pad_id,
                    reduction="sum",
                )

            sum_loss += loss
            sum_perplexities += torch.exp(loss / batch_data.tokens_valid[i])

    avg_loss = sum_loss / total_tokens_valid
    avg_perplexity = sum_perplexities / batch_data.batches

    return avg_loss, avg_perplexity
