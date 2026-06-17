import time
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import torch
from pydantic import BaseModel, NonNegativeFloat, NonNegativeInt
from torch import Tensor
from torch.optim import Optimizer

from .model import Model


class TrainInfo(BaseModel):
    time: NonNegativeFloat = 0
    token_count: NonNegativeInt = 0
    index: NonNegativeInt = 0
    offset: NonNegativeInt = 0
    dataset_id: str = ""


@torch.compile
def compute_loss_step(
    model: Model,
    inputs: Tensor,
    masks: Tensor,
    targets: Tensor,
    pad_index: int,
):
    return model.loss(inputs, masks, targets, pad_index=pad_index)


@torch.compile
def generate_step(model: Model, input: Tensor) -> Tensor:
    return model(input)


@dataclass
class StepInfo:
    step: int
    loss: float
    time_load_batch: float
    time_transfer_batch: float
    time_forward: float
    time_backward: float


def train(
    info: TrainInfo,
    model: Model,
    optimizer: Optimizer,
    batch_iterator: Iterator[tuple[Tensor, Tensor, Tensor]],
    pad_index: int,
    batch_size: int,
    steps: int,
    *,
    device: str = "cpu",
    step_callback: Optional[Callable[[StepInfo], None]] = None,
    gradient_accumulation_steps: int = 1,
) -> int:
    """Run up to *steps* training steps.

    *gradient_accumulation_steps* (>=1) accumulates gradients over that many
    consecutive batches before calling ``optimizer.step()``, simulating a
    larger effective batch size without extra GPU memory.

    Returns the number of steps actually executed (may be less than *steps*
    if the data iterator is exhausted).
    """
    assert gradient_accumulation_steps >= 1

    accumulate = gradient_accumulation_steps

    for step in range(steps):
        t0 = time.perf_counter()

        try:
            inputs, masks, targets = next(batch_iterator)
        except StopIteration:
            print("Stopping early because there are no more data.")
            return step

        t1 = time.perf_counter()

        inputs2 = inputs.to(device)
        masks2 = masks.to(device)
        targets2 = targets.to(device)

        t2 = time.perf_counter()

        # train
        model.train()

        # zero_grad at the start of each effective batch.
        if step % accumulate == 0:
            optimizer.zero_grad()

        loss = compute_loss_step(
            model=model,
            inputs=inputs2,
            masks=masks2,
            targets=targets2,
            pad_index=pad_index,
        )
        loss = loss / accumulate

        t3 = time.perf_counter()

        loss.backward()

        if (step + 1) % accumulate == 0:
            optimizer.step()

        t4 = time.perf_counter()

        # Report the unscaled per-batch loss for consistent logging.
        loss_val = loss.detach().item() * accumulate

        info.time += t4 - t0

        # count tokens (not paddings)
        token_count = int(torch.sum(~masks).item())
        info.token_count += token_count

        if step_callback is not None:
            step_callback(
                StepInfo(
                    step=step,
                    loss=loss_val,
                    time_load_batch=t1 - t0,
                    time_transfer_batch=t2 - t1,
                    time_forward=t3 - t2,
                    time_backward=t4 - t3,
                )
            )

    return steps
