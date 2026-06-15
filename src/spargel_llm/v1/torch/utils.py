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
def forward_step(
    model: Model,
    optimizer: Optimizer,
    inputs: Tensor,
    masks: Tensor,
    targets: Tensor,
    pad_index: int,
):
    optimizer.zero_grad()
    loss = model.loss(inputs, masks, targets, pad_index=pad_index)
    return loss


@torch.compile
def backward_step(loss: Tensor, optimizer: Optimizer):
    loss.backward()
    optimizer.step()


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
) -> int:
    """Run up to *steps* training steps.

    Returns the number of steps actually executed (may be less than *steps*
    if the data iterator is exhausted).
    """

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

        loss = forward_step(
            model=model,
            optimizer=optimizer,
            inputs=inputs2,
            masks=masks2,
            targets=targets2,
            pad_index=pad_index,
        )

        t3 = time.perf_counter()

        backward_step(loss, optimizer)

        t4 = time.perf_counter()

        info.time += t4 - t0

        # count tokens (not paddings)
        token_count = int(torch.sum(~masks).item())
        info.token_count += token_count

        if step_callback is not None:
            step_callback(
                StepInfo(
                    step=step,
                    loss=loss.detach().item(),
                    time_load_batch=t1 - t0,
                    time_transfer_batch=t2 - t1,
                    time_forward=t3 - t2,
                    time_backward=t4 - t3,
                )
            )

    return steps
