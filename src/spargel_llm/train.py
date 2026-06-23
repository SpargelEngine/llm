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
    time_forward: float
    time_backward: float


def train(
    info: TrainInfo,
    model: Model,
    optimizer: Optimizer,
    batch_iterator: Iterator[tuple[Tensor, Tensor, Tensor]],
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
    The effective batch size is ``micro_batch_size × micro_batches``.

    Returns the number of steps actually executed (may be less than *steps*
    if the data iterator is exhausted).
    """
    assert micro_batches >= 1

    torch_device = torch.device(device)
    device_type = torch_device.type

    for step in range(steps):
        optimizer.zero_grad()

        step_loss = 0.0
        step_time_load_batch = 0.0
        step_time_forward = 0.0
        step_time_backward = 0.0
        step_token_count = 0

        actual_mb = 0
        for mb in range(micro_batches):
            t0 = time.perf_counter()

            try:
                inputs, masks, targets = next(batch_iterator)
            except StopIteration:
                if mb == 0:
                    print("Stopping early because there are no more data.")
                    return step
                # partial step — still apply what we've accumulated
                break

            actual_mb += 1

            inputs2 = inputs.to(torch_device)
            masks2 = masks.to(torch_device)
            targets2 = targets.to(torch_device)

            t1 = time.perf_counter()

            model.train()

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_bf16):
                loss = compute_loss_step(
                    model=model,
                    inputs=inputs2,
                    masks=masks2,
                    targets=targets2,
                    pad_index=pad_index,
                )
            loss = loss / micro_batches

            t2 = time.perf_counter()

            loss.backward()

            t3 = time.perf_counter()

            step_loss += loss.detach().item() * micro_batches
            step_time_load_batch += t1 - t0
            step_time_forward += t2 - t1
            step_time_backward += t3 - t2
            step_token_count += int(torch.sum(~masks).item())

        step_loss = step_loss / actual_mb

        optimizer.step()

        info.time += (
            step_time_load_batch
            + step_time_forward
            + step_time_backward
        )
        info.token_count += step_token_count

        if step_callback is not None:
            step_callback(
                StepInfo(
                    step=step,
                    loss=step_loss,
                    time_load_batch=step_time_load_batch,
                    time_forward=step_time_forward,
                    time_backward=step_time_backward,
                )
            )

    return steps
