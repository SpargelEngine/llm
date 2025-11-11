import time
from collections import namedtuple
from dataclasses import dataclass
from typing import Callable, Iterator, Optional, cast, override

import torch
from pydantic import BaseModel, NonNegativeFloat, NonNegativeInt
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, IterableDataset

from spargel_llm.data import DataSource

from .model import LLM


class TrainInfo(BaseModel):
    time: NonNegativeFloat = 0
    token_count: NonNegativeInt = 0


Sample = namedtuple("Sample", ["input", "mask", "target"])


class TrainDataset(IterableDataset[Sample]):
    data_source: DataSource[list[int]]
    seq_len: int
    pad_index: int

    def __init__(
        self, data_source: DataSource[list[int]], seq_len: int, pad_index: int
    ):
        self.data_source = data_source
        self.seq_len = seq_len
        self.pad_index = pad_index

    @override
    def __iter__(self) -> Iterator[Sample]:

        def generator():
            while True:
                tokens = self.data_source.sample()

                if len(tokens) == 0:
                    continue

                length = len(tokens) - 1

                input = torch.tensor(
                    tokens[:-1] + (self.seq_len - length) * [self.pad_index],
                    dtype=torch.int,
                )
                mask = torch.tensor(
                    length * [False] + (self.seq_len - length) * [True],
                    dtype=torch.bool,
                )
                target = torch.tensor(
                    tokens[1:] + (self.seq_len - length) * [self.pad_index],
                    dtype=torch.int,
                )

                yield Sample(input=input, mask=mask, target=target)

        return iter(generator())


@torch.compile
def forward_step(
    model: LLM,
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
def generate_step(model: LLM, input: Tensor) -> Tensor:
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
    model: LLM,
    seq_len: int,
    optimizer: Optimizer,
    data_source: DataSource[list[int]],  # tokens
    pad_index: int,
    batch_size: int,
    steps: int,
    *,
    device: str = "cpu",
    step_callback: Optional[Callable[[StepInfo], None]] = None,
):
    """
    Args:
        step_callback: (step, token_count, loss, step_time) -> None
    """

    data_loader = DataLoader(
        TrainDataset(data_source, seq_len, pad_index), batch_size=batch_size
    )
    iterator = iter(data_loader)

    for step in range(steps):
        t0 = time.perf_counter()

        try:
            inputs, masks, targets = cast(tuple[Tensor, Tensor, Tensor], next(iterator))
        except StopIteration:
            print("Stopping early because there are no more data.")
            break

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
        token_count = int(torch.sum(masks == False).item())
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
