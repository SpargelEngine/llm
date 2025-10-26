import time
from collections import namedtuple
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
def train_step(
    model: LLM,
    optimizer: Optimizer,
    inputs: Tensor,
    masks: Tensor,
    targets: Tensor,
    pad_index: int,
):
    optimizer.zero_grad()
    loss = model.loss(inputs, masks, targets, pad_index=pad_index)
    loss.backward()
    optimizer.step()

    return loss


@torch.compile
def generate_step(model: LLM, input: Tensor) -> Tensor:
    return model(input)


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
    log_period: int = 10,
    loss_callback: Optional[Callable[[int, int, float], None]] = None,
    save_period: int = 100,
    save_callback: Optional[Callable[[], None]] = None,
):
    data_loader = DataLoader(
        TrainDataset(data_source, seq_len, pad_index), batch_size=batch_size
    )
    iterator = iter(data_loader)

    sum_of_time = 0
    sum_of_loss = 0.0
    total_token_count = 0
    t_start = t_last = time.perf_counter()

    for step in range(1, steps + 1):

        try:
            inputs, masks, targets = cast(tuple[Tensor, Tensor, Tensor], next(iterator))
        except StopIteration:
            print("Stopping early because there are no more data.")
            break

        # train
        model.train()

        loss = train_step(
            model=model,
            optimizer=optimizer,
            inputs=inputs.to(device),
            masks=masks.to(device),
            targets=targets.to(device),
            pad_index=pad_index,
        )

        t = time.perf_counter()
        delta_t = t - t_last
        t_last = t
        info.time += delta_t
        sum_of_time += delta_t

        sum_of_loss += loss.detach().item()

        # count tokens (not paddings)
        token_count = int(torch.sum(masks == False).item())
        info.token_count += token_count
        total_token_count += token_count

        # log the average loss from last time
        if step % log_period == 0:
            avg_of_loss = sum_of_loss / log_period

            print(f"  Step {step}: loss={avg_of_loss:.6f}, time={sum_of_time:.6f}")

            if loss_callback is not None:
                loss_callback(step, info.token_count, avg_of_loss)

            sum_of_time = 0
            sum_of_loss = 0

        if save_callback is not None and step % save_period == 0:
            save_callback()

    t_end = time.perf_counter()

    print()
    if save_callback is not None:
        save_callback()

    print(f"time: {t_end - t_start:.6f}")
    print(f"tokens: {total_token_count}")
