import time
from enum import Enum
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from .model import Config, LLM


def calculate_loss(
    model: LLM,
    input: Tensor,
    mask: Tensor,
    target: Tensor,
    pad_index: int,
) -> Tensor:
    """
    Args:
        input: (..., seq_len), dtype=int
        mask: (..., seq_len), dtype=bool
        target: (..., seq_len), dtype=int
    """

    torch._assert(input.shape == target.shape, "shape not matched")

    logits: Tensor = model(input, mask)  # (..., seq_len, vocab_size)
    loss = nn.functional.cross_entropy(
        logits.flatten(0, -2), target.flatten(0, -1), ignore_index=pad_index
    )

    return loss


class TrainInfo:
    trained_steps: int = 0
    trained_time: float = 0


class TokenizerInfo:
    vocab: list[str] = []
    pad: int = 0
    unknown: int = 0


class State:
    model: LLM
    tokenizer_info: TokenizerInfo
    train_info: TrainInfo

    def __init__(self, model: LLM):
        self.model = model
        self.tokenizer_info = TokenizerInfo()
        self.train_info = TrainInfo()


class TrainStage(Enum):
    START = 0
    RUNNING = 1
    END = 2


@torch.compile
def train_step(
    model: LLM,
    optimizer: optim.Optimizer,
    inputs: Tensor,
    masks: Tensor,
    targets: Tensor,
    pad_index: int,
):
    optimizer.zero_grad()
    loss = calculate_loss(model, inputs, masks, targets, pad_index=pad_index)
    loss.backward()
    optimizer.step()


type DataLoader = Callable[
    [int], tuple[Tensor, Tensor, Tensor]
]  # batch_size -> inputs, masks, targets


def train(
    state: State,
    optimizer: optim.Optimizer,
    data_loader: DataLoader,
    batch_size: int,
    steps: int,
    *,
    callback: Optional[Callable[[State, TrainStage], None]] = None,
    callback_period: int = 100,
):
    model = state.model
    train_info = state.train_info

    if callback is not None:
        callback(state, TrainStage.START)

    t_last = time.perf_counter()

    for i in range(steps):
        # prepare data
        inputs, masks, targets = data_loader(batch_size)

        # train
        model.train()

        train_step(
            model=model,
            optimizer=optimizer,
            inputs=inputs,
            masks=masks,
            targets=targets,
            pad_index=state.tokenizer_info.pad,
        )

        # update info
        train_info.trained_steps += 1

        t = time.perf_counter()
        train_info.trained_time += t - t_last
        t_last = t

        # callback
        if (i + 1) % callback_period == 0 and callback is not None:
            callback(state, TrainStage.RUNNING)

    if callback is not None:
        callback(state, TrainStage.END)
