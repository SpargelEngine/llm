import time

import torch
import torch.optim as optim
from pydantic import BaseModel, NonNegativeFloat, NonNegativeInt
from torch import Tensor

from spargel_llm.data import DataLoader

from .model import LLM, Config


class TrainInfo(BaseModel):
    trained_steps: NonNegativeInt = 0
    trained_time: NonNegativeFloat = 0


class TokenInfo(BaseModel):
    vocab: list[str] = []
    pad: NonNegativeInt = 0
    unknown: NonNegativeInt = 0


class ModelInfo(BaseModel):
    train_info: TrainInfo
    token_info: TokenInfo
    config: Config


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
    loss = model.loss(inputs, masks, targets, pad_index=pad_index)
    loss.backward()
    optimizer.step()

    return loss


def train(
    info: TrainInfo,
    model: LLM,
    optimizer: optim.Optimizer,
    data_loader: DataLoader[list[int]],  # tokens
    seq_len: int,
    pad_index: int,
    batch_size: int,
    epochs: int,
    *,
    log_period: int = 100,
):
    print(16 * "=")

    for epoch in range(epochs):
        print(f"Epoch {epoch}:")

        t_epoch_start = t_last = time.perf_counter()

        iterator = iter(data_loader)

        step = 0
        stop = False

        sum_of_time = 0

        while True:
            # prepare a batch of data
            inputs = torch.zeros(batch_size, seq_len, dtype=torch.int)
            masks = torch.zeros(batch_size, seq_len, dtype=torch.bool)
            targets = torch.zeros(batch_size, seq_len, dtype=torch.int)

            for i in range(batch_size):
                try:
                    tokens = next(iterator)
                except StopIteration:
                    stop = True
                    break

                length = len(tokens) - 1

                if length <= 0 or 1 > seq_len:
                    raise ValueError("incorrect number of tokens")

                inputs[i] = torch.tensor(
                    tokens[:-1] + (seq_len - length) * [pad_index], dtype=torch.int
                )
                masks[i] = torch.tensor(
                    length * [False] + (seq_len - length) * [True], dtype=torch.bool
                )
                targets[i] = torch.tensor(
                    tokens[1:] + (seq_len - length) * [pad_index], dtype=torch.int
                )

            if stop:
                break

            # train
            model.train()

            loss = train_step(
                model=model,
                optimizer=optimizer,
                inputs=inputs,
                masks=masks,
                targets=targets,
                pad_index=pad_index,
            )

            step += 1
            info.trained_steps += 1

            t = time.perf_counter()
            delta_t = t - t_last
            t_last = t
            info.trained_time += delta_t
            sum_of_time += delta_t

            if step % log_period == 0:
                print(f"  Step {step}: loss={loss.item():.6f}, time={sum_of_time:.6f}")
                sum_of_time = 0

        t_epoch_end = time.perf_counter()

        print(f"Epoch time: {t_epoch_end - t_epoch_start:.6f}")
        print(f"Total steps: {info.trained_steps}")
        print(f"Total time: {info.trained_time:.6f}")
        print()

    print("Done.")
