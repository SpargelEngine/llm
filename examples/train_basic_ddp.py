import os
import random
import sys

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

from spargel_llm.model import Config, Model
from spargel_llm.train.basic import (
    Item,
    compile_batch_data,
    compute_validation_metrics,
    train_step,
)

# Vocab: 0123456789, <pad>, <eot>, +, =

PAD, EOT, PLUS, EQ = 10, 11, 12, 13

seq_len = 16
batch_size = 16


def split_digits(a: int, n: int) -> list[int]:
    digits = [0] * n
    for i in range(n):
        digits[n - 1 - i] = a % 10
        a = a // 10
    return digits


def gen_data() -> Item:
    a, b = random.randint(0, 999), random.randint(0, 999)
    s = a + b

    input_ids = np.full(seq_len, PAD, dtype=np.int64)
    target_ids = np.full(seq_len, PAD, dtype=np.int64)
    mask = np.zeros(seq_len, dtype=np.bool)

    input_ids[:12] = (
        split_digits(a, 3) + [PLUS] + split_digits(b, 3) + [EQ] + split_digits(s, 4)
    )
    target_ids[7:12] = split_digits(s, 4) + [EOT]
    mask[:12] = True

    return Item(
        seq_len,
        input_ids=input_ids,
        target_ids=target_ids,
        mask=mask,
        tokens_valid=5,
    )


def main(rank: int, world_size: int, steps: int):
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

    model = DDP(
        Model(
            Config(
                vocab_size=16,
                max_seq_len=16,
                num_layer=4,
                num_head=4,
                dim=64,
                dim_key=16,
                dim_value=16,
                dim_ff_hidden=256,
            )
        ).to(rank),
        device_ids=[rank],
    )
    optimizer = AdamW(model.parameters())

    # Prepare validation data
    if rank == 0:
        val_batch_data = compile_batch_data(
            [gen_data() for _ in range(batch_size)], batch_size, seq_len, pad_id=PAD
        ).to(0)
    else:
        val_batch_data = None

    # Train loop
    count = 0
    sum_loss = 0.0

    for i in range(steps):
        items = [gen_data() for _ in range(batch_size)]
        batch_data = compile_batch_data(items, batch_size, seq_len, pad_id=PAD).to(rank)
        loss = train_step(model, optimizer, batch_data, pad_id=PAD)

        count += 1
        sum_loss += loss.item()

        if (i + 1) % 100 == 0:
            if val_batch_data:
                avg_loss, avg_perplexity = compute_validation_metrics(
                    model, val_batch_data, pad_id=PAD
                )
                print(
                    f"{i+1}@{rank}: avg_loss={sum_loss / count:.6f}, val_loss={avg_loss.item():.6f}, val_ppl={avg_perplexity.item():.4f}"
                )
            else:
                print(f"{i+1}@{rank}: avg_loss={sum_loss / count:.6f}")

            count = 0
            sum_loss = 0.0


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print(f"Usage: {sys.argv[0]} steps")
        sys.exit(1)
    steps = int(sys.argv[1])

    num_threads = os.cpu_count() or 8
    torch.set_num_threads(num_threads)
    print(f"PyTorch will use {num_threads} CPU thread(s).")

    if not torch.cuda.is_available():
        print("CUDA not available.")
        sys.exit(1)
    print("Using device: cuda")

    device_count = torch.cuda.device_count()
    print(f"Device count: {device_count}")

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    mp.spawn(main, args=(device_count, steps), nprocs=device_count, join=True)
