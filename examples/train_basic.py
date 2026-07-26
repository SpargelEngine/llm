import os
import random

import numpy as np
import torch
from torch.optim import AdamW

from spargel_llm.model import Config, Model
from spargel_llm.train.basic import (
    Item,
    compile_batch_data,
    compute_validation_metrics,
    train_step,
)

num_threads = os.cpu_count() or 8
torch.set_num_threads(num_threads)
print(f"PyTorch will use {num_threads} CPU thread(s).")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

use_bf16 = device == "cuda"
print(f"Use BF16: {use_bf16}")


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


model = Model(
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
)
model = model.to(device)
optimizer = AdamW(model.parameters())


# Prepare validation data
val_batch_data = compile_batch_data(
    [gen_data() for _ in range(batch_size * 10)], batch_size, seq_len, pad_id=PAD
).to(device)


# Train loop
count = 0
sum_loss = 0.0

for i in range(1000):
    items = [gen_data() for _ in range(batch_size)]
    batch_data = compile_batch_data(items, batch_size, seq_len, pad_id=PAD).to(device)
    loss = train_step(model, optimizer, batch_data, pad_id=PAD, use_bf16=use_bf16)

    count += 1
    sum_loss += loss.item()

    if (i + 1) % 100 == 0:
        avg_loss, avg_perplexity = compute_validation_metrics(
            model, val_batch_data, pad_id=PAD, use_bf16=use_bf16
        )
        print(
            f"{i+1}: avg_loss={sum_loss / count:.6f}, val_loss={avg_loss.item():.6f}, val_ppl={avg_perplexity.item():.4f}"
        )

        count = 0
        sum_loss = 0.0
