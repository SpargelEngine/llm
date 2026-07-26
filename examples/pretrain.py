import os
from collections.abc import Iterator

import torch
from pyarrow.parquet import ParquetFile
from torch.optim import AdamW

from spargel_llm.model import Config, Model
from spargel_llm.train.basic import (
    Item,
    compile_batch_data,
    compute_validation_metrics,
    train_step,
)
from spargel_llm.train.pretrain import iterate_parquet_concat

num_threads = os.cpu_count() or 8
torch.set_num_threads(num_threads)
print(f"PyTorch will use {num_threads} CPU thread(s).")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

use_bf16 = device == "cuda"
print(f"Use BF16: {use_bf16}")


PAD, SEP = 1, 4

seq_len = 16
batch_size = 16

model = Model(
    Config(
        vocab_size=8192,
        max_seq_len=16,
        num_layer=4,
        num_head=4,
        dim=256,
        dim_key=64,
        dim_value=64,
        dim_ff_hidden=1024,
    )
)
model = model.to(device)
optimizer = AdamW(model.parameters())


def next_batches(
    iterator: Iterator[Item],
    batches: int,
    batch_size: int,
    seq_len: int,
    *,
    pad_id: int,
):
    items = []
    for _ in range(batch_size * batches):
        try:
            items.append(next(iterator))
        except StopIteration:
            break
    if len(items) == 0:
        return None

    return compile_batch_data(items, batch_size, seq_len, pad_id=pad_id)


pf_train = ParquetFile("tokens.parquet")
pf_val = ParquetFile("tokens_val.parquet")
iterator_train = iterate_parquet_concat(pf_train, seq_len, pad_id=PAD, sep_id=SEP)

# Prepare validation data
iterator_val = iterate_parquet_concat(pf_val, seq_len, pad_id=PAD, sep_id=SEP)
val_batch_data = next_batches(iterator_val, 10, batch_size, seq_len, pad_id=PAD)
del iterator_val
if val_batch_data is not None:
    val_batch_data = val_batch_data.to(device)

# Show initial validation loss
if val_batch_data is not None:
    avg_loss, avg_perplexity = compute_validation_metrics(
        model, val_batch_data, pad_id=PAD, use_bf16=use_bf16
    )
    print(f"{0}: val_loss={avg_loss.item():.6f}, val_ppl={avg_perplexity.item():.4f}")


count = 0
sum_loss = 0.0

for step in range(1000):
    batch_data = next_batches(iterator_train, 1, batch_size, seq_len, pad_id=PAD)
    if batch_data is None:
        print(f"No data, stop early at step {step}")
        break
    batch_data = batch_data.to(device)
    loss = train_step(model, optimizer, batch_data, pad_id=PAD, use_bf16=use_bf16)
    loss = loss.item()

    count += 1
    sum_loss += loss

    if (step + 1) % 100 == 0:
        if val_batch_data is not None:
            avg_loss, avg_perplexity = compute_validation_metrics(
                model, val_batch_data, pad_id=PAD, use_bf16=use_bf16
            )

            print(
                f"{step+1}: avg_loss={sum_loss / count:.6f}, val_loss={avg_loss.item():.6f}, val_ppl={avg_perplexity.item():.4f}"
            )
        else:
            print(f"{step+1}: avg_loss={sum_loss / count:.6f}")

        count = 0
        sum_loss = 0.0
