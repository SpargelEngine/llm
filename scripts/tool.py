import json
import os
import random
import shutil
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pyarrow.parquet as pq
import torch
from pydantic import BaseModel, NonNegativeInt
from rich import print as rich_print
from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.utils.tensorboard import SummaryWriter

from spargel_llm.logging import log_info, log_success
from spargel_llm.model import Config, Model
from spargel_llm.train import StepInfo, TrainInfo, generate_step, train
from spargel_llm.typing import StrOrPath
from spargel_llm.utils import (
    PromptAbortError,
    prompt_overwrite,
)

PAD, EOT = 1, 2


class TrainConfig(BaseModel):
    seq_len: NonNegativeInt
    batch_size: NonNegativeInt
    learning_rate: float
    weight_decay: float
    optimizer_state_file: str
    micro_batches: NonNegativeInt = 1


class ProjectInfo(BaseModel):
    # configuration (hyper-parameters) for the model
    config: Config

    # weight file location
    model_state_file: str

    # tokenizer file location (e.g. tokenizer.json)
    tokenizer: str

    # training information & statistics
    train_info: TrainInfo

    # training config
    train_config: TrainConfig


#### load/store helper functions ####


def load_project(path: StrOrPath) -> ProjectInfo:
    with open(path, "r") as f:
        return ProjectInfo.model_validate_json(f.read())


def save_project(path: StrOrPath, project_info: ProjectInfo):
    with open(path, "w") as f:
        f.write(project_info.model_dump_json(indent=2))


def load_model_state(path: StrOrPath, model: Model, *, device: str):
    model.load_state_dict(torch.load(path, weights_only=True, map_location=device))


def save_model_state(path: StrOrPath, model: Model):
    torch.save(model.state_dict(), path)


def load_optimizer_state(path: StrOrPath, optimizer: Optimizer):
    optimizer.load_state_dict(torch.load(path))


def save_optimizer_state(path: StrOrPath, optimizer: Optimizer):
    torch.save(optimizer.state_dict(), path)


#### other helpers ####


def resolve_parent(path: StrOrPath):
    return Path(path).resolve().parent


def load_tokenizer(path: StrOrPath) -> Tokenizer:
    """Load a HuggingFace ``tokenizers.Tokenizer`` from a JSON file."""
    return Tokenizer.from_file(str(path))


def load_dataset(path: StrOrPath) -> pq.ParquetFile:
    """Load a pre-tokenized Parquet dataset.

    The returned ``ParquetFile`` provides streaming batch iteration and
    does not fully load the data into memory.
    """
    path = Path(path).resolve()
    parquet_file = pq.ParquetFile(str(path))
    log_info(
        f"Loaded dataset from {path} "
        f"({parquet_file.metadata.num_rows:,} rows, "
        f"{parquet_file.metadata.num_row_groups} row groups)."
    )
    return parquet_file


def get_dataset_id(parquet_file: pq.ParquetFile) -> str:
    """Extract the ``dataset_id`` from a Parquet file's schema metadata."""
    metadata = parquet_file.schema_arrow.metadata
    if metadata is None:
        return ""
    return metadata.get(b"dataset_id", b"").decode()


def iter_batches(
    dataset: pq.ParquetFile,
    seq_len: int,
    batch_size: int,
    pad_index: int,
    stride: int | None = None,
    start_index: int = 0,
    start_offset: int = 0,
    tracker: dict | None = None,
    eot_index: int | None = None,
) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
    """Iterate through a pre-tokenized Parquet dataset and yield tensor batches.

    For each row, a window of ``seq_len + 1`` tokens slides with the given
    *stride* (default: ``seq_len``).  Each window produces one sample:
    ``(input_ids, mask, target_ids)``.  Short tails are padded with
    *pad_index*.

    Works directly on Arrow's underlying NumPy buffers to avoid the
    Python-object overhead of ``to_pylist()``.  Samples are accumulated
    into pre-allocated tensor batches.  Partial batches at the end are
    silently discarded.

    *start_index* skips the first N items of *dataset*.

    *start_offset* is used as the initial window position only for the
    first item (the one at *start_index*); all subsequent items start from
    the beginning.

    If *tracker* is a dict, it is updated before each sample with keys
    ``index`` (current dataset row) and ``offset`` (window position
    within that row).

    If *eot_index* is not ``None``, each row is treated as if it ends with
    an additional token of that value, so the effective row length becomes
    ``row_len + 1`` and the final window's target includes the EOT token.
    """
    if stride is None:
        stride = seq_len

    first = True
    row_index = 0

    # Pre-allocate batch buffers as numpy arrays (reused across batches).
    batch_inputs = np.full((batch_size, seq_len), pad_index, dtype=np.int32)
    batch_masks = np.ones((batch_size, seq_len), dtype=bool)
    batch_targets = np.full((batch_size, seq_len), pad_index, dtype=np.int32)
    sample_in_batch = 0

    # Skip row groups that are entirely before start_index.
    start_rg = 0
    rg_start = 0
    while start_rg < dataset.metadata.num_row_groups:
        rg_rows = dataset.metadata.row_group(start_rg).num_rows
        if rg_start + rg_rows <= start_index:
            rg_start += rg_rows
            row_index += rg_rows
            start_rg += 1
            continue
        break

    for rg_idx in range(start_rg, dataset.metadata.num_row_groups):
        table = dataset.read_row_group(rg_idx)
        col = table.column("tokens").combine_chunks()
        offsets = col.offsets.to_numpy()
        values = col.values.to_numpy().astype(np.int32)

        n_rows = len(col)

        for i in range(n_rows):
            if row_index < start_index:
                row_index += 1
                continue

            row_start = int(offsets[i])
            row_end = int(offsets[i + 1])
            row_len = row_end - row_start

            row_len_with_eot = row_len + 1 if eot_index is not None else row_len

            if row_len_with_eot <= 1:
                row_index += 1
                continue

            pos = start_offset if first else 0
            first = False

            while pos + 1 < row_len_with_eot:
                end = min(pos + seq_len + 1, row_len_with_eot)
                L = end - pos - 1

                # Write sample directly into the pre-allocated batch row.
                batch_inputs[sample_in_batch, :L] = values[
                    row_start + pos : row_start + pos + L
                ]
                batch_inputs[sample_in_batch, L:] = pad_index
                batch_masks[sample_in_batch, :L] = False
                batch_masks[sample_in_batch, L:] = True

                # Target: may include EOT at the final position.
                orig_target_len = min(L, row_len - (pos + 1))
                if orig_target_len > 0:
                    batch_targets[sample_in_batch, :orig_target_len] = values[
                        row_start + pos + 1 : row_start + pos + 1 + orig_target_len
                    ]
                if end > row_len and eot_index is not None:
                    batch_targets[sample_in_batch, orig_target_len] = eot_index
                batch_targets[sample_in_batch, L:] = pad_index

                if tracker is not None:
                    tracker["index"] = row_index
                    tracker["offset"] = pos

                sample_in_batch += 1

                if sample_in_batch == batch_size:
                    yield (
                        torch.tensor(batch_inputs),
                        torch.tensor(batch_masks),
                        torch.tensor(batch_targets),
                    )
                    # Reset for reuse.
                    batch_inputs.fill(pad_index)
                    batch_masks.fill(True)
                    batch_targets.fill(pad_index)
                    sample_in_batch = 0

                pos += stride

            row_index += 1


def get_backup_path(path: StrOrPath):
    return (
        resolve_parent(path)
        / f".{Path(path).resolve().name}.{int(datetime.now().timestamp())}"
    )


def writer_add_graph(
    writer: SummaryWriter,
    model: Model,
    *,
    seq_len: int,
    device: str = "cpu",
    pad_index: int = PAD,
):
    """Write the model graph to TensorBoard using dummy input."""
    model.eval()
    dummy_input = torch.full((1, seq_len), pad_index, dtype=torch.long, device=device)
    dummy_mask = torch.ones((1, seq_len), dtype=torch.bool, device=device)
    writer.add_graph(model, (dummy_input, dummy_mask))


def writer_add_embedding(
    writer: SummaryWriter,
    model: Model,
    tokenizer: Tokenizer,
    *,
    device: str = "cpu",
):
    model.eval()
    with torch.no_grad():
        vocab_size = tokenizer.get_vocab_size()
        indices = torch.arange(vocab_size, dtype=torch.int, device=device)
        embed_vectors = model.embedding(indices)

        labels = [tokenizer.id_to_token(i) for i in range(vocab_size)]

    writer.add_embedding(embed_vectors.cpu(), labels)


@torch.compile
def validation_loss_step(
    model: Model,
    inputs: Tensor,
    masks: Tensor,
    targets: Tensor,
    pad_index: int,
):
    return model.loss(inputs, masks, targets, pad_index=pad_index)


def compute_validation_loss(
    model: Model,
    dataset: pq.ParquetFile,
    seq_len: int,
    batch_size: int,
    pad_index: int,
    device: str,
    num_batches: int = 10,
    eot_index: int | None = None,
) -> float:
    assert num_batches > 0

    total_loss = 0.0

    def make_iterator():
        return iter_batches(
            dataset, seq_len, batch_size, pad_index, eot_index=eot_index
        )

    iterator = make_iterator()

    model.eval()

    with torch.no_grad():
        for _ in range(num_batches):
            try:
                inputs, masks, targets = next(iterator)
            except StopIteration:
                iterator = make_iterator()
                inputs, masks, targets = next(iterator)

            loss = validation_loss_step(
                model,
                inputs.to(device),
                masks.to(device),
                targets.to(device),
                pad_index=PAD,
            )

            total_loss += loss.detach().item()

    return total_loss / num_batches


def always_true():
    while True:
        yield True


#### actions ####


def action_info(path: StrOrPath):
    project_info = load_project(path)

    log_info("==== Project Info ====")
    print("Project file:", path)
    rich_print(project_info)


def action_init(path: StrOrPath, *, yes: bool = False):
    prompt_overwrite(path, yes=yes)

    # default config
    dim = 256
    num_layer = 4
    num_head = 4
    assert dim % num_head == 0

    config = Config(
        vocab_size=1000,
        max_seq_len=256,
        num_layer=num_layer,
        num_head=num_head,
        dim=dim,
        dim_key=dim // num_head,
        dim_value=dim // num_head,
        dim_feed_forward=dim * 4,
    )

    model_state_file = "model_state.pth"
    tokenizer = "tokenizer.json"

    project_info = ProjectInfo(
        config=config,
        model_state_file=model_state_file,
        tokenizer=tokenizer,
        train_info=TrainInfo(),
        train_config=TrainConfig(
            seq_len=0,
            batch_size=0,
            learning_rate=1e-3,
            weight_decay=0.1,
            optimizer_state_file="optimizer_state.pth",
        ),
    )

    save_project(path, project_info)
    log_success(f"Initialized project at {path}.")


def action_gen(
    path: StrOrPath,
    seq_len: int,
    prompt: str,
    count: int = 0,
    temperature: float = 0.5,
    *,
    device: str = "cpu",
    stream: bool = False,
    all: bool = False,
    stop_token: int = EOT,
    add_eot: bool = False,
    dump_file: Optional[str] = None,
    random_seed: Optional[int] = None,
):
    assert seq_len > 0
    assert temperature > 0

    if not add_eot:
        assert len(prompt) > 0

    project_info = load_project(path)

    model = Model(project_info.config).to(device)

    load_model_state(
        resolve_parent(path) / project_info.model_state_file, model, device=device
    )

    tokenizer = load_tokenizer(resolve_parent(path) / project_info.tokenizer)

    prompt_tokens = tokenizer.encode(prompt).ids

    if random_seed is None:
        random_seed = random.randrange(2**63)
    torch.manual_seed(random_seed)

    dump_f = None
    if dump_file is not None:
        dump_f = open(dump_file, "w")
        json.dump(
            {
                "type": "info",
                "temperature": temperature,
                "prompt": prompt_tokens,
                "random_seed": random_seed,
            },
            dump_f,
        )
        dump_f.write("\n")

    tokens = list(prompt_tokens)

    print("Prompt:", repr(tokenizer.decode(tokens)))
    print("Prompt token count:", len(tokens))
    print("Sequence length:", seq_len)
    print("Temperature:", temperature)
    print("Max generation count:", count)
    print("Stop token id:", stop_token)
    print("Random seed:", random_seed)

    print("Generated text:")
    log_info("********")

    start_pos = len(tokens)

    model.eval()

    cnt = 0

    decode_stream = DecodeStream() if stream else None

    if stream and all:
        print(tokenizer.decode(tokens), end="")

    for _ in range(count) if count >= 0 else always_true():
        input = tokens[-seq_len:]

        # pad length to a power of two
        length = len(input)
        if length < seq_len:
            length_expected = 1 << (length - 1).bit_length()
            input = input + [PAD] * (length_expected - length)

        with torch.no_grad():
            logits = generate_step(model, torch.tensor(input, device=device))

        logits = logits[length - 1, :]  # get the new token
        probs = torch.softmax(logits / temperature, dim=-1)
        next = int(torch.multinomial(probs, num_samples=1).item())

        tokens.append(next)

        if dump_f is not None:
            json.dump(
                {
                    "type": "gen",
                    "id": next,
                    "logits": logits.cpu().tolist(),
                },
                dump_f,
            )
            dump_f.write("\n")

        if stream and decode_stream:
            chunk = decode_stream.step(tokenizer, next)
            if chunk is not None:
                print(chunk, end="", flush=True)

        cnt += 1

        if stop_token >= 0 and next == stop_token:
            break

    if stream:
        print()
    else:
        if all:
            print(tokenizer.decode(tokens))
        else:
            print(tokenizer.decode(tokens[start_pos:]))

    log_info("********")
    print("Generated token count:", len(tokens) - start_pos)

    if dump_f is not None:
        dump_f.close()


def action_train(
    path: StrOrPath,
    data_path: str,
    steps: int,
    *,
    seq_len: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    weight_decay: Optional[float] = None,
    val_path: Optional[str] = None,
    log_period: int = 10,
    tensorboard_dir: Optional[str] = None,
    device: str = "cpu",
    start_index: Optional[int] = None,
    start_offset: Optional[int] = None,
    add_eot: bool = False,
    gradient_accumulation_steps: Optional[int] = None,
    loop_dataset: bool = False,
    use_bf16: bool = True,
):
    assert log_period > 0

    project_info = load_project(path)
    model_state_file = resolve_parent(path) / project_info.model_state_file
    train_config = project_info.train_config
    optimizer_state_file = resolve_parent(path) / train_config.optimizer_state_file

    # Optimizer state (momentum, etc.) depends on the effective batch size —
    # reset whenever it changes.
    new_bs = batch_size if batch_size is not None else train_config.batch_size
    prev_bs = train_config.batch_size
    should_reset_optimizer = new_bs != prev_bs

    if seq_len is None:
        seq_len = train_config.seq_len
    else:
        train_config.seq_len = seq_len
    assert seq_len > 0

    if batch_size is None:
        batch_size = train_config.batch_size
    else:
        train_config.batch_size = batch_size
    assert batch_size > 0

    if learning_rate is None:
        learning_rate = train_config.learning_rate
    else:
        train_config.learning_rate = learning_rate

    if weight_decay is None:
        weight_decay = train_config.weight_decay
    else:
        train_config.weight_decay = weight_decay

    if gradient_accumulation_steps is None:
        micro_batches = train_config.micro_batches or 1
    else:
        micro_batches = gradient_accumulation_steps
        train_config.micro_batches = micro_batches
    assert micro_batches >= 1
    assert batch_size % micro_batches == 0, (
        f"batch_size ({batch_size}) must be divisible by "
        f"micro_batches ({micro_batches})"
    )

    micro_batch_size = batch_size // micro_batches

    # data

    log_info("Loading training dataset.")
    dataset = load_dataset(data_path)

    # detect dataset replacement
    new_id = get_dataset_id(dataset)
    old_id = project_info.train_info.dataset_id
    if new_id and new_id != old_id:
        log_info(f"Dataset changed ({old_id!r} -> {new_id!r}).")
        if start_index is None:
            project_info.train_info.index = 0
        if start_offset is None:
            project_info.train_info.offset = 0
        project_info.train_info.dataset_id = new_id

    # start position: CLI args override TrainInfo
    if start_index is None:
        start_index = project_info.train_info.index
    if start_offset is None:
        start_offset = project_info.train_info.offset

    log_info(f"Start position: index={start_index}, offset={start_offset}")

    eot_index = EOT if add_eot else None

    train_tracker: dict = {"index": start_index, "offset": start_offset}

    def make_iterator():
        return iter_batches(
            dataset,
            seq_len,
            micro_batch_size,
            PAD,
            start_index=start_index or 0,
            start_offset=start_offset or 0,
            tracker=train_tracker,
            eot_index=eot_index,
        )

    val_batches = max(log_period // 10, 1)
    val_dataset = None
    if val_path is not None:
        log_info("Loading validation dataset.")
        val_dataset = load_dataset(val_path)

    # model

    model = Model(project_info.config).to(device)
    log_info("Loading model state.")
    load_model_state(model_state_file, model, device=device)

    # optimizer

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if should_reset_optimizer:
        log_info("Batch size changed, optimizer reset.")
    else:
        log_info("Loading optimzier state.")
        load_optimizer_state(optimizer_state_file, optimizer)

    log_info(f"Use BF16: {use_bf16}")

    # TensorBoard

    writer = None
    if tensorboard_dir is not None:
        log_info("Opening TensorBoard writer.")
        writer = SummaryWriter(tensorboard_dir)

    # helper functions

    def log_important(msg: str):
        log_info(msg)
        if writer is not None:
            writer.add_text("train/log", msg, project_info.train_info.token_count)

    def save():
        project_info.train_info.index = train_tracker["index"]
        project_info.train_info.offset = train_tracker["offset"]
        log_info(f"Saving. (train_info: {project_info.train_info})")
        save_project(path, project_info)
        save_model_state(model_state_file, model)
        save_optimizer_state(optimizer_state_file, optimizer)

    def backup():
        log_info("Making backups.")
        shutil.copyfile(path, get_backup_path(path))
        shutil.copyfile(model_state_file, get_backup_path(model_state_file))
        shutil.copyfile(optimizer_state_file, get_backup_path(optimizer_state_file))

    state = {
        "sum_loss": 0.0,
        "sum_time_load_batch": 0.0,
        "sum_time_forward": 0.0,
        "sum_time_backward": 0.0,
    }

    def step_callback(info: StepInfo):
        token_count = project_info.train_info.token_count

        if writer is not None:
            writer.add_scalar("loss/train", info.loss, token_count)

        state["sum_loss"] += info.loss
        state["sum_time_load_batch"] += info.time_load_batch
        state["sum_time_forward"] += info.time_forward
        state["sum_time_backward"] += info.time_backward

        step = info.step + 1

        if step % log_period == 0:
            avg_loss = state["sum_loss"] / log_period
            avg_time_load_batch = state["sum_time_load_batch"] / log_period
            avg_time_forward = state["sum_time_forward"] / log_period
            avg_time_backward = state["sum_time_backward"] / log_period
            avg_time = avg_time_load_batch + avg_time_forward + avg_time_backward

            state["sum_loss"] = 0.0
            state["sum_time_load_batch"] = 0.0
            state["sum_time_forward"] = 0.0
            state["sum_time_backward"] = 0.0

            val_loss = None
            if val_dataset is not None:
                val_loss = compute_validation_loss(
                    model=model,
                    dataset=val_dataset,
                    seq_len=seq_len,
                    batch_size=micro_batch_size,
                    pad_index=PAD,
                    device=device,
                    num_batches=val_batches,
                    eot_index=eot_index,
                )

            time_log_msg = f"avg_time={avg_time:.6f} ({avg_time_load_batch:.6f} + {avg_time_forward:.6f} + {avg_time_backward:.6f})"
            if val_loss is not None:
                print(
                    f"  {step}: avg_loss={avg_loss:.6f}, val_loss={val_loss:.6f}, {time_log_msg}"
                )
                if writer is not None:
                    writer.add_scalar("loss/val", val_loss, token_count)
            else:
                print(f"  {step}: avg_loss={avg_loss:.6f}, {time_log_msg}")

            if step % (log_period * 10) == 0:
                save()

                if step % (log_period * 100) == 0:
                    backup()

    # train

    t_start = time.perf_counter()

    if micro_batches > 1:
        train_msg = (
            f"Training for {steps} steps (seq_len={seq_len}, batch_size={batch_size}, "
            f"micro_batches={micro_batches}, "
            f"micro_batch_size={micro_batch_size}). Time: {datetime.now()}"
        )
    else:
        train_msg = (
            f"Training for {steps} steps (seq_len={seq_len}, batch_size={batch_size}). "
            f"Time: {datetime.now()}"
        )

    log_important(train_msg)

    backup()

    if loop_dataset:
        steps_remaining = steps
        while steps_remaining > 0:
            batch_iterator = make_iterator()
            actual_steps = train(
                info=project_info.train_info,
                model=model,
                optimizer=optimizer,
                batch_iterator=batch_iterator,
                pad_index=PAD,
                steps=steps_remaining,
                device=device,
                step_callback=step_callback,
                micro_batches=micro_batches,
                use_bf16=use_bf16,
            )
            steps_remaining -= actual_steps
            if steps_remaining > 0:
                log_info("Dataset exhausted - restarting from beginning.")
                start_index = 0
                start_offset = 0
                train_tracker["index"] = 0
                train_tracker["offset"] = 0
    else:
        batch_iterator = make_iterator()
        actual_steps = train(
            info=project_info.train_info,
            model=model,
            optimizer=optimizer,
            batch_iterator=batch_iterator,
            pad_index=PAD,
            steps=steps,
            device=device,
            step_callback=step_callback,
            micro_batches=micro_batches,
            use_bf16=use_bf16,
        )

        if actual_steps < steps:
            log_info("Dataset exhausted - resetting train_info position to zero.")
            project_info.train_info.index = 0
            project_info.train_info.offset = 0
            project_info.train_info.dataset_id = ""
            train_tracker["index"] = 0
            train_tracker["offset"] = 0

    save()

    t_end = time.perf_counter()

    log_important(f"Training completed. (time: {t_end - t_start:.6f})")
    log_info(
        f"Reached sample {train_tracker['index']}, offset {train_tracker['offset']} in the dataset."
    )

    if writer is not None:
        writer.close()


def action_model_init(path: StrOrPath, *, yes: bool = False):
    project_info = load_project(path)
    model_state_file = resolve_parent(path) / project_info.model_state_file
    train_config = project_info.train_config
    optimizer_state_file = resolve_parent(path) / train_config.optimizer_state_file

    prompt_overwrite(model_state_file, yes=yes)
    prompt_overwrite(optimizer_state_file, yes=yes)

    model = Model(project_info.config)
    save_model_state(model_state_file, model)
    log_success(f"Initialized model state at {model_state_file}.")

    optimizer = AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    save_optimizer_state(optimizer_state_file, optimizer)
    log_success(f"Initialized optimizer state at {optimizer_state_file}.")

    project_info.train_info = TrainInfo()

    save_project(path, project_info)


def action_graph(
    path: StrOrPath, tensorboard_dir: str, seq_len: int, *, device: str = "cpu"
):
    project_info = load_project(path)
    model_state_file = resolve_parent(path) / project_info.model_state_file

    model = Model(project_info.config).to(device)
    log_info("Loading model state.")
    load_model_state(model_state_file, model, device=device)

    log_info("Opening TensorBoard writer.")
    writer = SummaryWriter(tensorboard_dir)
    writer_add_graph(writer, model, seq_len=seq_len, device=device)
    writer.close()
    log_success(f"Model graph written to {tensorboard_dir}.")


def action_dump_param(path: StrOrPath):
    project_info = load_project(path)
    model_state_file = resolve_parent(path) / project_info.model_state_file

    device = "cpu"
    model = Model(project_info.config).to(device)
    log_info("Loading model state.")
    load_model_state(model_state_file, model, device=device)

    for name, param in model.named_parameters():
        print(f"==== {name} ====")
        print(param)


def action_embed(path: StrOrPath, tensorboard_dir: str, *, device: str = "cpu"):
    project_info = load_project(path)
    model_state_file = resolve_parent(path) / project_info.model_state_file
    tokenizer_file = resolve_parent(path) / project_info.tokenizer

    model = Model(project_info.config).to(device)
    log_info("Loading model state.")
    load_model_state(model_state_file, model, device=device)

    tokenizer = load_tokenizer(tokenizer_file)

    log_info("Opening TensorBoard writer.")
    writer = SummaryWriter(tensorboard_dir)
    writer_add_embedding(writer, model, tokenizer, device=device)
    writer.close()
    log_success(f"Embedding projection written to {tensorboard_dir}.")


#### main ####


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="V1 Model CLI Tool (PyTorch)", fromfile_prefix_chars="@"
    )

    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="always say YES and avoid interactive prompts",
    )
    parser.add_argument(
        "-t",
        "--thread",
        type=int,
        help="number of threads PyTorch will use",
    )

    parser.add_argument(
        "-f32p",
        "--float32-precision",
        choices=["highest", "high", "medium"],
        default="high",
        help="(CUDA) for set_float32_matmul_precision",
    )

    subparsers = parser.add_subparsers(dest="action", help="actions", required=True)

    # info
    info_parser = subparsers.add_parser("info", help="show info")
    info_parser.add_argument("path", help="project file")

    # init
    init_parser = subparsers.add_parser("init", help="initialize a new project")
    init_parser.add_argument("path", help="project file")

    # gen
    gen_parser = subparsers.add_parser("gen", help="generate text")
    gen_parser.add_argument("path", help="project file")
    gen_parser.add_argument("seq_len", type=int, help="sequence length")
    gen_parser.add_argument("prompt", help="prompt from which to start generating")
    gen_parser.add_argument(
        "-c",
        "--count",
        type=int,
        default=200,
        help="maximum number of tokens to generate (negative for infinite)",
    )
    gen_parser.add_argument(
        "-t", "--temp", type=float, default=0.5, help="temperature for sampling"
    )
    gen_parser.add_argument("-s", "--stream", action="store_true", help="stream output")
    gen_parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="show prompt and generated text together",
    )
    gen_parser.add_argument(
        "-st",
        "--stop-token",
        type=int,
        default=EOT,
        help="stop token id (-1: never stop)",
    )
    gen_parser.add_argument("--eot", action="store_true", help="add EOT")
    gen_parser.add_argument(
        "-df",
        "--dump-file",
        type=str,
        default=None,
        help="dump generation details (prompt, temperature, token logits) to the specified file",
    )
    gen_parser.add_argument(
        "-rs",
        "--random-seed",
        type=int,
        default=None,
        help="random seed for reproducibility (randomly generated if not specified)",
    )

    # train
    train_parser = subparsers.add_parser("train", help="train model")
    train_parser.add_argument("path", help="project file")
    train_parser.add_argument("data", help="dataset directory")
    train_parser.add_argument("steps", type=int, help="number of steps")
    train_parser.add_argument("-l", "--seq-len", type=int, help="sequence length")
    train_parser.add_argument("-bs", "--batch-size", type=int, help="batch size")
    train_parser.add_argument(
        "-lr", "--learning-rate", type=float, help="learning rate"
    )
    train_parser.add_argument("-wd", "--weight-decay", type=float, help="weight decay")
    train_parser.add_argument("-v", "--val", help="validation dataset directory")
    train_parser.add_argument(
        "-tb", "--tensorboard-dir", help="TensorBoard write directory"
    )
    train_parser.add_argument(
        "-lp",
        "--log-period",
        type=int,
        default=10,
        help="log each this number of steps",
    )
    train_parser.add_argument(
        "-si",
        "--start-index",
        type=int,
        default=None,
        help="start training from this sample index in the dataset (default: read from project)",
    )
    train_parser.add_argument(
        "-so",
        "--start-offset",
        type=int,
        default=None,
        help="initial window offset for the first sample (default: read from project)",
    )
    train_parser.add_argument(
        "-et",
        "--eot",
        action="store_true",
        help="append EOT to each training/validation text (effective length +1)",
    )
    train_parser.add_argument(
        "-mb",
        "--micro-batches",
        type=int,
        default=None,
        dest="gradient_accumulation_steps",
        help="split each step into N micro-batches with accumulated gradients (default: 1)",
    )
    train_parser.add_argument(
        "-ld",
        "--loop-dataset",
        action="store_true",
        help="restart from the beginning when the dataset is exhausted instead of stopping early",
    )
    train_parser.add_argument(
        "-n16",
        "--no-bf16",
        action="store_true",
        help="disable bfloat16 autocast (useful for CPU or consumer GPUs that don't support bf16)",
    )

    # model_init
    model_init_parser = subparsers.add_parser(
        "model_init",
        help="initialize model accroding to configuration and fill with random weights",
    )
    model_init_parser.add_argument("path", help="project file")

    # embed
    embed_parser = subparsers.add_parser(
        "embed", help="write embedding projection to TensorBoard"
    )
    embed_parser.add_argument("path", help="project file")
    embed_parser.add_argument("tensorboard_dir", help="TensorBoard write directory")

    # graph
    graph_parser = subparsers.add_parser(
        "graph", help="write model graph to TensorBoard"
    )
    graph_parser.add_argument("path", help="project file")
    graph_parser.add_argument("tensorboard_dir", help="TensorBoard write directory")
    graph_parser.add_argument(
        "seq_len",
        type=int,
        help="sequence length for dummy input (use training seq_len)",
    )

    # param
    dump_param_parser = subparsers.add_parser("dump_param", help="dump parameters")
    dump_param_parser.add_argument("path", help="project file")

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    torch.set_printoptions(linewidth=160)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_info(f"Using device: {device}")

    num_threads = 8
    if args.thread is not None:
        num_threads = args.thread
    elif device == "cpu":
        num_threads = os.cpu_count() or 8

    torch.set_num_threads(num_threads)
    log_info(f"PyTorch will use {num_threads} CPU thread(s).")

    # CUDA
    if device == "cuda":
        torch.set_float32_matmul_precision(args.float32_precision)

    match args.action:
        case "info":
            action_info(args.path)
        case "init":
            action_init(args.path, yes=args.yes)
        case "gen":
            action_gen(
                args.path,
                seq_len=args.seq_len,
                prompt=args.prompt,
                count=args.count,
                temperature=args.temp,
                device=device,
                stream=args.stream,
                all=args.all,
                stop_token=args.stop_token,
                add_eot=args.eot,
                dump_file=args.dump_file,
                random_seed=args.random_seed,
            )
        case "train":
            action_train(
                args.path,
                data_path=args.data,
                steps=args.steps,
                seq_len=args.seq_len,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                val_path=args.val,
                device=device,
                tensorboard_dir=args.tensorboard_dir,
                log_period=args.log_period,
                start_index=args.start_index,
                start_offset=args.start_offset,
                add_eot=args.eot,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                loop_dataset=args.loop_dataset,
                use_bf16=not args.no_bf16,
            )
        case "model_init":
            action_model_init(args.path, yes=args.yes)
        case "embed":
            action_embed(args.path, args.tensorboard_dir, device=device)
        case "graph":
            action_graph(args.path, args.tensorboard_dir, args.seq_len, device=device)
        case "dump_param":
            action_dump_param(args.path)
        case _:
            raise ValueError(f"unrecognized action: {args.action}")


if __name__ == "__main__":
    try:
        main()
    except PromptAbortError:
        log_info("Aborting.")
        sys.exit(1)
