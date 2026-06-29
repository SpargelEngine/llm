import json
import math
import os
import random
import shutil
import socket
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import pyarrow.parquet as pq
import torch
from pydantic import BaseModel, NonNegativeInt
from rich import print as rich_print
from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.utils.tensorboard import SummaryWriter

from spargel_llm.logging import log_info, log_success, log_warning
from spargel_llm.model import Config, Model, compute_param_counts
from spargel_llm.parquet_utils import get_dataset_id
from spargel_llm.train import (
    StepInfo,
    TrainInfo,
    TrainTracker,
    compute_validation_metrics,
    generate_step,
    iter_batches,
    train,
)
from spargel_llm.typing import StrOrPath
from spargel_llm.utils import escape_whitespace

PAD = 1
SOT = 2
EOT = 3


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


@dataclass
class LogState:
    """Mutable state accumulated across training steps for periodic logging."""

    sum_loss: float = 0.0
    sum_time: float = 0.0
    tokens: int = 0
    tokens_non_pad: int = 0
    val_exhausted_warned: bool = False


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


def create_optimizer(model: Model, learning_rate: float, weight_decay: float) -> AdamW:
    return AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


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
    layer: str = "embedding",
    device: str = "cpu",
):
    model.eval()
    with torch.no_grad():
        vocab_size = tokenizer.get_vocab_size()

        def _make_label(i: int) -> str:
            decoded = tokenizer.decode([i], skip_special_tokens=False)
            if not decoded or not decoded.strip():
                return tokenizer.id_to_token(i) or f"<id:{i}>"
            return escape_whitespace(decoded)

        labels = [_make_label(i) for i in range(vocab_size)]

        def add_token_vectors(tag: str, vectors: Tensor):
            writer.add_embedding(vectors.detach().float().cpu(), labels, tag=tag)

        if layer in ("embedding", "both"):
            indices = torch.arange(vocab_size, dtype=torch.long, device=device)
            embed_vectors = model.embedding(indices)
            tag = "default" if layer == "embedding" else "embedding"
            add_token_vectors(tag, embed_vectors)

        if layer in ("head", "both"):
            add_token_vectors("lm_head", model.out.weight)


def always_true():
    while True:
        yield True


def estimate_memory(
    config: Config,
    *,
    seq_len: int,
    micro_batch_size: int,
    use_bf16: bool = True,
) -> dict:
    """Estimate peak GPU memory required for training.

    Covers model weights (fp32), gradients (fp32), AdamW optimizer state
    (fp32 momentum + variance), and per-micro-batch activations.
    Returns a breakdown in bytes.
    """
    embedding_params, body_params = compute_param_counts(config)
    total_params = embedding_params + body_params

    # fp32 master weights
    model_mem = total_params * 4

    # fp32 gradients
    grad_mem = total_params * 4

    # AdamW: fp32 momentum + fp32 variance
    optim_mem = total_params * 8

    B = micro_batch_size
    S = seq_len
    H = config.num_head
    D = config.dim
    d_k = config.dim_key
    d_v = config.dim_value
    d_ff = config.dim_ff_hidden
    n_layer = config.num_layer

    # Most activations stay in BF16 (2 bytes) under autocast;
    # attention softmax outputs are typically fp32 (4 bytes).
    act_bytes = 2 if use_bf16 else 4

    # Embedding lookup output
    embed_act = B * S * D * act_bytes

    # Per transformer block activations saved for backward:
    #   norm1 input (residual)         B*S*D
    #   Q, K, V                        3 * B*H*S*d_k
    #   attention softmax output       B*H*S*S  (fp32)
    #   pre-W_o (attention output)     B*H*S*d_v
    #   post-W_o (residual)            B*S*D
    #   norm2 input (residual)         B*S*D
    #   FF hidden (post-activation)    B*S*d_ff
    attn_score_bytes = 4  # softmax upcasts to fp32
    per_block = (
        B * S * D * act_bytes
        + 3 * B * H * S * d_k * act_bytes
        + B * H * S * S * attn_score_bytes
        + B * H * S * d_v * act_bytes
        + B * S * D * act_bytes
        + B * S * D * act_bytes
        + B * S * d_ff * act_bytes
    )

    all_blocks = n_layer * per_block

    # LM head logits (fp32 for cross-entropy)
    logit_act = B * S * config.vocab_size * 4

    peak_act = embed_act + all_blocks + logit_act

    total = int(model_mem + grad_mem + optim_mem + peak_act)

    return {
        "total_params": total_params,
        "embedding_params": embedding_params,
        "body_params": body_params,
        "model_mem": model_mem,
        "grad_mem": grad_mem,
        "optim_mem": optim_mem,
        "activation_mem": int(peak_act),
        "total": total,
    }


def _report_memory_estimate(
    result: dict,
    seq_len: int,
    batch_size: int,
    micro_batches: int,
    use_bf16: bool,
) -> None:
    """Print a human-readable GPU memory estimate."""
    dtype_label = "BF16" if use_bf16 else "FP32"
    log_info("==== Memory Estimate ====")
    print(
        f"Parameters:  {result['total_params']:,} "
        f"(embedding: {result['embedding_params']:,}, "
        f"body: {result['body_params']:,})"
    )
    print(f"Precision:   {dtype_label} activations, FP32 master weights")
    print(
        f"Batch size:  {batch_size} (micro_batch_size={batch_size // micro_batches}"
        f"{' x ' + str(micro_batches) + ' micro-batches' if micro_batches > 1 else ''})"
    )
    print(f"Seq length:  {seq_len}")
    print()

    def fmt_mib(b: int) -> str:
        return f"{b / (1024 * 1024):,.1f} MiB"

    def fmt_gib(b: int) -> str:
        return f"{b / (1024 * 1024 * 1024):,.2f} GiB"

    print(
        f"  Model weights:     {fmt_mib(result['model_mem']):>10s}  ({fmt_gib(result['model_mem'])})"
    )
    print(
        f"  Gradients:         {fmt_mib(result['grad_mem']):>10s}  ({fmt_gib(result['grad_mem'])})"
    )
    print(
        f"  Optimizer (AdamW): {fmt_mib(result['optim_mem']):>10s}  ({fmt_gib(result['optim_mem'])})"
    )
    print(
        f"  Activations:       {fmt_mib(result['activation_mem']):>10s}  ({fmt_gib(result['activation_mem'])})"
    )
    print("  ─────────────────────────────")
    print(
        f"  Estimated total:   {fmt_mib(result['total']):>10s}  ({fmt_gib(result['total'])})"
    )


def log_step_callback(
    info: StepInfo,
    state: LogState,
    train_info: TrainInfo,
    *,
    log_period: int,
    model: Model,
    val_dataset: pq.ParquetFile | None,
    val_batches: int,
    seq_len: int,
    batch_size: int,
    pad_index: int,
    device: str,
    eot_index: int | None,
    sot_index: int | None,
    use_bf16: bool = True,
    writer: SummaryWriter | None,
    on_save: Callable[[], None],
    on_backup: Callable[[], None],
) -> None:
    token_count = train_info.token_count

    if writer is not None:
        writer.add_scalar("loss/train", info.loss, token_count)
        writer.add_scalar("metric/time/elapsed", train_info.time, token_count)

    state.sum_loss += info.loss * info.tokens_non_pad
    state.sum_time += info.time
    state.tokens += info.tokens
    state.tokens_non_pad += info.tokens_non_pad

    step = info.step + 1

    if step % log_period == 0:
        avg_loss = state.sum_loss / max(state.tokens_non_pad, 1)
        avg_time = state.sum_time / log_period

        val_metrics = None
        val_time = 0.0
        if val_dataset is not None:
            t_val_start = time.perf_counter()

            val_loss, val_entropy, val_actual = compute_validation_metrics(
                model=model,
                dataset=val_dataset,
                seq_len=seq_len,
                batch_size=batch_size,
                pad_index=pad_index,
                device=device,
                num_batches=val_batches,
                eot_index=eot_index,
                sot_index=sot_index,
                use_bf16=use_bf16,
            )

            if val_actual < val_batches and not state.val_exhausted_warned:
                log_warning(
                    f"Validation dataset exhausted early "
                    f"({val_actual}/{val_batches} batches)."
                )
                state.val_exhausted_warned = True

            val_metrics = (val_loss, val_entropy)

            if device == "cuda":
                torch.cuda.synchronize()
            val_time = time.perf_counter() - t_val_start

        # TensorBoard metrics
        if writer is not None:
            writer.add_scalar("metric/time/avg", avg_time, token_count)
            if val_metrics is not None:
                writer.add_scalar("metric/time/val", val_time, token_count)

        # Console output
        if val_metrics is not None:
            val_loss, val_entropy = val_metrics
            parts = [
                f"  {step}: avg_loss={avg_loss:.6f}, val_loss={val_loss:.6f}, val_entropy={val_entropy:.6f}",
                f"avg_time={avg_time:.6f}, val_time={val_time:.6f}",
            ]
            print(", ".join(parts))
            if writer is not None:
                writer.add_scalar("loss/val", val_loss, token_count)
                writer.add_scalar("entropy/val", val_entropy, token_count)
        else:
            parts = [f"  {step}: avg_loss={avg_loss:.6f}", f"avg_time={avg_time:.6f}"]
            print(", ".join(parts))

        if step % (log_period * 10) == 0:
            non_pad = state.tokens_non_pad
            total = state.tokens
            if total > 0:
                ratio = non_pad / total
                print(f"non_pad_ratio={ratio:.6f} (non_pad={non_pad}, total={total})")

            on_save()

            if step % (log_period * 100) == 0:
                on_backup()

        state.sum_loss = 0.0
        state.sum_time = 0.0
        state.tokens = 0
        state.tokens_non_pad = 0


#### actions ####


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


def action_embed(
    path: StrOrPath,
    tensorboard_dir: str,
    *,
    layer: str = "embedding",
    device: str = "cpu",
):
    project_info = load_project(path)
    model_state_file = resolve_parent(path) / project_info.model_state_file
    tokenizer_file = resolve_parent(path) / project_info.tokenizer

    model = Model(project_info.config).to(device)
    log_info("Loading model state.")
    load_model_state(model_state_file, model, device=device)

    tokenizer = load_tokenizer(tokenizer_file)

    log_info("Opening TensorBoard writer.")
    writer = SummaryWriter(tensorboard_dir)
    writer_add_embedding(writer, model, tokenizer, layer=layer, device=device)
    writer.close()
    log_success(f"Embedding projection written to {tensorboard_dir}.")


def action_gen(
    path: StrOrPath,
    seq_len: int,
    prompt: str,
    count: int = -1,
    temperature: float = 0.5,
    *,
    device: str = "cpu",
    stream: bool = False,
    show_all: bool = False,
    stop_token: int = EOT,
    add_sot: bool = False,
    dump_file: Optional[str] = None,
    random_seed: Optional[int] = None,
):
    assert seq_len > 0
    assert temperature > 0

    if not add_sot:
        assert len(prompt) > 0

    project_info = load_project(path)

    model = Model(project_info.config).to(device)

    load_model_state(
        resolve_parent(path) / project_info.model_state_file, model, device=device
    )

    tokenizer = load_tokenizer(resolve_parent(path) / project_info.tokenizer)

    prompt_tokens = tokenizer.encode(prompt).ids

    if add_sot:
        prompt_tokens = [SOT] + prompt_tokens

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

    print("Prompt:", repr(tokenizer.decode(tokens, skip_special_tokens=False)))
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

    if stream and show_all:
        print(tokenizer.decode(tokens, skip_special_tokens=False), end="")

    for _ in range(count) if count >= 0 else always_true():
        input = tokens[-seq_len:]

        # Keep generation at a fixed shape so torch.compile does not recompile
        # as the prompt grows. Right padding preserves RoPE positions for real
        # tokens because the selected logit is still at length - 1.
        length = len(input)
        if length < seq_len:
            input = input + [PAD] * (seq_len - length)

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
        if show_all:
            print(tokenizer.decode(tokens, skip_special_tokens=False))
        else:
            print(tokenizer.decode(tokens[start_pos:], skip_special_tokens=False))

    log_info("********")
    print("Generated token count:", len(tokens) - start_pos)

    if dump_f is not None:
        dump_f.close()


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


def action_info(path: StrOrPath):
    project_info = load_project(path)

    log_info("==== Project Info ====")
    print("Project file:", path)
    rich_print(project_info)

    embedding_params, body_params = compute_param_counts(project_info.config)
    total = embedding_params + body_params
    log_info("==== Parameter Counts ====")
    print(f"Embedding (in+out): {embedding_params:,}")
    print(f"Body (transformer):  {body_params:,}")
    print(f"Total:               {total:,}")


def action_init(path: StrOrPath):
    config = Config(
        vocab_size=8192,
        max_seq_len=4096,
        num_layer=4,
        num_head=4,
        dim=256,
        dim_key=64,
        dim_value=64,
        dim_ff_hidden=1024,
        use_rope=True,
        ff_activation="relu",
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


def action_model_init(path: StrOrPath):
    project_info = load_project(path)
    model_state_file = resolve_parent(path) / project_info.model_state_file
    train_config = project_info.train_config
    optimizer_state_file = resolve_parent(path) / train_config.optimizer_state_file

    model = Model(project_info.config)
    save_model_state(model_state_file, model)
    log_success(f"Initialized model state at {model_state_file}.")

    optimizer = create_optimizer(
        model, train_config.learning_rate, train_config.weight_decay
    )
    save_optimizer_state(optimizer_state_file, optimizer)
    log_success(f"Initialized optimizer state at {optimizer_state_file}.")

    project_info.train_info = TrainInfo()

    save_project(path, project_info)


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
    val_batches: Optional[int] = None,
    log_period: int = 10,
    tensorboard_dir: Optional[str] = None,
    device: str = "cpu",
    start_index: Optional[int] = None,
    start_offset: Optional[int] = None,
    add_eot: bool = False,
    add_sot: bool = False,
    micro_batches: Optional[int] = None,
    loop_dataset: bool = False,
    use_bf16: bool = True,
    estimate: bool = False,
):
    assert log_period > 0

    # === Load project & resolve paths ===

    project_info = load_project(path)
    model_state_file = resolve_parent(path) / project_info.model_state_file
    train_config = project_info.train_config
    optimizer_state_file = resolve_parent(path) / train_config.optimizer_state_file

    # === Parse & validate configuration ===

    def _apply[T](attr: str, value: T | None) -> T:
        if value is not None:
            setattr(train_config, attr, value)
            return value
        v = getattr(train_config, attr)
        return v

    seq_len = _apply("seq_len", seq_len)
    assert seq_len > 0

    prev_batch_size = train_config.batch_size
    batch_size = _apply("batch_size", batch_size)
    assert batch_size > 0
    should_reset_optimizer = batch_size != prev_batch_size

    learning_rate = _apply("learning_rate", learning_rate)
    weight_decay = _apply("weight_decay", weight_decay)
    micro_batches = _apply("micro_batches", micro_batches) or 1
    assert micro_batches >= 1
    assert batch_size % micro_batches == 0, (
        f"batch_size ({batch_size}) must be divisible by "
        f"micro_batches ({micro_batches})"
    )

    micro_batch_size = batch_size // micro_batches

    if estimate:
        result = estimate_memory(
            project_info.config,
            seq_len=seq_len,
            micro_batch_size=micro_batch_size,
            use_bf16=use_bf16,
        )
        _report_memory_estimate(result, seq_len, batch_size, micro_batches, use_bf16)
        tokens_per_step = seq_len * batch_size
        total_tokens = tokens_per_step * steps
        log_info("==== Token Throughput ====")
        print(f"Tokens per step: {tokens_per_step:,}")
        print(f"Total tokens ({steps} steps): {total_tokens:,}")
        return

    # === Load training data ===

    log_info("Loading training dataset.")
    dataset = load_dataset(data_path)

    new_id = get_dataset_id(dataset)
    old_id = project_info.train_info.dataset_id
    if new_id and new_id != old_id:
        log_info(f"Dataset changed ({old_id!r} -> {new_id!r}).")
        if start_index is None:
            project_info.train_info.index = 0
        if start_offset is None:
            project_info.train_info.offset = 0
        project_info.train_info.dataset_id = new_id

    if start_index is None:
        start_index = project_info.train_info.index
    if start_offset is None:
        start_offset = project_info.train_info.offset

    log_info(f"Start position: index={start_index}, offset={start_offset}")

    eot_index = EOT if add_eot else None
    sot_index = SOT if add_sot else None

    # Mutable state shared between iter_batches (writes progress),
    # save() (reads progress), and the exhaustion-reset path below.
    train_tracker = TrainTracker(index=start_index, offset=start_offset)

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
            sot_index=sot_index,
        )

    # === Load validation data ===

    if val_batches is None:
        val_batches = max(log_period // 10, 1)
    val_dataset = None
    if val_path is not None:
        log_info("Loading validation dataset.")
        val_dataset = load_dataset(val_path)

    # === Build model ===

    model = Model(project_info.config).to(device)
    log_info("Loading model state.")
    load_model_state(model_state_file, model, device=device)

    # === Build optimizer ===

    optimizer = create_optimizer(model, learning_rate, weight_decay)
    if should_reset_optimizer:
        log_info("Batch size changed, optimizer reset.")
    elif optimizer_state_file.exists():
        log_info("Loading optimizer state.")
        load_optimizer_state(optimizer_state_file, optimizer)
    else:
        log_info("Optimizer state file not found, using fresh optimizer.")

    if use_bf16 and device == "cpu":
        log_warning(
            "BF16 is enabled but the device is CPU. "
            "BF16 is typically not supported on CPU and may cause errors or poor performance. "
            "Use -n16 / --no-bf16 to disable BF16."
        )

    # === TensorBoard writer ===

    writer = None
    if tensorboard_dir is not None:
        writer = SummaryWriter(tensorboard_dir)

    # === Helper functions ===

    def save():
        project_info.train_info.index = train_tracker.index
        project_info.train_info.offset = train_tracker.offset
        log_info(f"Saving. (train_info: {project_info.train_info})")
        save_project(path, project_info)
        save_model_state(model_state_file, model)
        save_optimizer_state(optimizer_state_file, optimizer)

    def backup():
        log_info("Making backups.")
        shutil.copyfile(path, get_backup_path(path))
        shutil.copyfile(model_state_file, get_backup_path(model_state_file))
        shutil.copyfile(optimizer_state_file, get_backup_path(optimizer_state_file))

    state = LogState()

    def step_callback(info: StepInfo):
        log_step_callback(
            info,
            state,
            project_info.train_info,
            log_period=log_period,
            model=model,
            val_dataset=val_dataset,
            val_batches=val_batches,
            seq_len=seq_len,
            batch_size=micro_batch_size,
            pad_index=PAD,
            device=device,
            eot_index=eot_index,
            sot_index=sot_index,
            use_bf16=use_bf16,
            writer=writer,
            on_save=save,
            on_backup=backup,
        )

    # === Log training configuration ===

    dataset_id = project_info.train_info.dataset_id

    # console (include file paths for troubleshooting)
    val_info = f"{val_batches} batches"
    if val_path is not None:
        val_info += f", dataset={val_path!r}"

    data_msg = f"{data_path!r}"
    if dataset_id:
        data_msg += f", dataset_id={dataset_id!r}"
    data_msg += f", start_index={start_index}, start_offset={start_offset}"

    log_info("==== Training Configuration ====")
    print(f"Project:       {path}")
    print(f"Data:          {data_msg}")
    print(f"Steps:         {steps}")
    print(f"Seq length:    {seq_len}")
    if micro_batches > 1:
        print(
            f"Batch size:    {batch_size} ({micro_batches} micro-batches x {micro_batch_size})"
        )
    else:
        print(f"Batch size:    {batch_size}")
    print(f"Learning rate: {learning_rate}, weight decay: {weight_decay}")
    print(f"BF16 autocast: {use_bf16}")
    print(f"Loop dataset:  {loop_dataset}")
    print(f"SOT/EOT:       {add_sot}/{add_eot}")
    print(f"Validation:    {val_info}")
    print(f"Time:          {datetime.now()}")
    log_info("==============================")

    if writer is not None:
        device_info: dict = {"type": device}
        if device == "cuda":
            device_info["name"] = torch.cuda.get_device_name()
            free_mem, total_mem = torch.cuda.mem_get_info(device)
            device_info["gpu_mem_free"] = free_mem
            device_info["gpu_mem_total"] = total_mem

        dataset_info: dict = {
            "index": start_index,
            "offset": start_offset,
        }
        if dataset_id:
            dataset_info["id"] = dataset_id

        writer_val_info: dict = {"batches": val_batches}
        if val_dataset is not None:
            val_id = get_dataset_id(val_dataset)
            if val_id:
                writer_val_info["dataset_id"] = val_id

        log_entry: dict = {
            "event": "train_start",
            "time": datetime.now().isoformat(),
            "host": socket.gethostname(),
            "steps": steps,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "bf16": use_bf16,
            "loop_dataset": loop_dataset,
            "marker": {"sot": add_sot, "eot": add_eot},
            "device": device_info,
            "dataset": dataset_info,
            "validation": writer_val_info,
        }
        if micro_batches > 1:
            log_entry["micro_batches"] = micro_batches
        writer.add_text(
            "train/log",
            json.dumps(log_entry),
            project_info.train_info.token_count,
        )

    # === Run training loop ===

    t_start = time.perf_counter()
    backup()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    steps_remaining = steps
    steps_done = 0
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
            step_offset=steps_done,
            use_bf16=use_bf16,
        )
        steps_remaining -= actual_steps
        steps_done += actual_steps
        if steps_remaining > 0:
            # Dataset exhausted before reaching the requested step count.
            if loop_dataset:
                log_info("Dataset exhausted - restarting from beginning.")
                train_tracker.index = 0
                train_tracker.offset = 0
                start_index = 0
                start_offset = 0
            else:
                log_info("Dataset exhausted - stopping early.")
                project_info.train_info.index = 0
                project_info.train_info.offset = 0
                project_info.train_info.dataset_id = ""
                break

    # === Final save & report ===

    save()

    t_end = time.perf_counter()
    elapsed = t_end - t_start

    if device == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
    else:
        peak_allocated = 0
        peak_reserved = 0

    log_info(f"Training completed. (time: {elapsed:.6f})")
    log_info(
        f"Reached sample {train_tracker.index}, offset {train_tracker.offset} in the dataset."
    )

    if device == "cuda":
        log_info(
            f"Peak GPU memory: "
            f"{peak_allocated / (1024**3):.2f} GiB allocated, "
            f"{peak_reserved / (1024**3):.2f} GiB reserved"
        )

    if writer is not None:
        end_entry: dict = {
            "event": "train_end",
            "elapsed": round(elapsed, 6),
            "dataset": {
                "index": train_tracker.index,
                "offset": train_tracker.offset,
            },
        }
        if device == "cuda":
            end_entry["gpu"] = {
                "peak_allocated": peak_allocated,
                "peak_reserved": peak_reserved,
            }
        writer.add_text(
            "train/log",
            json.dumps(end_entry),
            project_info.train_info.token_count,
        )
        writer.close()


def action_validate(
    path: StrOrPath,
    val_path: str,
    num_batches: int,
    *,
    seq_len: Optional[int] = None,
    batch_size: Optional[int] = None,
    start_index: int = 0,
    start_offset: int = 0,
    add_eot: bool = False,
    add_sot: bool = False,
    use_bf16: bool = True,
):
    assert num_batches > 0

    project_info = load_project(path)
    model_state_file = resolve_parent(path) / project_info.model_state_file
    train_config = project_info.train_config

    if seq_len is None:
        seq_len = train_config.seq_len
    if seq_len <= 0:
        raise ValueError(
            "seq_len is not configured in the project; specify -l / --seq-len"
        )

    if batch_size is None:
        micro_batches = train_config.micro_batches or 1
        batch_size = train_config.batch_size // micro_batches
    if batch_size <= 0:
        raise ValueError(
            "batch_size is not configured in the project; specify -bs / --batch-size"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model(project_info.config).to(device)
    log_info("Loading model state.")
    load_model_state(model_state_file, model, device=device)

    log_info("Loading validation dataset.")
    dataset = load_dataset(val_path)

    eot_index = EOT if add_eot else None
    sot_index = SOT if add_sot else None

    log_info("Running validation.")
    t_start = time.perf_counter()

    avg_loss, avg_entropy, val_actual = compute_validation_metrics(
        model=model,
        dataset=dataset,
        seq_len=seq_len,
        batch_size=batch_size,
        pad_index=PAD,
        device=device,
        num_batches=num_batches,
        eot_index=eot_index,
        sot_index=sot_index,
        start_index=start_index,
        start_offset=start_offset,
        use_bf16=use_bf16,
    )

    if val_actual < num_batches:
        log_warning(
            f"Validation dataset exhausted early "
            f"({val_actual}/{num_batches} batches)."
        )

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_start

    perplexity = math.exp(avg_loss)

    log_info("==== Validation Results ====")
    print(f"Batches:       {num_batches}")
    print(f"Batch size:    {batch_size}")
    print(f"Seq length:    {seq_len}")
    print(f"Start index:   {start_index}")
    print(f"Start offset:  {start_offset}")
    print(f"SOT/EOT:       {add_sot}/{add_eot}")
    print(f"BF16:          {use_bf16}")
    print(f"Device:        {device}")
    print(f"Time:          {elapsed:.4f}s")
    print()
    print(f"Loss:          {avg_loss:.6f}")
    print(f"Entropy:       {avg_entropy:.6f}")
    print(f"Perplexity:    {perplexity:.4f}")


#### main ####


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="V1 Model CLI Tool (PyTorch)", fromfile_prefix_chars="@"
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

    # dump_param
    dump_param_parser = subparsers.add_parser("dump_param", help="dump parameters")
    dump_param_parser.add_argument("path", help="project file")

    # embed
    embed_parser = subparsers.add_parser(
        "embed", help="write embedding projection to TensorBoard"
    )
    embed_parser.add_argument("path", help="project file")
    embed_parser.add_argument("tensorboard_dir", help="TensorBoard write directory")
    embed_parser.add_argument(
        "--layer",
        choices=("embedding", "head", "both"),
        default="embedding",
        help="token vectors to visualize (default: embedding)",
    )

    # gen
    gen_parser = subparsers.add_parser("gen", help="generate text")
    gen_parser.add_argument("path", help="project file")
    gen_parser.add_argument("seq_len", type=int, help="sequence length")
    gen_parser.add_argument("prompt", help="prompt from which to start generating")
    gen_parser.add_argument(
        "-c",
        "--count",
        type=int,
        default=-1,
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
        dest="show_all",
        help="show prompt and generated text together",
    )
    gen_parser.add_argument(
        "-st",
        "--stop-token",
        type=int,
        default=EOT,
        help="stop token id (-1: never stop)",
    )
    gen_parser.add_argument("--sot", action="store_true", help="add SOT")
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

    # graph
    graph_parser = subparsers.add_parser(
        "graph", help="write model graph to TensorBoard"
    )
    graph_parser.add_argument("path", help="project file")
    graph_parser.add_argument("tensorboard_dir", help="TensorBoard write directory")
    graph_parser.add_argument(
        "seq_len", type=int, help="sequence length for dummy input"
    )

    # info
    info_parser = subparsers.add_parser("info", help="show info")
    info_parser.add_argument("path", help="project file")

    # init
    init_parser = subparsers.add_parser("init", help="initialize a new project")
    init_parser.add_argument("path", help="project file")

    # model_init
    model_init_parser = subparsers.add_parser(
        "model_init",
        help="initialize model accroding to configuration and fill with random weights",
    )
    model_init_parser.add_argument("path", help="project file")

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
        "--sot",
        action="store_true",
        help="prepend SOT to each training/validation text (effective length +1)",
    )
    train_parser.add_argument(
        "--eot",
        action="store_true",
        help="append EOT to each training/validation text (effective length +1)",
    )
    train_parser.add_argument(
        "-mb",
        "--micro-batches",
        type=int,
        default=None,
        dest="micro_batches",
        help="split each step into N micro-batches with accumulated gradients (default: 1)",
    )
    train_parser.add_argument(
        "-ld",
        "--loop-dataset",
        action="store_true",
        help="restart from the beginning when the dataset is exhausted instead of stopping early",
    )
    train_parser.add_argument(
        "-vb",
        "--validation-batches",
        type=int,
        default=None,
        help="number of batches for validation (default: log_period // 10, at least 1)",
    )
    train_parser.add_argument(
        "-n16",
        "--no-bf16",
        action="store_true",
        help="disable bfloat16 autocast (useful for CPU or consumer GPUs that don't support bf16)",
    )
    train_parser.add_argument(
        "--estimate",
        action="store_true",
        help="estimate and report GPU memory required for training, then exit without training",
    )

    # validate
    validate_parser = subparsers.add_parser("validate", help="validate model")
    validate_parser.add_argument("path", help="project file")
    validate_parser.add_argument("val_data", help="validation dataset path")
    validate_parser.add_argument(
        "num_batches",
        type=int,
        help="number of validation batches",
    )
    validate_parser.add_argument(
        "-l",
        "--seq-len",
        type=int,
        default=None,
        help="sequence length (default: from training config)",
    )
    validate_parser.add_argument(
        "-bs",
        "--batch-size",
        type=int,
        default=None,
        help="batch size for validation (default: micro-batch size from training config)",
    )
    validate_parser.add_argument(
        "-si",
        "--start-index",
        type=int,
        default=0,
        help="start validation from this sample index in the dataset (default: 0)",
    )
    validate_parser.add_argument(
        "-so",
        "--start-offset",
        type=int,
        default=0,
        help="initial window offset for the first sample (default: 0)",
    )
    validate_parser.add_argument(
        "--sot",
        action="store_true",
        help="prepend SOT to each validation text",
    )
    validate_parser.add_argument(
        "--eot",
        action="store_true",
        help="append EOT to each validation text",
    )
    validate_parser.add_argument(
        "-n16",
        "--no-bf16",
        action="store_true",
        help="disable bfloat16 autocast",
    )

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    torch.set_printoptions(linewidth=160)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        device_name = torch.cuda.get_device_name()
        log_info(f"Using device: {device} ({device_name})")
        free_mem, total_mem = torch.cuda.mem_get_info(device)
        log_info(
            f"GPU memory: {free_mem / (1024**3):.2f} GiB free / "
            f"{total_mem / (1024**3):.2f} GiB total"
        )
    else:
        log_info(f"Using device: {device}")

    log_info(f"Host: {socket.gethostname()}")

    num_threads = 8
    if args.thread is not None:
        num_threads = args.thread
    else:
        num_threads = os.cpu_count() or 8

    torch.set_num_threads(num_threads)
    log_info(f"PyTorch will use {num_threads} CPU thread(s).")

    # CUDA
    if device == "cuda":
        torch.set_float32_matmul_precision(args.float32_precision)
        log_info(f"Float32 matmul precision: {args.float32_precision}")

    match args.action:
        case "dump_param":
            action_dump_param(args.path)
        case "embed":
            action_embed(
                args.path, args.tensorboard_dir, layer=args.layer, device=device
            )
        case "gen":
            action_gen(
                args.path,
                seq_len=args.seq_len,
                prompt=args.prompt,
                count=args.count,
                temperature=args.temp,
                device=device,
                stream=args.stream,
                show_all=args.show_all,
                stop_token=args.stop_token,
                add_sot=args.sot,
                dump_file=args.dump_file,
                random_seed=args.random_seed,
            )
        case "graph":
            action_graph(args.path, args.tensorboard_dir, args.seq_len, device=device)
        case "info":
            action_info(args.path)
        case "init":
            action_init(args.path)
        case "model_init":
            action_model_init(args.path)
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
                val_batches=args.validation_batches,
                device=device,
                tensorboard_dir=args.tensorboard_dir,
                log_period=args.log_period,
                start_index=args.start_index,
                start_offset=args.start_offset,
                add_eot=args.eot,
                add_sot=args.sot,
                micro_batches=args.micro_batches,
                loop_dataset=args.loop_dataset,
                use_bf16=not args.no_bf16,
                estimate=args.estimate,
            )
        case "validate":
            action_validate(
                args.path,
                args.val_data,
                args.num_batches,
                seq_len=args.seq_len,
                batch_size=args.batch_size,
                start_index=args.start_index,
                start_offset=args.start_offset,
                add_eot=args.eot,
                add_sot=args.sot,
                use_bf16=not args.no_bf16,
            )
        case _:
            raise ValueError(f"unrecognized action: {args.action}")


if __name__ == "__main__":
    main()
