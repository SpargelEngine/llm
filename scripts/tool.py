import os
import shutil
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

import torch
from datasets import Dataset, load_from_disk
from pydantic import BaseModel, NonNegativeInt
from rich import print as rich_print
from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.utils.tensorboard import SummaryWriter

from spargel_llm.tools.logging import log_info, log_success
from spargel_llm.tools.typing import StrOrPath
from spargel_llm.tools.utils import (
    PromptAbortError,
    prompt_overwrite,
)
from spargel_llm.v1.torch.model import Config, Model
from spargel_llm.v1.torch.utils import StepInfo, TrainInfo, generate_step, train

PAD, EOT = 1, 2


class TrainConfig(BaseModel):
    seq_len: NonNegativeInt
    batch_size: NonNegativeInt
    learning_rate: float
    weight_decay: float
    optimizer_state_file: str


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


def load_dataset(path: StrOrPath) -> Dataset:
    """Load a HuggingFace dataset directory.

    The returned ``Dataset`` uses memory-mapped Arrow tables so the data
    is not fully loaded into memory.
    """
    path = Path(path).resolve()
    dataset = load_from_disk(str(path))
    log_info(f"Loaded dataset from {path} ({len(dataset):,} rows).")
    return dataset


def iter_samples(
    dataset: Dataset,
    seq_len: int,
    pad_index: int,
    offset: int = 0,
    stride: int | None = None,
) -> Iterator[tuple[list[int], list[bool], list[int]]]:
    """Iterate through a dataset sequentially and yield individual samples.

    For each row, a window of ``seq_len + 1`` tokens slides from *offset*
    with the given *stride* (default: ``seq_len + 1``).  Each window
    produces one sample: ``(input_ids, mask, target_ids)`` as plain Python
    lists.  Short tails are padded with *pad_index*.
    """
    if stride is None:
        stride = seq_len + 1

    for item in dataset:
        tokens: list[int] = item["tokens"]
        if tokens is None or len(tokens) <= 1:
            continue

        pos = offset
        while pos + 1 < len(tokens):
            end = min(pos + seq_len + 1, len(tokens))
            segment = tokens[pos:end]
            L = len(segment) - 1

            input_ids = segment[:L] + [pad_index] * (seq_len - L)
            mask = [False] * L + [True] * (seq_len - L)
            target_ids = segment[1 : L + 1] + [pad_index] * (seq_len - L)

            yield input_ids, mask, target_ids

            pos += stride


def make_batches(
    samples: Iterator[tuple[list[int], list[bool], list[int]]],
    batch_size: int,
) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
    """Accumulate samples into tensor batches.

    Samples are collected into a buffer; when the buffer reaches
    *batch_size* the lists are converted to tensors, stacked, and yielded.
    At the end any partial buffer is silently discarded — training stops
    early.
    """
    buffer: list[tuple[Tensor, Tensor, Tensor]] = []

    for input_ids, mask, target_ids in samples:
        buffer.append(
            (
                torch.tensor(input_ids, dtype=torch.int),
                torch.tensor(mask, dtype=torch.bool),
                torch.tensor(target_ids, dtype=torch.int),
            )
        )

        if len(buffer) == batch_size:
            yield (
                torch.stack([x[0] for x in buffer]),
                torch.stack([x[1] for x in buffer]),
                torch.stack([x[2] for x in buffer]),
            )
            buffer.clear()


def iter_batches(
    dataset: Dataset,
    seq_len: int,
    batch_size: int,
    pad_index: int,
    offset: int = 0,
    stride: int | None = None,
) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
    """Convenience wrapper: ``make_batches(iter_samples(...))``."""
    return make_batches(
        iter_samples(dataset, seq_len, pad_index, offset, stride), batch_size
    )


def get_backup_path(path: StrOrPath):
    return (
        resolve_parent(path)
        / f".{Path(path).resolve().name}.{int(datetime.now().timestamp())}"
    )


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
    dataset: Dataset,
    seq_len: int,
    batch_size: int,
    pad_index: int,
    device: str,
    num_batches: int = 10,
) -> float:
    assert num_batches > 0

    total_loss = 0.0

    def make_iterator():
        return iter_batches(dataset, seq_len, batch_size, pad_index)

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

    tokens = list(prompt_tokens)

    print("Prompt:", repr(tokenizer.decode(tokens)))
    print("Prompt token count:", len(tokens))
    print("Sequence length:", seq_len)
    print("Temperature:", temperature)
    print("Max generation count:", count)
    print("Stop token id:", stop_token)

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


def action_train(
    path: StrOrPath,
    steps: int,
    data_path: str,
    *,
    seq_len: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    weight_decay: Optional[float] = None,
    val_path: Optional[str] = None,
    log_period: int = 10,
    tensorboard_dir: Optional[str] = None,
    device: str = "cpu",
):
    assert log_period > 0

    project_info = load_project(path)
    model_state_file = resolve_parent(path) / project_info.model_state_file
    tokenizer_file = resolve_parent(path) / project_info.tokenizer
    train_config = project_info.train_config
    optimizer_state_file = resolve_parent(path) / train_config.optimizer_state_file

    should_reset_optimizer = (
        batch_size is not None and batch_size != train_config.batch_size
    )

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

    # data

    tokenizer = load_tokenizer(tokenizer_file)

    log_info("Loading training dataset.")
    dataset = load_dataset(data_path)

    # show a few samples as example
    print("Data example:")
    sample_iter = iter_samples(dataset, seq_len, PAD)
    for i, (input_ids, mask, target_ids) in enumerate(sample_iter):
        if i >= 3:
            break
        print(f"  input_ids  {input_ids!r}")
        print(f"  mask       {mask!r}")
        print(f"  target_ids {target_ids!r}")
        print(f"  decoded `{tokenizer.decode(input_ids)!r}`")

    batch_iterator = iter_batches(dataset, seq_len, batch_size, PAD)

    val_batches = max(log_period // 10, 1)
    val_dataset = None
    if val_path is not None:
        log_info("Loading validation dataset.")
        val_dataset = load_dataset(val_path)

        print("Validation data example:")
        val_sample_iter = iter_samples(val_dataset, seq_len, PAD)
        for i, (input_ids, mask, target_ids) in enumerate(val_sample_iter):
            if i >= 3:
                break
            print(f"  input   {input_ids!r}")
            print(f"  mask    {mask!r}")
            print(f"  target  {target_ids!r}")
            print(f"  decoded `{tokenizer.decode(input_ids)!r}`")

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

    # TensorBoard

    writer = None
    if tensorboard_dir is not None:
        log_info("Opening TensorBoard writer.")
        writer = SummaryWriter(tensorboard_dir)

        # show graph
        dummy_iter = iter_batches(dataset, seq_len, 1, PAD)
        dummy_input, dummy_mask, _ = next(dummy_iter)
        writer.add_graph(model, (dummy_input.to(device), dummy_mask.to(device)))

        # show input word embedding
        writer_add_embedding(writer, model, tokenizer, device=device)

    # helper functions

    def log_important(msg: str):
        log_info(msg)
        if writer is not None:
            writer.add_text("train/log", msg, project_info.train_info.token_count)

    def save():
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
        "sum_time_transfer_batch": 0.0,
        "sum_time_forward": 0.0,
        "sum_time_backward": 0.0,
    }

    def step_callback(info: StepInfo):
        token_count = project_info.train_info.token_count

        if writer is not None:
            writer.add_scalar("loss/train", info.loss, token_count)

        state["sum_loss"] += info.loss
        state["sum_time_load_batch"] += info.time_load_batch
        state["sum_time_transfer_batch"] += info.time_transfer_batch
        state["sum_time_forward"] += info.time_forward
        state["sum_time_backward"] += info.time_backward

        step = info.step + 1

        if step % log_period == 0:
            avg_loss = state["sum_loss"] / log_period
            avg_time_load_batch = state["sum_time_load_batch"] / log_period
            avg_time_transfer_batch = state["sum_time_transfer_batch"] / log_period
            avg_time_forward = state["sum_time_forward"] / log_period
            avg_time_backward = state["sum_time_backward"] / log_period
            avg_time = (
                avg_time_load_batch
                + avg_time_transfer_batch
                + avg_time_forward
                + avg_time_backward
            )

            state["sum_loss"] = 0.0
            state["sum_time_load_batch"] = 0.0
            state["sum_time_transfer_batch"] = 0.0
            state["sum_time_forward"] = 0.0
            state["sum_time_backward"] = 0.0

            val_loss = None
            if val_dataset is not None:
                val_loss = compute_validation_loss(
                    model=model,
                    dataset=val_dataset,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    pad_index=PAD,
                    device=device,
                    num_batches=val_batches,
                )

            time_log_msg = f"avg_time={avg_time:.6f} (({avg_time_load_batch:.6f} + {avg_time_transfer_batch:.6f}) + ({avg_time_forward:.6f} + {avg_time_backward:.6f}))"
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

                    if writer is not None:
                        writer_add_embedding(writer, model, tokenizer, device=device)

    # train

    t_start = time.perf_counter()

    log_important(
        f"Training for {steps} steps (seq_len={seq_len}, batch_size={batch_size}). Time: {datetime.now()}"
    )

    backup()

    train(
        info=project_info.train_info,
        model=model,
        optimizer=optimizer,
        batch_iterator=batch_iterator,
        pad_index=PAD,
        batch_size=batch_size,
        steps=steps,
        device=device,
        step_callback=step_callback,
    )

    save()

    t_end = time.perf_counter()

    log_important(f"Training completed. (time: {t_end - t_start:.6f})")

    if writer is not None:
        writer_add_embedding(writer, model, tokenizer, device=device)

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

    # train
    train_parser = subparsers.add_parser("train", help="train model")
    train_parser.add_argument("path", help="project file")
    train_parser.add_argument("steps", type=int, help="number of steps")
    train_parser.add_argument("data", help="dataset directory")
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

    # model_init
    model_init_parser = subparsers.add_parser(
        "model_init",
        help="initialize model accroding to configuration and fill with random weights",
    )
    model_init_parser.add_argument("path", help="project file")

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
            )
        case "train":
            action_train(
                args.path,
                steps=args.steps,
                data_path=args.data,
                seq_len=args.seq_len,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                val_path=args.val,
                device=device,
                tensorboard_dir=args.tensorboard_dir,
                log_period=args.log_period,
            )
        case "model_init":
            action_model_init(args.path, yes=args.yes)
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
