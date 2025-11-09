import os
import sys
import time
from argparse import ArgumentParser
from codecs import getincrementaldecoder
from datetime import datetime
from enum import Enum
from os.path import basename, dirname
from pathlib import Path
from typing import cast

import torch
from pydantic import BaseModel
from torch import Tensor
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from spargel_llm.data import (
    DataSource,
    SeqDataSource,
    SliceDataSource,
    WeightedDataSource,
)
from spargel_llm.text_splitter import GPT2_SPLIT_PATTERN, RegexSplitter
from spargel_llm.tokenizer import WordTokenizer
from spargel_llm.tools.logging import log_info, log_success
from spargel_llm.tools.typing import StrOrPath
from spargel_llm.tools.utils import (
    NO_STRINGS,
    YES_STRINGS,
    PromptAbortError,
    load_gzip_pickle,
    prompt_overwrite,
)

from .model import LLM, Config
from .utils import StepInfo, TrainDataset, TrainInfo, generate_step, train

reserved_words = [
    b"<|pad|>",  # padding
    b"<|unk|>",  # unknown
    b"<|sot|>",  # start of text
    b"<|eot|>",  # end of text
]

PAD, UNK, SOT, EOT = range(len(reserved_words))

assert len(reserved_words) <= 16
for i in range(len(reserved_words), 16):
    reserved_words.append(f"<|placeholder{i}|>".encode())


class ProjectInfo(BaseModel):
    # configuration (hyper-parameters) for the model
    config: Config

    # weight file location
    weight_file: str

    # vocab file location
    vocab_file: str

    # training information & statistics
    train_info: TrainInfo


#### load/store helper functions ####


def load_project(path: StrOrPath) -> ProjectInfo:
    with open(path, "r") as f:
        return ProjectInfo.model_validate_json(f.read())


def save_project(path: StrOrPath, project_info: ProjectInfo):
    with open(path, "w") as f:
        f.write(project_info.model_dump_json(indent=2))


def load_weights(path: StrOrPath, model: LLM, *, device: str):
    model.load_state_dict(torch.load(path, weights_only=True, map_location=device))


def save_weights(path: StrOrPath, model: LLM):
    torch.save(model.state_dict(), path)


def load_optimizer_state(path: StrOrPath, optimizer: Optimizer):
    optimizer.load_state_dict(torch.load(path))


def save_optimizer_state(path: StrOrPath, optimizer: Optimizer):
    torch.save(optimizer.state_dict(), path)


#### other helpers ####


def create_tokenizer(words: list[bytes]) -> WordTokenizer:
    all_words = (
        reserved_words + words + [b"<|?|>"]
    )  # append one token due to a PyTorch bug

    return WordTokenizer(
        all_words,
        encode_blacklist=list(range(len(reserved_words))) + [len(all_words) - 1],
        unknown=UNK,
        text_splitter=RegexSplitter(GPT2_SPLIT_PATTERN),
    )


def get_backup_path(path: StrOrPath):
    return Path(dirname(path), f".{basename(path)}.{int(datetime.now().timestamp())}")


class SampleType(Enum):
    NORMAL = 0
    START = 1


def parse_data(s: str) -> tuple[str, float, SampleType]:
    parts = s.split(":")

    path = parts[0]

    if len(parts) >= 2:
        weight = float(parts[1])
        if weight <= 0:
            raise ValueError(f"weight {weight} for {path} should be > 0")
    else:
        weight = 1.0

    if len(parts) >= 3:
        match parts[2]:
            case "n":
                sample_type = SampleType.NORMAL
            case "s":
                sample_type = SampleType.START
            case _:
                raise ValueError(f"unrecognized sample type {parts[2]} for {path}")
    else:
        sample_type = SampleType.NORMAL

    return path, weight, sample_type


def create_data_source(paths: list[str], seq_len: int, mark: bool = False):
    assert len(paths) > 0

    data_sources: list[DataSource[list[int]]] = []
    weights: list[float] = []

    count_total, count_selected = 0, 0

    for path, weight, sample_type in map(parse_data, paths):
        data = cast(list[list[int]], load_gzip_pickle(path))

        # Adjust token indices since we have special tokens.
        for tokens in data:
            for i in range(len(tokens)):
                tokens[i] += len(reserved_words)

        count_total += len(data)

        match sample_type:
            case SampleType.NORMAL:
                child_data_sources = []
                child_weights = []

                for tokens in data:
                    if mark:
                        tokens = [SOT] + tokens + [EOT]

                    if len(tokens) < seq_len + 1:
                        continue

                    child_data_sources.append(SliceDataSource(tokens, seq_len + 1))
                    child_weights.append(len(tokens))

                    count_selected += 1

                if len(child_data_sources) == 0:
                    raise ValueError(
                        f"no data meet requirement in {path} (sample type: {sample_type.name})"
                    )

                data_sources.append(
                    WeightedDataSource(child_data_sources, child_weights)
                )
            case SampleType.START:
                selected = []

                for tokens in data:
                    if len(tokens) <= 0:
                        continue

                    if mark:
                        tokens = [SOT] + tokens + [EOT]

                    selected.append(tokens[: seq_len + 1])

                    count_selected += 1

                if len(selected) == 0:
                    raise ValueError(
                        f"no data meet requirement in {path} (sample type: {sample_type})"
                    )

                data_sources.append(SeqDataSource(selected))

        weights.append(weight)

    log_info(f"Sample selection ratio: {count_selected}/{count_total}.")

    return WeightedDataSource(data_sources, weights)


def writer_add_embedding(
    writer: SummaryWriter,
    model: LLM,
    tokenizer: WordTokenizer,
    *,
    device: str = "cpu",
):
    model.eval()
    with torch.no_grad():
        indices = torch.arange(len(tokenizer.words), dtype=torch.int, device=device)
        embed_vectors = model.embed(indices)

        labels = []
        for word in tokenizer.words:
            try:
                label = word.decode("utf-8")
            except UnicodeDecodeError:
                label = repr(word)
            labels.append(label if len(label.strip()) == len(label) else repr(label))

    writer.add_embedding(embed_vectors.cpu(), labels)


@torch.compile
def validation_loss_step(
    model: LLM,
    inputs: Tensor,
    masks: Tensor,
    targets: Tensor,
    pad_index: int,
):
    return model.loss(inputs, masks, targets, pad_index=pad_index)


def compute_validation_loss(
    model: LLM,
    data_source: DataSource[list[int]],
    seq_len: int,
    batch_size: int,
    pad_index: int,
    device: str,
    num_batches: int = 10,
) -> float:
    assert num_batches > 0

    total_loss = 0.0

    dataset = TrainDataset(data_source, seq_len, pad_index)
    loader = DataLoader(dataset, batch_size=batch_size)
    iterator = iter(loader)

    model.eval()

    with torch.no_grad():
        for _ in range(num_batches):
            try:
                inputs, masks, targets = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                inputs, masks, targets = next(iterator)

            loss = validation_loss_step(
                model, inputs.to(device), masks.to(device), targets, pad_index=PAD
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
    print("Config:")
    print(f"  {project_info.config}")
    print("Weight file:", project_info.weight_file)
    print("Vocabulary file:", project_info.vocab_file)
    print("Train info:")
    print(f"  {project_info.train_info}")


def action_init(path: StrOrPath, *, yes: bool = False):
    prompt_overwrite(path, yes=yes)

    # default config
    dim = 256
    cnt_layer = 4
    cnt_head = 16
    assert dim % cnt_head == 0

    config = Config(
        vocab_size=1000,
        max_seq_len=4096,
        cnt_layer=cnt_layer,
        cnt_head=cnt_head,
        dim=dim,
        d_key=dim // cnt_head,
        d_value=dim // cnt_head,
        d_feed_forward=dim * 4,
    )

    weight_file = "weights.pth"
    vocab_file = "vocab.pkl.gz"

    project_info = ProjectInfo(
        config=config,
        weight_file=weight_file,
        vocab_file=vocab_file,
        train_info=TrainInfo(),
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
    mark: bool = False,
):
    assert seq_len > 0
    assert temperature > 0

    if not mark:
        assert len(prompt) > 0

    project_info = load_project(path)

    model = LLM(project_info.config).to(device)

    load_weights(Path(dirname(path), project_info.weight_file), model, device=device)

    words = load_gzip_pickle(Path(dirname(path), project_info.vocab_file))
    tokenizer = create_tokenizer(words)

    prompt_tokens = tokenizer.encode(prompt)

    if mark:
        tokens = [SOT] + prompt_tokens
    else:
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

    decoder = getincrementaldecoder("utf-8")(errors="ignore")

    if stream and all:
        print(tokenizer.decode(tokens), end="")

    for _ in range(count) if count >= 0 else always_true():
        input = tokens[-seq_len:]

        with torch.no_grad():
            logits = generate_step(model, torch.tensor(input, device=device))

        logits = logits[-1, :]  # get the last one
        probs = torch.softmax(logits / temperature, dim=-1)
        next = int(torch.multinomial(probs, num_samples=1).item())

        tokens.append(next)

        if stream:
            print(decoder.decode(tokenizer.words[next]), end="", flush=True)

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
    seq_len: int,
    batch_size: int,
    steps: int,
    data_paths: list[str],
    *,
    learning_rate: float = 0.001,
    val_paths: list[str] | None = None,
    log_period: int = 10,
    tensorboard_dir: str | None = None,
    device: str = "cpu",
    mark: bool = False,
):
    assert seq_len > 0
    assert batch_size > 0
    assert log_period > 0

    project_info = load_project(path)
    weight_file = Path(dirname(path), project_info.weight_file)
    vocab_file = Path(dirname(path), project_info.vocab_file)

    model = LLM(project_info.config).to(device)
    load_weights(weight_file, model, device=device)

    words = load_gzip_pickle(vocab_file)
    tokenizer = create_tokenizer(words)

    log_info("Making backups.")
    save_weights(get_backup_path(weight_file), model)
    save_project(get_backup_path(path), project_info)

    log_info("Preparing data.")
    if mark:
        log_info("Markers (SOT, EOT, etc.) will be added.")
    data_source = create_data_source(data_paths, seq_len, mark=mark)

    print("Data example:")
    example_tokens = data_source.sample()
    print(
        f"(len={len(example_tokens)}) {example_tokens} ({repr(tokenizer.decode(example_tokens))})"
    )

    val_batches = max(log_period // 10, 1)
    val_data_source = None
    if val_paths and len(val_paths) > 0:
        log_info("Preparing validation data.")
        val_data_source = create_data_source(val_paths, seq_len, mark=mark)

        print("Validation data example:")
        val_example_tokens = val_data_source.sample()
        print(
            f"(len={len(val_example_tokens)}) {val_example_tokens} ({repr(tokenizer.decode(val_example_tokens))})"
        )

    optimizer = Adam(model.parameters(), lr=learning_rate)

    writer = None
    if tensorboard_dir is not None:
        log_info("Opening TensorBoard writer.")
        writer = SummaryWriter(tensorboard_dir)

        # show graph
        data_loader = DataLoader(TrainDataset(data_source, seq_len, PAD))
        dummy_input, dummy_mask, _ = next(iter(data_loader))
        writer.add_graph(model, (dummy_input.to(device), dummy_mask.to(device)))

        # show input word embedding
        writer_add_embedding(writer, model, tokenizer, device=device)

    def log_important(msg: str):
        log_info(msg)
        if writer is not None:
            writer.add_text("train/log", msg, project_info.train_info.token_count)

    def save():
        log_info(f"Saving. (train_info: {project_info.train_info})")
        save_weights(weight_file, model)
        save_project(path, project_info)

    log_important(
        f"Training for {steps} steps (seq_len={seq_len}, batch_size={batch_size}). Time: {datetime.now()}"
    )

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
            if val_data_source is not None:
                val_loss = compute_validation_loss(
                    model=model,
                    data_source=val_data_source,
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
                    log_info("Making backups.")
                    save_weights(get_backup_path(weight_file), model)
                    save_project(get_backup_path(path), project_info)

                    if writer is not None:
                        writer_add_embedding(writer, model, tokenizer, device=device)

    t_start = time.perf_counter()

    train(
        info=project_info.train_info,
        model=model,
        seq_len=seq_len,
        optimizer=optimizer,
        data_source=data_source,
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
        # show input word embedding
        writer_add_embedding(writer, model, tokenizer, device=device)

        writer.close()


def action_model_init(path: StrOrPath, *, yes: bool = False):
    project_info = load_project(path)
    weight_file = Path(dirname(path), project_info.weight_file)
    vocab_file = Path(dirname(path), project_info.vocab_file)

    prompt_overwrite(weight_file)

    words = load_gzip_pickle(vocab_file)
    tokenizer = create_tokenizer(words)

    config = project_info.config

    correct_vocab_size = False
    if config.vocab_size != tokenizer.vocab_size():
        if not yes:
            ans = input(
                f"Current config.vocab_size {config.vocab_size} != actual vocab_size {tokenizer.vocab_size()}. Correct it? "
            )
            if ans.strip().lower() in YES_STRINGS:
                correct_vocab_size = True
            elif ans.strip().lower() in NO_STRINGS:
                pass
            else:
                raise PromptAbortError
        else:
            correct_vocab_size = True

    if correct_vocab_size:
        config.vocab_size = tokenizer.vocab_size()
        print("Corrected.")

    model = LLM(config)
    save_weights(weight_file, model)

    project_info.train_info = TrainInfo()

    save_project(path, project_info)

    log_success(f"Initialized model at {weight_file}.")


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
        default=os.cpu_count() or 8,
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
    gen_parser.add_argument(
        "-m", "--mark", action="store_true", help="add markers (SOT, EOT, etc.)"
    )

    # train
    train_parser = subparsers.add_parser("train", help="train model")
    train_parser.add_argument("path", help="project file")
    train_parser.add_argument("seq_len", type=int, help="sequence length")
    train_parser.add_argument("batch_size", type=int, help="batch size")
    train_parser.add_argument("steps", type=int, help="number of steps")
    train_parser.add_argument("data", nargs="+", help="token data file")
    train_parser.add_argument("-v", "--val", nargs="*", help="validation data file")
    train_parser.add_argument(
        "-lr", "--learning-rate", type=float, default=0.001, help="learning rate"
    )
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
        "-m", "--mark", action="store_true", help="add markers (SOT, EOT, etc.)"
    )

    # model_init
    model_init_parser = subparsers.add_parser(
        "model_init",
        help="initialize model accroding to configuration and fill with random weights",
    )
    model_init_parser.add_argument("path", help="project file")

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    torch.set_printoptions(linewidth=160)

    log_info(f"PyTorch will use {args.thread} CPU thread(s).")
    torch.set_num_threads(args.thread)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_info(f"Using device: {device}")

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
                mark=args.mark,
            )
        case "train":
            action_train(
                args.path,
                seq_len=args.seq_len,
                batch_size=args.batch_size,
                steps=args.steps,
                data_paths=args.data,
                learning_rate=args.learning_rate,
                val_paths=args.val,
                device=device,
                tensorboard_dir=args.tensorboard_dir,
                log_period=args.log_period,
                mark=args.mark,
            )
        case "model_init":
            action_model_init(args.path, yes=args.yes)
        case _:
            raise ValueError(f"unrecognized action: {args.action}")


if __name__ == "__main__":
    try:
        main()
    except PromptAbortError:
        log_info("Aborting.")
        sys.exit(1)
