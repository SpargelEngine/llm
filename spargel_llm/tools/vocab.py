import math
import random
import sys
from argparse import ArgumentParser
from random import Random
from typing import Optional, cast

from tqdm import tqdm

from spargel_llm.text_splitter import GPT2_SPLIT_PATTERN, RegexSplitter
from spargel_llm.tokenizer import WordTokenizer
from spargel_llm.utils import bpe_train, demo_tokenization

from .logging import log_info, log_success, log_warning
from .source import get_texts
from .typing import StrOrPath
from .utils import (
    PromptAbortError,
    load_gzip_pickle,
    prompt_overwrite,
    save_gzip_pickle,
)

#### helpers ####


def parse_source(s: str):
    parts = s.split(":")

    path = parts[0]

    if len(parts) >= 2:
        count = int(parts[1])
        if count <= 0:
            raise ValueError(f"count {count} for {path} should be > 0")
    else:
        count = 0

    return path, count


#### actions ####


def action_info(path: StrOrPath, *, dump: bool = False, min_len: Optional[int] = None):
    words = cast(list[bytes], load_gzip_pickle(path))

    print("Vocabulary size:", len(words))

    if dump:
        for i, word in enumerate(words):
            if min_len is None or len(word) >= min_len:
                print(f"{i}:\t{word} ({repr(word.decode(errors="ignore"))})")


def action_init(path: StrOrPath, *, yes: bool = False):
    prompt_overwrite(path, yes)

    save_gzip_pickle(path, [i.to_bytes() for i in range(256)])
    log_success(f"Initialized vocabulary at {path}.")


def action_expand(
    path: StrOrPath,
    size: int,
    source_paths: list[str],
    *,
    random_seed=None,
    parallel: bool = False,
):
    words = load_gzip_pickle(path)

    if size <= len(words):
        log_warning(f"Target size {size} <= current size {len(words)}, no nothing.")
        return

    if parallel:
        log_warning("Using experimental multi-processing feature.")

    text_splitter = RegexSplitter(GPT2_SPLIT_PATTERN)
    tokenizer = WordTokenizer(words)
    samples: list[list[int]] = []

    log_info("Preparing data.")

    texts = []
    random = Random(random_seed)

    for item in source_paths:
        source_path, count = parse_source(item)
        _texts = list(get_texts(source_path))

        if count > 0 and count < len(_texts):
            random.shuffle(_texts)
            _texts = _texts[:count]

        texts.extend(_texts)

    if len(texts) == 0:
        log_warning("No texts. Do nothing.")
        return

    log_info(f"Loaded {len(texts)} texts.")

    log_info("Splitting and tokenizing.")

    percent = 0
    for i, text in enumerate(texts):
        cuts = text_splitter.split(text) + [len(text)]
        for j in range(len(cuts) - 1):
            start, end = cuts[j], cuts[j + 1]
            tokens = tokenizer.encode(text[start:end])
            samples.append(tokens)

        # show progress
        new_percent = math.ceil((i + 1) * 100 / len(texts))
        for j in range(percent + 1, new_percent + 1):
            if j % 10 == 0:
                print(f"{j}%", end="", flush=True)
            else:
                print(".", end="", flush=True)
        percent = new_percent
    print()

    log_info(f"Loaded {len(samples)} samples.")
    log_info(
        f"Average sample length (in tokens): {sum(len(sample) for sample in samples) / len(samples)}"
    )

    log_info("Vocabulary expansion begins.")

    bpe_train(words, samples, size - len(words), parallel=parallel)

    log_info(f"Finished. Vocabulary size expaned to {len(words)}")

    save_gzip_pickle(path, words)


def action_copy(path: StrOrPath, out_path: StrOrPath, size: int, *, yes: bool = False):
    if size < 0:
        raise ValueError(f"Size {size} <= 0.")

    prompt_overwrite(out_path, yes)

    words = load_gzip_pickle(path)

    if size >= len(words):
        log_warning(
            f"Size {size} >= current vocabulary size {len(words)}. Copying all words."
        )

    if size == 0 or size >= len(words):
        save_gzip_pickle(out_path, words)
    else:
        save_gzip_pickle(out_path, words[:size])

    log_success("Copied.")


def action_tokenize(
    vocab_path: StrOrPath,
    out_path: StrOrPath,
    source_path: StrOrPath,
    *,
    yes: bool = False,
):
    prompt_overwrite(out_path, yes=yes)

    words = load_gzip_pickle(vocab_path)

    text_splitter = RegexSplitter(GPT2_SPLIT_PATTERN)
    tokenizer = WordTokenizer(words, text_splitter=text_splitter)

    data: list[list[int]] = []

    log_info(f"Tokenization begins.")

    count = 0
    for text in tqdm(get_texts(source_path)):
        tokens = tokenizer.encode(text)
        data.append(tokens)
        count += 1

    log_info(f"Number of texts: {count}")
    log_info(f"Total token count: {sum(len(seq) for seq in data)}")

    save_gzip_pickle(out_path, data)


def action_demo(path: StrOrPath, text: str):
    words = load_gzip_pickle(path)
    tokenizer = WordTokenizer(words)

    demo_tokenization(tokenizer, text)


def action_source_show(path: StrOrPath, count: int = 1):
    log_info("Loading texts.")
    texts = list(tqdm(get_texts(path)))
    print("Number of texts:", len(texts))

    if count > 0:
        print("Example:")
        for i in range(count):
            text = random.choice(texts)
            log_info(f"**** Sample {i} (len={len(text)})****")
            print(text, end="")
            if not text.endswith("\n"):
                log_info("%")
            log_info("****************")
            print()


def action_data_show(path: StrOrPath, vocab_path: StrOrPath, count: int = 1):
    words = load_gzip_pickle(vocab_path)
    text_splitter = RegexSplitter(GPT2_SPLIT_PATTERN)
    tokenizer = WordTokenizer(words, text_splitter=text_splitter)

    data = cast(list[list[int]], load_gzip_pickle(path))
    print("Number of samples:", len(data))

    if count > 0:
        print("Example:")
        for i in range(count):
            log_info(f"**** Sample {i} ****")

            tokens = random.choice(data)

            print(f"Tokens (len={len(tokens)}):", tokens)
            text = tokenizer.decode(tokens)
            print(f"Text (len={len(text)}):", repr(text))

            print()


#### main ####


def create_parser() -> ArgumentParser:
    parser = ArgumentParser("Vocabulary Management CLI Tool", fromfile_prefix_chars="@")

    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="always say YES and avoid interactive prompts",
    )

    subparsers = parser.add_subparsers(dest="action", help="actions", required=True)

    # info
    info_parser = subparsers.add_parser("info", help="show vocabulary information")
    info_parser.add_argument("path", help="vocabulary file")
    info_parser.add_argument("-d", "--dump", action="store_true", help="show words")
    info_parser.add_argument(
        "-m",
        "--min-len",
        type=int,
        help="only show words of length >= MIN_LEN when dumping",
    )

    # init
    init_parser = subparsers.add_parser(
        "init", help="initialize a vocabulary (to bytes 0x00~0xFF)"
    )
    init_parser.add_argument("path", help="vocabulary file")

    # expand
    expand_parser = subparsers.add_parser("expand", help="expand the vocabulary")
    expand_parser.add_argument("path", help="vocabulary file")
    expand_parser.add_argument("size", type=int, help="target vocabulary size")
    expand_parser.add_argument("source", nargs="+", help="text source file")
    expand_parser.add_argument("-rs", "--random-seed", help="random seed")
    expand_parser.add_argument(
        "-p",
        "--parallel",
        action="store_true",
        help="use multi-processing (experimental)",
    )

    # copy
    copy_parser = subparsers.add_parser("copy", help="copy vocabulary")
    copy_parser.add_argument("path", help="vocabulary file")
    copy_parser.add_argument("out_path", help="output file")
    copy_parser.add_argument("size", type=int, help="number of words to copy")

    # tokenize
    tokenize_parser = subparsers.add_parser(
        "tokenize", help="tokenize texts from source to token ids"
    )
    tokenize_parser.add_argument("vocab_path", help="vocabulary file")
    tokenize_parser.add_argument("out_path", help="output file")
    tokenize_parser.add_argument("source", help="text source file")

    # demo
    demo_parser = subparsers.add_parser("demo", help="demonstrate tokenization")
    demo_parser.add_argument("path", help="vocabulary file")
    demo_parser.add_argument("text", help="example text")

    # source_show
    source_show_parser = subparsers.add_parser(
        "source_show", help="show source text examples"
    )
    source_show_parser.add_argument("path", help="source path")
    source_show_parser.add_argument(
        "count", nargs="?", type=int, default=1, help="number of examples to show"
    )

    # data_show
    data_show_parser = subparsers.add_parser("data_show", help="show data examples")
    data_show_parser.add_argument("path", help="data path")
    data_show_parser.add_argument("vocab_path", help="vocabulary file")
    data_show_parser.add_argument(
        "count", nargs="?", type=int, default=1, help="number of examples to show"
    )

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    match args.action:
        case "info":
            action_info(args.path, dump=args.dump, min_len=args.min_len)
        case "init":
            action_init(args.path, yes=args.yes)
        case "expand":
            action_expand(
                args.path,
                args.size,
                args.source,
                random_seed=args.random_seed,
                parallel=args.parallel,
            )
        case "copy":
            action_copy(args.path, args.out_path, args.size, yes=args.yes)
        case "tokenize":
            action_tokenize(
                args.vocab_path,
                args.out_path,
                args.source,
                yes=args.yes,
            )
        case "demo":
            action_demo(args.path, args.text)
        case "source_show":
            action_source_show(args.path, args.count)
        case "data_show":
            action_data_show(args.path, args.vocab_path, args.count)
        case _:
            raise ValueError(f"unrecognized action: {args.action}")


if __name__ == "__main__":
    try:
        main()
    except PromptAbortError:
        log_info("Aborting.")
        sys.exit(1)
