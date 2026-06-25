import math
from argparse import ArgumentParser

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm

from spargel_llm.parquet_utils import (
    make_tokens_schema,
    read_row,
    resolve_row,
)


def action_demo(path: str, text: str, *, show_id: bool = False):
    tokenizer = Tokenizer.from_file(path)
    output = tokenizer.encode(text)
    if show_id:
        print(output.ids)
    else:
        print(output.tokens)


def action_encode(path: str, texts_path: str, output: str, batch_size: int = 1000):
    tokenizer = Tokenizer.from_file(path)
    pf = pq.ParquetFile(texts_path)
    schema = make_tokens_schema()

    with pq.ParquetWriter(output, schema, compression="zstd") as writer:
        total = math.ceil(pf.metadata.num_rows / batch_size)
        for batch in tqdm(
            pf.iter_batches(batch_size=batch_size), desc="batches", total=total
        ):
            texts = batch.column("text").to_pylist()
            encoded = tokenizer.encode_batch_fast(texts)
            writer.write_table(
                pa.table({"tokens": [x.ids for x in encoded]}, schema=schema)
            )
            del texts, encoded  # free memory

    print(f'Saved encoded Parquet to "{output}".')


def action_info(path: str):
    tokenizer = Tokenizer.from_file(path)
    print(f"Model:      {type(tokenizer.model).__name__}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")
    special_tokens = [
        t for t in tokenizer.get_added_tokens_decoder().values() if t.special
    ]
    if special_tokens:
        print(f"Specials:   {', '.join(t.content for t in special_tokens)}")


def action_init(path: str):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.save(path, pretty=True)
    print(f'Initialized tokenizer at "{path}".')


def action_show(tokenizer_path: str, tokens_path: str, row: int | None = None):
    """Decode and print the tokens at a specific row from a tokens.parquet file.

    Only reads the row group containing the target row, skipping all others.
    If row is not specified, a random row is selected.
    """
    tokenizer = Tokenizer.from_file(tokenizer_path)
    pf = pq.ParquetFile(tokens_path)
    row = resolve_row(pf, row)
    local_idx, table = read_row(pf, row, ["tokens"])
    token_ids = table.column("tokens")[local_idx].as_py()
    text = tokenizer.decode(token_ids)
    print(f"[{row}/{pf.metadata.num_rows}] length={len(token_ids)}")
    print(text)


def action_train(path: str, texts_path: str, vocab_size: int):
    tokenizer = Tokenizer.from_file(path)

    special_tokens = ["<unk>", "<pad>", "<sot>", "<eot>"]
    special_tokens += [f"<placeholder{i}>" for i in range(len(special_tokens), 16)]
    assert len(special_tokens) == 16

    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    pf = pq.ParquetFile(texts_path)

    def text_iter():
        for batch in pf.iter_batches():
            for text in batch.column("text").to_pylist():
                yield text

    tokenizer.train_from_iterator(text_iter(), trainer, length=pf.metadata.num_rows)
    tokenizer.save(path, pretty=True)
    print(f'Saved tokenizer at "{path}".')


def create_parser() -> ArgumentParser:
    parser = ArgumentParser("Vocabulary Management CLI Tool", fromfile_prefix_chars="@")

    subparsers = parser.add_subparsers(dest="action", help="actions", required=True)

    # demo
    demo_parser = subparsers.add_parser("demo", help="demo: encode a single text")
    demo_parser.add_argument("path", help="tokenizer JSON file")
    demo_parser.add_argument("text", help="text to encode")
    demo_parser.add_argument("--id", action="store_true", help="show token ids")

    # encode
    encode_parser = subparsers.add_parser(
        "encode", help="encode dataset texts to tokens"
    )
    encode_parser.add_argument("path", help="tokenizer JSON file")
    encode_parser.add_argument("texts_path", help="path to Parquet file")
    encode_parser.add_argument("output", help="output Parquet file path")
    encode_parser.add_argument(
        "--batch-size",
        "-bs",
        type=int,
        default=1,
        help="batch size for encoding (default: 1000)",
    )

    # info
    info_parser = subparsers.add_parser("info", help="show tokenizer info")
    info_parser.add_argument("path", help="tokenizer JSON file")

    # init
    init_parser = subparsers.add_parser("init", help="initialize tokenizer")
    init_parser.add_argument("path", help="tokenizer JSON file")

    # show
    show_parser = subparsers.add_parser(
        "show", help="decode and print a row from a tokens.parquet file"
    )
    show_parser.add_argument("tokenizer_path", help="tokenizer JSON file")
    show_parser.add_argument("tokens_path", help="path to tokens.parquet file")
    show_parser.add_argument(
        "row",
        nargs="?",
        type=int,
        default=None,
        help="zero-based row index to print (negative indices count from the end; "
        "random if omitted)",
    )

    # train
    train_parser = subparsers.add_parser("train", help="train tokenizer")
    train_parser.add_argument("path", help="tokenizer JSON file")
    train_parser.add_argument("texts_path", help="path to Parquet file")
    train_parser.add_argument("vocab_size", type=int, help="vocabulary size")

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    match args.action:
        case "demo":
            action_demo(args.path, args.text, show_id=args.id)
        case "encode":
            action_encode(
                args.path, args.texts_path, args.output, batch_size=args.batch_size
            )
        case "info":
            action_info(args.path)
        case "init":
            action_init(args.path)
        case "show":
            action_show(args.tokenizer_path, args.tokens_path, args.row)
        case "train":
            action_train(args.path, args.texts_path, args.vocab_size)


if __name__ == "__main__":
    main()
