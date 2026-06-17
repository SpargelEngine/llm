from argparse import ArgumentParser
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm


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


def action_train(path: str, texts_path: str, vocab_size: int):
    tokenizer = Tokenizer.from_file(path)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<pad>", "<eot>"]
        + [f"<placeholder{i}>" for i in range(3, 16)],
    )
    pf = pq.ParquetFile(texts_path)

    def text_iter():
        for batch in pf.iter_batches():
            for text in batch.column("text").to_pylist():
                yield text

    tokenizer.train_from_iterator(text_iter(), trainer)
    tokenizer.save(path, pretty=True)
    print(f'Saved tokenizer at "{path}".')


def action_encode(path: str, texts_path: str, output: str, batch_size: int = 1):
    tokenizer = Tokenizer.from_file(path)
    pf = pq.ParquetFile(texts_path)

    schema = pa.schema([pa.field("tokens", pa.list_(pa.uint32()))])
    dataset_id = uuid4().hex
    schema = schema.with_metadata({"dataset_id": dataset_id})

    with pq.ParquetWriter(output, schema, compression="zstd") as writer:
        for batch in tqdm(pf.iter_batches(batch_size=batch_size), desc="batches"):
            texts = batch.column("text").to_pylist()
            encoded = tokenizer.encode_batch_fast(texts)
            writer.write_table(
                pa.table({"tokens": [x.ids for x in encoded]}, schema=schema)
            )

    print(f'Saved encoded Parquet to "{output}".')


def action_demo(path: str, text: str, *, show_id: bool = False):
    tokenizer = Tokenizer.from_file(path)
    output = tokenizer.encode(text)
    if show_id:
        print(output.ids)
    else:
        print(output.tokens)


def create_parser() -> ArgumentParser:
    parser = ArgumentParser("Vocabulary Management CLI Tool", fromfile_prefix_chars="@")

    subparsers = parser.add_subparsers(dest="action", help="actions", required=True)

    # info
    info_parser = subparsers.add_parser("info", help="show tokenizer info")
    info_parser.add_argument("path", help="tokenizer JSON file")

    # init
    init_parser = subparsers.add_parser("init", help="initialize tokenizer")
    init_parser.add_argument("path", help="tokenizer JSON file")

    # train
    train_parser = subparsers.add_parser("train", help="train tokenizer")
    train_parser.add_argument("path", help="tokenizer JSON file")
    train_parser.add_argument("texts_path", help="path to Parquet file")
    train_parser.add_argument("vocab_size", type=int, help="vocabulary size")

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
        help="batch size for encoding (default: 1)",
    )

    # demo
    demo_parser = subparsers.add_parser("demo", help="demo: encode a single text")
    demo_parser.add_argument("path", help="tokenizer JSON file")
    demo_parser.add_argument("text", help="text to encode")
    demo_parser.add_argument("--id", action="store_true", help="show token ids")

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    match args.action:
        case "info":
            action_info(args.path)
        case "init":
            action_init(args.path)
        case "train":
            action_train(args.path, args.texts_path, args.vocab_size)
        case "encode":
            action_encode(
                args.path, args.texts_path, args.output, batch_size=args.batch_size
            )
        case "demo":
            action_demo(args.path, args.text, show_id=args.id)


if __name__ == "__main__":
    main()
