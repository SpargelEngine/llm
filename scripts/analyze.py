import json
import sys
from argparse import ArgumentParser

import torch
from tokenizers import Tokenizer

from spargel_llm.logging import log_info
from spargel_llm.typing import StrOrPath
from spargel_llm.utils import PromptAbortError


def load_tokenizer(path: StrOrPath) -> Tokenizer:
    return Tokenizer.from_file(str(path))


#### actions ####


def action_gen(
    dump_file: str,
    tokenizer_path: str,
    top_n: int = 10,
):
    assert top_n > 0

    tokenizer = load_tokenizer(tokenizer_path)

    with open(dump_file, "r") as f:
        lines = f.readlines()

    # parse info line
    info = json.loads(lines[0].strip())
    assert info.get("type") == "info", "first line must be type=info"

    prompt_tokens = info["prompt"]
    print("==== Meta Info ====")
    print(f"Prompt: {tokenizer.decode(prompt_tokens)!r}")
    print(f"Temperature: {info['temperature']}")
    print(f"Random seed: {info.get('random_seed', 'N/A')}")
    print(f"Prompt tokens: {prompt_tokens}")
    print()

    # parse gen lines
    print("==== Generated ====")
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        assert obj.get("type") == "gen"

        token_id = obj["id"]
        logits = torch.tensor(obj["logits"])
        top_values, top_indices = torch.topk(logits, top_n)

        parts = [f"{repr(tokenizer.decode([token_id]))}[{token_id}] |"]
        for idx, val in zip(top_indices.tolist(), top_values.tolist()):
            parts.append(f"{repr(tokenizer.decode([idx]))}[{idx}]:{val:.4f}")

        print(" ".join(parts))


#### main ####


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Analyze tool for model outputs",
    )

    subparsers = parser.add_subparsers(dest="action", help="actions", required=True)

    # gen
    gen_parser = subparsers.add_parser("gen", help="analyze generation dump")
    gen_parser.add_argument(
        "dump_file", help="dump file (JSONL) produced by tool.py gen --dump-file"
    )
    gen_parser.add_argument(
        "tokenizer", help="tokenizer file for decoding token ids"
    )
    gen_parser.add_argument(
        "-n",
        type=int,
        default=10,
        help="number of top logits to show (default: 10)",
    )

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    match args.action:
        case "gen":
            action_gen(
                args.dump_file,
                args.tokenizer,
                top_n=args.n,
            )
        case _:
            raise ValueError(f"unrecognized action: {args.action}")


if __name__ == "__main__":
    try:
        main()
    except PromptAbortError:
        log_info("Aborting.")
        sys.exit(1)
