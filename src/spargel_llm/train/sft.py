# Utilities for SFT (Supervised Fine-Tuning)

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel
from tokenizers import Tokenizer

from .basic import Item


class Message(BaseModel):
    role: str
    content: str


def build_sft_item_chatml(
    messages: Sequence[Message],
    tokenizer: Tokenizer,
    seq_len: int,
    *,
    pad_id: int,
    im_start_id: int,
    im_end_id: int,
    assistant_role: str = "assistant",
) -> Item:
    """ChatML style."""

    newline_ids = tokenizer.encode("\n").ids
    role_ids_cache: dict[str, list[int]] = {
        assistant_role: tokenizer.encode(assistant_role).ids
    }

    def get_role_ids(role: str):
        if role in role_ids_cache:
            return role_ids_cache[role]
        else:
            ids = tokenizer.encode(role).ids
            role_ids_cache[role] = ids
            return ids

    tokens: list[int] = []
    assistant_ranges: list[tuple[int, int]] = []

    for message in messages:
        tokens.append(im_start_id)
        tokens.extend(get_role_ids(message.role))
        tokens.extend(newline_ids)

        start = len(tokens)
        if start >= seq_len + 1:
            break

        tokens.extend(tokenizer.encode(message.content).ids)
        tokens.append(im_end_id)

        end = len(tokens)

        if message.role == assistant_role:
            assistant_ranges.append((start, end))

        tokens.extend(newline_ids)

    L = min(len(tokens) - 1, seq_len)

    input_ids = np.full(seq_len, pad_id, dtype=np.int64)
    target_ids = np.full(seq_len, pad_id, dtype=np.int64)
    mask = np.zeros(seq_len, dtype=np.bool)

    input_ids[:L] = tokens[:L]
    mask[:L] = True

    num_non_pad = 0

    for start, end in assistant_ranges:
        end = min(end, seq_len + 1)
        target_ids[start - 1 : end - 1] = tokens[start:end]
        num_non_pad += end - start

    return Item(
        seq_len=seq_len,
        input_ids=input_ids,
        target_ids=target_ids,
        mask=mask,
        num_non_pad=num_non_pad,
        num_valid=L,
    )
