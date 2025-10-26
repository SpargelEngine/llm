#
# Some useful tools
#

import time
from typing import Sequence

from .bpe import find_most_frequent_pair, find_most_frequent_pair_parallel
from .tokenizer import Tokenizer


def apply_merge(seq: list[int], id1: int, id2: int, new_id: int):
    """
    Find all adjacent pairs of (id1, id2) in a sequence and replace them with new_id.
    """

    # no pairs: nothing to do
    if len(seq) <= 1:
        return

    # We do the merging in place so no extra memory allocation is needed.
    # There won't be any RAW (Read After Write) since we always have pos <= cursor.
    cursor = 0
    pos = 0
    while cursor < len(seq) - 1:
        if seq[cursor] == id1 and seq[cursor + 1] == id2:
            seq[pos] = new_id
            cursor += 2
        else:
            seq[pos] = seq[cursor]
            cursor += 1
        pos += 1

    # last one
    if cursor == len(seq) - 1:
        seq[pos] = seq[cursor]
        pos += 1

    del seq[pos:]


def bpe_train(
    words: list[bytes],
    samples: Sequence[Sequence[int]],
    count: int,
    *,
    parallel: bool = False,
):
    """
    Find frequent pairs and replace them with new words.

    Args:
        words: the list of words, id == index; new words will be appended to it.
        samples: from these we find pairs
        count: number of new words to find (will stop early if no more pairs)
    """

    samples_list = list(list(sample) for sample in samples)

    for _ in range(count):

        t_start = time.perf_counter()

        new_id = len(words)

        if parallel:
            id1, id2, freq = find_most_frequent_pair_parallel(samples_list)
        else:
            id1, id2, freq = find_most_frequent_pair(samples_list)

        if freq == 0:
            print("No more pairs. Stopping.")
            break

        new_word = words[id1] + words[id2]

        words.append(new_word)

        for sample in samples_list:
            apply_merge(sample, id1, id2, new_id)

        t_end = time.perf_counter()

        print(
            f"word {new_id}: {id1}+{id2} freq={freq} \t{new_word} ({repr(new_word.decode(errors="ignore"))}) {t_end-t_start:.6f}s"
        )


def demo_tokenization(
    tokenizer: Tokenizer,
    text: str,
    *,
    color_codes: Sequence[str] = ["\033[40;97m", "\033[107;30m"],
):
    num_colors = len(color_codes)

    tokens = tokenizer.encode(text)

    print("Number of tokens:", len(tokens))

    for i, token in enumerate(tokens):
        if num_colors > 0:
            print(color_codes[i % num_colors], end="")
        print(tokenizer.decode([token]), end="")

    if num_colors > 0:
        print("\033[0m", end="")

    print(flush=True)
