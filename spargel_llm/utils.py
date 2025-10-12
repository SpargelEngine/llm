#
# Some useful tools
#

import time
from typing import Sequence

from .bpe import find_most_frequent_pair
from .tokenizer import Tokenizer


def _apply_merge(seq: Sequence[int], id1: int, id2: int, new_id: int):
    """
    Find all adjacent pairs of (id1, id2) in a sequence and replace them with new_id.
    """

    # no pairs: nothing to do
    if len(seq) <= 1:
        return list(seq)

    result: list[int] = []

    pos = 0
    while pos < len(seq) - 1:
        if seq[pos] == id1 and seq[pos + 1] == id2:
            result.append(new_id)
            pos += 2
        else:
            result.append(seq[pos])
            pos += 1

    if pos == len(seq) - 1:
        result.append(seq[pos])

    return result


def bpe_train(words: list[bytes], samples: Sequence[Sequence[int]], count: int):
    """
    Find frequent pairs and replace them with new words.

    Args:
        words: the list of words, id == index; new words will be appended to it.
        samples: from these we find pairs
        count: number of new words to find (will stop early if no more pairs)
    """

    samples_list = list(samples)

    for _ in range(count):

        t_start = time.perf_counter()

        new_id = len(words)

        id1, id2, freq = find_most_frequent_pair(samples_list)

        if freq == 0:
            print("No more pairs. Stopping.")
            break

        new_word = words[id1] + words[id2]

        words.append(new_word)

        for i, sample in enumerate(samples_list):
            samples_list[i] = _apply_merge(sample, id1, id2, new_id)

        t_end = time.perf_counter()

        print(
            f"word {new_id}: {id1}+{id2} freq={freq} \t{new_word} ({repr(new_word.decode(errors="ignore"))}) {t_end-t_start:.6f}s"
        )


def demo_tokenization(
    text: str,
    tokenizer: Tokenizer,
    *,
    color_codes: Sequence[str] = ["\033[40;37m", "\033[47;30m"],
):
    num_colors = len(color_codes)

    tokens = tokenizer.encode(text)

    for i, token in enumerate(tokens):
        if num_colors > 0:
            print(color_codes[i % num_colors], end="")
            print(tokenizer.decode([token]), end="")

    if num_colors > 0:
        print("\033[0m", end="")

    print(flush=True)
