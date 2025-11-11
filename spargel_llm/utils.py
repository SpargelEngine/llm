#
# Some useful tools
#

import math
import time
from random import Random
from typing import Sequence

from .bpe import find_most_frequent_pair, find_most_frequent_pair_parallel
from .tokenizer import Tokenizer


class RandomPicker[T]:
    """O(1) random picker using alias method

    See https://ieeexplore.ieee.org/document/92917 .
    (A linear algorithm for generating random numbhers with a given distribution)
    """

    objects: Sequence[T]

    _n: int
    _prop: list[float]
    _alias: list[int]

    def __init__(self, objects: Sequence[T], weights: Sequence[float]):
        assert len(objects) == len(weights) and len(objects) > 0
        assert all(w >= 0 for w in weights) and sum(weights) > 0.0

        self._n = n = len(objects)
        self.objects = objects
        self._prop = [0.0] * n
        self._alias = list(range(n))

        s = sum(weights)
        weights = [w / s for w in weights]

        large: list[int] = []
        small: list[int] = []

        for i, w in enumerate(weights):
            self._prop[i] = p = w * n
            if p < 1.0:
                small.append(i)
            else:
                large.append(i)

        while small and large:
            s_index = small.pop()
            l_index = large.pop()

            self._alias[s_index] = l_index

            self._prop[l_index] = p = self._prop[s_index] + self._prop[l_index] - 1.0
            if p < 1.0:
                small.append(l_index)
            else:
                large.append(l_index)

        while small:
            self._prop[small.pop()] = 1.0
        while large:
            self._prop[large.pop()] = 1.0

    def sample(self, random: Random = Random()) -> T:
        x = random.random() * self._n
        i = math.floor(x)
        if (x - i) < self._prop[i]:
            return self.objects[i]
        else:
            return self.objects[self._alias[i]]


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


def bpe_expand(
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

        # remove seqs with len=1
        pos = 0
        for sample in samples_list:
            if len(sample) >= 2:
                samples_list[pos] = sample
                pos += 1
        del samples_list[pos:]

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
