#
# Some useful tools
#

import math
from random import Random
from typing import Sequence

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
