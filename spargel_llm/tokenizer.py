from __future__ import annotations

import abc
from typing import override


class Tokenizer(abc.ABC):
    @abc.abstractmethod
    def encode(self, input: str) -> list[int]:
        pass

    @abc.abstractmethod
    def decode(self, tokens: list[int]) -> str:
        pass

    @property
    @abc.abstractmethod
    def vocab_size(self) -> int:
        pass


class ByteTokenizer(Tokenizer):
    @override
    def encode(self, input: str) -> list[int]:
        return list(input.encode(encoding="utf-8"))

    @override
    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode(encoding="utf-8")

    @property
    @override
    def vocab_size(self) -> int:
        return 256


class UnicodeTokenizer(Tokenizer):
    _vocab: list[str]
    _stoi: dict[str, int]
    _iots: dict[int, str]

    def __init__(self, vocab: list[str]):
        """
        Args:
            vocab: a list of unicode characters
        """
        self._vocab = vocab
        self._stoi = {ch: i for i, ch in enumerate(vocab)}
        self._iots = {i: ch for i, ch in enumerate(vocab)}

    @staticmethod
    def train_from_text(text: str) -> UnicodeTokenizer:
        """
        Train a Unicode tokenizer using the codepoints in the given text.
        The codepoints will be sorted.

        Args:
            text: This should include all codepoints that will be encountered during encoding.
        """
        vocab = sorted(list(set(text)))
        return UnicodeTokenizer(vocab)

    @override
    def encode(self, input: str) -> list[int]:
        return [self._stoi[c] for c in input]

    @override
    def decode(self, tokens: list[int]) -> str:
        return "".join([self._iots[i] for i in tokens])

    @property
    @override
    def vocab_size(self) -> int:
        return len(self._vocab)
