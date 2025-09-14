from __future__ import annotations

import abc
from typing import Optional, override


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
    vocab: list[str]
    unknown: Optional[int]

    _stoi: dict[str, int]
    _itos: dict[int, str]

    def __init__(self, vocab: list[str], *, unknown: Optional[int] = None):
        """
        Args:
            vocab: a list of unicode characters
            unknown (Optional): the fallback token id for unknown token
        """
        if unknown is not None:
            assert 0 <= unknown < len(vocab)

        self.vocab = vocab
        self.unknown = unknown
        self._stoi = {ch: i for i, ch in enumerate(vocab)}
        self._itos = {i: ch for i, ch in enumerate(vocab)}

    @staticmethod
    def train_from_text(
        text: str,
        reserved_vocab: Optional[list[str]] = None,
        *,
        unknown: Optional[int] = None,
        reserved_vocab_tail: Optional[list[str]] = None,
    ) -> UnicodeTokenizer:
        """
        Train a Unicode tokenizer using the codepoints in the given text.
        The codepoints will be sorted.

        Args:
            text: This should include all codepoints that will be encountered during encoding.
            reserved_vocab: these tokens will be prepended to the list
            unknown (Optional): the token id for error fallback
            reserved_vocab_tail (Optional): these tokens will be appended to the list
        """

        vocab = sorted(list(set(text)))

        if reserved_vocab is not None:
            vocab = reserved_vocab + vocab

        if reserved_vocab_tail is not None:
            vocab += reserved_vocab_tail

        return UnicodeTokenizer(vocab, unknown=unknown)

    @override
    def encode(self, input: str) -> list[int]:
        if self.unknown is not None:
            return [(self._stoi[c] if c in self._stoi else self.unknown) for c in input]
        else:
            return [self._stoi[c] for c in input]

    @override
    def decode(self, tokens: list[int]) -> str:
        if self.unknown is not None:
            return "".join(
                [self._itos[i if i in self._itos else self.unknown] for i in tokens]
            )
        else:
            return "".join([self._itos[i] for i in tokens])

    @property
    @override
    def vocab_size(self) -> int:
        return len(self.vocab)
