from abc import ABC, abstractmethod
from typing import Optional, Sequence, override

from .bpe import byte_pair_merge
from .text_splitter import TextSplitter, TrivialSplitter


class Tokenizer(ABC):
    @abstractmethod
    def encode(self, input: str) -> list[int]: ...

    @abstractmethod
    def decode(self, tokens: list[int]) -> str: ...

    @abstractmethod
    def vocab_size(self) -> int: ...


class ByteTokenizer(Tokenizer):
    @override
    def encode(self, input: str) -> list[int]:
        return list(input.encode(encoding="utf-8"))

    @override
    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode(encoding="utf-8", errors="ignore")

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

    @override
    def vocab_size(self) -> int:
        return len(self.vocab)


class WordTokenizer(Tokenizer):
    """
    This tokenizer will do encoding by first cutting the text to segments,
    and then dividing it into words according to the word list.
    """

    words: Sequence[bytes]

    unknown: Optional[int]
    text_splitter: TextSplitter

    _word_to_id: dict[bytes, int]

    def __init__(
        self,
        words: Sequence[bytes],
        *,
        encode_blacklist: Sequence[int] = [],
        text_splitter: TextSplitter = TrivialSplitter(),
        unknown: Optional[int] = None,
    ):
        """
        Args:
            words: list of words, with word id == index in list
            encode_blacklist: word (indices) that are not allowed to encode
            text_splitter: the splitter used to cut the text before tokenizing (pre-tokenization)
            unknown: the token index for unknown word
        """

        self.words = words

        self.unknown = unknown
        self.text_splitter = text_splitter

        self._word_to_id = {}
        for i, word in enumerate(words):
            if i not in encode_blacklist:
                self._word_to_id[word] = i

    @override
    def encode(self, input: str) -> list[int]:
        if len(input) == 0:
            return []

        tokens = []

        # split text into segments
        cuts = self.text_splitter.split(input) + [len(input)]

        for i in range(len(cuts) - 1):
            start, end = cuts[i], cuts[i + 1]
            segment = input[start:end].encode("utf-8")

            # find words
            positions = byte_pair_merge(self._word_to_id, segment)
            positions.append(len(segment))

            for j in range(len(positions) - 1):
                word = segment[positions[j] : positions[j + 1]]

                if self.unknown is not None:
                    token = self._word_to_id.get(word, self.unknown)
                else:
                    token = self._word_to_id.get(word)

                tokens.append(token)

        return tokens

    @override
    def decode(self, tokens: list[int]) -> str:
        return b"".join(self.words[id] for id in tokens).decode(
            "utf-8", errors="ignore"
        )

    @override
    def vocab_size(self) -> int:
        return len(self.words)
