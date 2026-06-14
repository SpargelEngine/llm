from abc import ABC, abstractmethod
from typing import override

import regex


class TextSplitter(ABC):
    @abstractmethod
    def split(self, text: str) -> list[int]:
        """
        Returns: a list of segment start positions (empty when input is empty)
        """
        pass


class TrivialSplitter(TextSplitter):
    """A trivial splitter that do no splitting at all"""

    @override
    def split(self, text: str) -> list[int]:
        return [0] if len(text) > 0 else []


class FixedLengthSplitter(TextSplitter):
    """
    Splits text into segements of given length.
    The last segment might be shorter.
    """

    length: int

    def __init__(self, length: int):
        self.length = length

    @override
    def split(self, text: str) -> list[int]:
        return list(range(0, len(text), self.length))


class RegexSplitter(TextSplitter):
    """
    Find split points by RegExp matching.

    Splits at all endpoints of matched substrings.
    """

    _pattern: regex.Pattern

    def __init__(self, pattern: str):
        self._pattern = regex.compile(pattern)

    @override
    def split(self, text: str) -> list[int]:
        positions = []
        last_end = 0

        for match in self._pattern.finditer(text):
            start, end = match.span()

            # there is a segment between two matches
            if start > last_end:
                positions.append(last_end)

            positions.append(start)

            last_end = end

        # there is a segment after the last match
        if last_end < len(text):
            positions.append(last_end)

        return positions


GPT2_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
