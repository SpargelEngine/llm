from abc import ABC, abstractmethod
from typing import override


class TextSplitter(ABC):
    @abstractmethod
    def split(self, text: str) -> list[int]:
        """
        Returns: a list of cut positions (excluding endpoints 0 and len(text))
        """
        pass


class TrivialSplitter(TextSplitter):
    """A trivial splitter that do no splitting at all"""

    @override
    def split(self, text: str) -> list[int]:
        return []


class FixedLengthSplitter(TextSplitter):
    """This splits text into segements of given length"""

    length: int

    def __init__(self, length: int):
        self.length = length

    @override
    def split(self, text: str) -> list[int]:
        return list(range(self.length, len(text), self.length))
