from abc import ABC, abstractmethod
from random import Random


class TextSource(ABC):
    @abstractmethod
    def sample(self) -> str:
        pass


class PlainTextSource(TextSource):
    def __init__(
        self, text: str, min_len: int, max_len: int, random: Random = Random()
    ):
        assert min_len >= 0 and max_len >= min_len and max_len <= len(text)

        self.text = text
        self.random = random
        self.min_len, self.max_len = min_len, max_len

    def sample(self) -> str:
        length = self.random.randint(self.min_len, self.max_len)
        start = self.random.randint(0, len(self.text) - length)
        return self.text[start : start + length]
