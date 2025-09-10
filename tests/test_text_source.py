import string
import time
import unittest
from random import Random

from spargel_llm.text_source import PlainTextSource

seed = time.time()
# print("seed:", seed)


class TestPlainTextSource(unittest.TestCase):
    def test(self):
        random = Random(seed)

        text_length = random.randint(1, 1000000)
        text = "".join(
            random.choices(string.ascii_letters + string.digits, k=text_length)
        )

        text_source = PlainTextSource(text, 1, text_length // 16, random)
        for _ in range(1000):
            sampled_text = text_source.sample()
            self.assertTrue(text.find(sampled_text) >= 0)


if __name__ == "__main__":
    unittest.main()
