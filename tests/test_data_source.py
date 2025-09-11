import string
import time
import unittest
from random import Random

from spargel_llm.data_source import (
    PlainTextSource,
    GeneratedDataSource,
    WeightedDataSource,
)

seed = time.time()
# print("seed:", seed)


class TestGeneratedSource(unittest.TestCase):
    def test_trivial(self):
        random = Random(seed)

        n = random.randint(1, 100)
        source = GeneratedDataSource(lambda _: n, random=random)
        for _ in range(100):
            self.assertEqual(source.sample(), n)

    def test_single(self):
        random = Random(seed)

        source = GeneratedDataSource(lambda r: r.randint(95, 105), random=random)
        for _ in range(100):
            x = source.sample()
            self.assertTrue(95 <= x <= 105)

    def test_multiple(self):
        random = Random(seed)

        source = GeneratedDataSource(lambda r: r.randint(10, 30), random=random)
        for x in source.sample_multiple(100):
            self.assertTrue(10 <= x <= 30)


class TestWeightedSource(unittest.TestCase):
    def test_one(self):
        random = Random(seed)

        n = random.randint(1, 100)
        source = WeightedDataSource([(1, GeneratedDataSource(lambda _: n))])
        for x in source.sample_multiple(100):
            self.assertEqual(x, n)

    def test_two(self):
        random = Random(seed)

        a, b = random.randint(1, 100), random.randint(1, 100)
        source = WeightedDataSource(
            [
                (1, GeneratedDataSource(lambda _: a)),
                (2, GeneratedDataSource(lambda _: b)),
            ]
        )
        for x in source.sample_multiple(100):
            self.assertIn(x, [a, b])


class TestPlainTextSource(unittest.TestCase):
    def _generate_text(self, random: Random, length: int) -> str:
        return "".join(
            random.choices(
                string.ascii_letters + string.digits + string.punctuation, k=length
            )
        )

    def test_fixed_length(self):
        random = Random(seed)

        text_length = random.randint(16, 1000)
        text = self._generate_text(random, text_length)

        sample_length = random.randint(1, text_length // 16)
        source = PlainTextSource(text, sample_length, random=random)
        for sampled_text in source.sample_multiple(100):
            self.assertTrue(text.find(sampled_text) >= 0)
            self.assertEqual(len(sampled_text), sample_length)

    def test(self):
        random = Random(seed)

        text_length = random.randint(16, 1000)
        text = self._generate_text(random, text_length)

        source = PlainTextSource(text, 1, text_length // 16, random=random)
        for sampled_text in source.sample_multiple(100):
            self.assertTrue(text.find(sampled_text) >= 0)


if __name__ == "__main__":
    unittest.main()
