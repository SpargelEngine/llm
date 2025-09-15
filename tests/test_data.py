import string
import time
import unittest
from random import Random

from spargel_llm.data import (
    DataLoader,
    FixedLengthTextDataset,
    GeneratedDataSource,
    ListDataset,
    PlainTextSource,
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


class TestDataset(unittest.TestCase):
    def test_basic_functionality(self):
        random = Random(seed)

        # Create a dataset with random data
        data = [random.randint(1, 100) for _ in range(10)]
        dataset = ListDataset(data)

        # Test __len__
        self.assertEqual(len(dataset), len(data))

        # Test __getitem__ for all indices
        for i in range(len(data)):
            self.assertEqual(dataset[i], data[i])

    def test_index_error(self):
        dataset = ListDataset([1, 2, 3])

        # Test valid indices
        self.assertEqual(dataset[0], 1)
        self.assertEqual(dataset[1], 2)
        self.assertEqual(dataset[2], 3)

        # Test invalid indices
        with self.assertRaises(IndexError):
            _ = dataset[3]
        with self.assertRaises(IndexError):
            _ = dataset[-1]


class TestFixedLengthTextDataset(unittest.TestCase):
    def test_basic_functionality(self):
        # Create a test text
        text = "abcdefghijklmnopqrstuvwxyz"
        length = 5
        dataset = FixedLengthTextDataset(text, length)

        # Test __len__
        expected_length = len(text) - length + 1
        self.assertEqual(len(dataset), expected_length)

        # Test __getitem__ for all indices
        for i in range(expected_length):
            expected_chunk = text[i : i + length]
            self.assertEqual(dataset[i], expected_chunk)

    def test_with_stride(self):
        text = "abcdefghijklmnopqrstuvwxyz"
        length = 5
        stride = 2
        dataset = FixedLengthTextDataset(text, length, stride=stride)

        # Test __len__
        expected_length = (len(text) - length) // stride + 1
        self.assertEqual(len(dataset), expected_length)

        # Test __getitem__ for all indices
        for i in range(expected_length):
            start = i * stride
            expected_chunk = text[start : start + length]
            self.assertEqual(dataset[i], expected_chunk)

    def test_with_offset(self):
        text = "abcdefghijklmnopqrstuvwxyz"
        length = 5
        offset = 3
        dataset = FixedLengthTextDataset(text, length, offset=offset)

        # Test __len__
        expected_length = (len(text) - offset - length) + 1
        self.assertEqual(len(dataset), expected_length)

        # Test __getitem__ for all indices
        for i in range(expected_length):
            start = offset + i
            expected_chunk = text[start : start + length]
            self.assertEqual(dataset[i], expected_chunk)

    def test_index_error(self):
        text = "abcdefghijklmnopqrstuvwxyz"
        length = 5
        dataset = FixedLengthTextDataset(text, length)

        # Test valid indices
        self.assertEqual(dataset[0], "abcde")
        self.assertEqual(dataset[1], "bcdef")
        self.assertEqual(dataset[21], "vwxyz")

        # Test invalid indices
        with self.assertRaises(IndexError):
            _ = dataset[22]
        with self.assertRaises(IndexError):
            _ = dataset[-1]

    def test_edge_cases(self):
        # Test with minimal text length
        text = "abc"
        length = 3
        dataset = FixedLengthTextDataset(text, length)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0], "abc")

        # Test with longer text and custom stride/offset
        text = "abcdefghij"
        length = 4
        stride = 2
        offset = 1
        dataset = FixedLengthTextDataset(text, length, stride=stride, offset=offset)
        self.assertEqual(len(dataset), 3)  # (10-1-4)//2 + 1 = 3
        self.assertEqual(dataset[0], "bcde")
        self.assertEqual(dataset[1], "defg")
        self.assertEqual(dataset[2], "fghi")


class TestDataLoader(unittest.TestCase):
    def test_no_shuffle(self):
        random = Random(seed)

        # Create dataset with predictable data
        data = list(range(10))
        dataset = ListDataset(data)

        # Test DataLoader without shuffling
        dataloader = DataLoader(dataset, shuffle=False, random=random)

        # Should iterate in order
        for i, item in enumerate(dataloader):
            self.assertEqual(item, i)

        # Test multiple iterations
        items = list(dataloader)
        self.assertEqual(items, list(range(10)))

        # Test explicit iteration
        dataloader = DataLoader(dataset, shuffle=False, random=random)
        collected = []
        for item in dataloader:
            collected.append(item)
        self.assertEqual(collected, list(range(10)))

    def test_with_shuffle(self):
        random = Random(seed)

        # Create dataset
        data = list(range(100))
        dataset = ListDataset(data)

        # Test DataLoader with shuffling
        dataloader = DataLoader(dataset, shuffle=True, random=random)

        # Should contain all items but in random order
        items = list(dataloader)
        self.assertEqual(len(items), 100)
        self.assertEqual(set(items), set(range(100)))

        # Order should be different from original
        self.assertNotEqual(items, list(range(100)))

    def test_empty_dataset(self):
        random = Random(seed)

        # Test with empty dataset
        dataset = ListDataset([])
        dataloader = DataLoader(dataset, random=random)

        # Should immediately raise StopIteration
        with self.assertRaises(StopIteration):
            next(iter(dataloader))

        # Should work with list()
        items = list(dataloader)
        self.assertEqual(items, [])

    def test_single_item(self):
        random = Random(seed)

        # Test with single item dataset
        dataset = ListDataset([42])
        dataloader = DataLoader(dataset, random=random)

        items = list(dataloader)
        self.assertEqual(items, [42])

        # Test with shuffling (should still return the single item)
        dataloader_shuffled = DataLoader(dataset, shuffle=True, random=random)
        items_shuffled = list(dataloader_shuffled)
        self.assertEqual(items_shuffled, [42])

    def test_multiple_iterations(self):
        random = Random(seed)

        data = list(range(100))
        dataset = ListDataset(data)

        # Test multiple iterations without shuffle (should be identical)
        dataloader = DataLoader(dataset, shuffle=False, random=random)
        first_pass = list(dataloader)
        second_pass = list(dataloader)
        self.assertEqual(first_pass, second_pass)
        self.assertEqual(first_pass, list(range(100)))

        # Test multiple iterations with shuffle (should be different each time)
        dataloader_shuffled = DataLoader(dataset, shuffle=True, random=random)
        first_shuffled = list(dataloader_shuffled)
        second_shuffled = list(dataloader_shuffled)

        # Both should contain all items
        self.assertEqual(set(first_shuffled), set(range(100)))
        self.assertEqual(set(second_shuffled), set(range(100)))

        # Order should be different between iterations (very high probability)
        self.assertNotEqual(first_shuffled, second_shuffled)


if __name__ == "__main__":
    unittest.main()
