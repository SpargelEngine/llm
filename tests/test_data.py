import time
import unittest
from random import Random

from spargel_llm.data import (
    GeneratedDataSource,
    ListDataset,
    SliceDataSource,
    WeightedDataSource,
)
from spargel_llm.datasets import FixedLengthDataset
from spargel_llm.meta import ai_marker

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
        source = WeightedDataSource(
            [1], [GeneratedDataSource(lambda _: n)], random=random
        )
        for x in source.sample_multiple(100):
            self.assertEqual(x, n)

    def test_two(self):
        random = Random(seed)

        a, b = random.randint(1, 100), random.randint(1, 100)
        source = WeightedDataSource(
            [1, 2],
            [GeneratedDataSource(lambda _: a), GeneratedDataSource(lambda _: b)],
            random=random,
        )
        for x in source.sample_multiple(100):
            self.assertIn(x, [a, b])


@ai_marker(human_checked=True)
class TestSliceDataSource(unittest.TestCase):
    def test_fixed_length_slices(self):
        """Test slices with fixed length (min_len == max_len)"""
        random = Random(seed)
        seq = list(range(100))
        slice_len = 10

        source = SliceDataSource(seq, slice_len, slice_len, random=random)

        for _ in range(50):
            slice_result = source.sample()
            self.assertEqual(len(slice_result), slice_len)
            # Verify slice is a valid subsequence
            self.assertTrue(all(x in seq for x in slice_result))

        source = SliceDataSource(seq, slice_len, random=random)

        for _ in range(50):
            slice_result = source.sample()
            self.assertEqual(len(slice_result), slice_len)
            self.assertTrue(all(x in seq for x in slice_result))

    def test_variable_length_slices(self):
        """Test slices with variable length (min_len != max_len)"""
        random = Random(seed)
        seq = list(range(50))
        min_len, max_len = 5, 15

        source = SliceDataSource(seq, min_len, max_len, random=random)

        for _ in range(50):
            slice_result = source.sample()
            self.assertTrue(min_len <= len(slice_result) <= max_len)
            # Verify slice is a valid subsequence
            self.assertTrue(all(x in seq for x in slice_result))

    def test_boundary_conditions(self):
        """Test slices at sequence boundaries"""
        random = Random(seed)
        seq = list(range(20))

        # Test minimum possible slice
        source_min = SliceDataSource(seq, 1, 1, random=random)
        for _ in range(20):
            slice_result = source_min.sample()
            self.assertEqual(len(slice_result), 1)

        # Test maximum possible slice
        source_max = SliceDataSource(seq, 20, 20, random=random)
        for _ in range(5):
            slice_result = source_max.sample()
            self.assertEqual(len(slice_result), 20)
            self.assertEqual(slice_result, seq)

    def test_different_sequence_types(self):
        """Test with different sequence types (list and string)"""
        random = Random(seed)

        # Test with list
        list_seq = list(range(30))
        list_source = SliceDataSource(list_seq, 5, 10, random=random)
        for _ in range(20):
            slice_result = list_source.sample()
            self.assertTrue(isinstance(slice_result, list))
            self.assertTrue(5 <= len(slice_result) <= 10)

        # Test with string
        string_seq = "abcdefghijklmnopqrstuvwxyz"
        string_source = SliceDataSource(string_seq, 3, 7, random=random)
        for _ in range(20):
            slice_result = string_source.sample()
            self.assertTrue(isinstance(slice_result, str))
            self.assertTrue(3 <= len(slice_result) <= 7)
            self.assertTrue(all(c in string_seq for c in slice_result))

    def test_randomness(self):
        """Test that multiple samples produce different results"""
        random = Random(seed)
        seq = list(range(100))

        source = SliceDataSource(seq, 10, 20, random=random)

        # Collect multiple samples
        samples = [source.sample() for _ in range(20)]

        # Check that we have some variation in slice positions
        start_positions = [sample[0] for sample in samples]
        self.assertTrue(len(set(start_positions)) > 1)

        # Check that we have some variation in slice lengths
        lengths = [len(sample) for sample in samples]
        self.assertTrue(len(set(lengths)) > 1)

    def test_parameter_validation(self):
        """Test that invalid parameters raise assertions"""
        seq = list(range(10))

        # Test invalid min_len
        with self.assertRaises(AssertionError):
            SliceDataSource(seq, -1, 5)

        # Test invalid max_len > sequence length
        with self.assertRaises(AssertionError):
            SliceDataSource(seq, 5, 15)

        # Test invalid min_len > max_len
        with self.assertRaises(AssertionError):
            SliceDataSource(seq, 8, 5)

    def test_sample_multiple(self):
        """Test sample_multiple method"""
        random = Random(seed)
        seq = list(range(50))

        source = SliceDataSource(seq, 5, 10, random=random)

        # Test sampling multiple slices
        samples = list(source.sample_multiple(10))
        self.assertEqual(len(samples), 10)

        for sample in samples:
            self.assertTrue(5 <= len(sample) <= 10)
            self.assertTrue(all(x in seq for x in sample))


@ai_marker(human_checked=True)
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


@ai_marker(human_checked=True)
class TestFixedLengthDataset(unittest.TestCase):
    def test_basic_functionality(self):
        # Create a test sequence
        seq = list(range(26))
        length = 5
        dataset = FixedLengthDataset(seq, length)

        # Test __len__
        expected_length = 22
        self.assertEqual(len(dataset), expected_length)

        # Test __getitem__ for all indices
        for i in range(expected_length):
            expected_chunk = seq[i : i + length]
            self.assertEqual(dataset[i], expected_chunk)

    def test_with_stride(self):
        seq = list(range(26))
        length = 5
        stride = 2
        dataset = FixedLengthDataset(seq, length, stride=stride)

        # Test __len__
        expected_length = 11
        self.assertEqual(len(dataset), expected_length)

        # Test __getitem__ for all indices
        for i in range(expected_length):
            start = i * stride
            expected_chunk = seq[start : start + length]
            self.assertEqual(dataset[i], expected_chunk)

    def test_with_offset(self):
        seq = list(range(26))
        length = 5
        offset = 3
        dataset = FixedLengthDataset(seq, length, offset=offset)

        # Test __len__
        expected_length = 19
        self.assertEqual(len(dataset), expected_length)

        # Test __getitem__ for all indices
        for i in range(expected_length):
            start = offset + i
            expected_chunk = seq[start : start + length]
            self.assertEqual(dataset[i], expected_chunk)

    def test_index_error(self):
        seq = list(range(26))
        length = 5
        dataset = FixedLengthDataset(seq, length)

        # Test valid indices
        self.assertEqual(dataset[0], [0, 1, 2, 3, 4])
        self.assertEqual(dataset[1], [1, 2, 3, 4, 5])
        self.assertEqual(dataset[21], [21, 22, 23, 24, 25])

        # Test invalid indices
        with self.assertRaises(IndexError):
            _ = dataset[22]
        with self.assertRaises(IndexError):
            _ = dataset[-1]

    def test_edge_cases(self):
        # Test with minimal sequence length
        seq = [0, 1, 2]
        length = 3
        dataset = FixedLengthDataset(seq, length)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0], [0, 1, 2])

        # Test with longer sequence and custom stride/offset
        seq = list(range(10))
        length = 4
        stride = 2
        offset = 1
        dataset = FixedLengthDataset(seq, length, stride=stride, offset=offset)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset[0], [1, 2, 3, 4])
        self.assertEqual(dataset[1], [3, 4, 5, 6])
        self.assertEqual(dataset[2], [5, 6, 7, 8])


if __name__ == "__main__":
    unittest.main()
