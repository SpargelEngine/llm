import unittest

from spargel_llm.bpe import byte_pair_merge, find_most_frequent_pair
from spargel_llm.meta import ai_marker


class TestBPE(unittest.TestCase):
    def test_merge(self):
        self.assertEqual(
            byte_pair_merge(
                {
                    b"ab": 0,
                    b"ac": 1,
                    b"ad": 2,
                    b"bc": 3,
                },
                b"abcacde",
            ),
            [0, 2, 3, 5, 6],
        )
        self.assertEqual(
            byte_pair_merge(
                {
                    b"ab": 0,
                    b"ac": 1,
                    b"ad": 2,
                    b"abc": 3,
                },
                b"abcacde",
            ),
            [0, 3, 5, 6],
        )
        self.assertEqual(
            byte_pair_merge(
                {
                    b"ab": 0,
                    b"ac": 1,
                    b"ad": 2,
                    b"abc": 3,
                    b"de": 4,
                },
                b"abcacde",
            ),
            [0, 3, 5],
        )
        self.assertEqual(
            byte_pair_merge(
                {
                    b"ab": 0,
                    b"ac": 1,
                    b"ad": 2,
                    b"abc": 3,
                    b"de": 4,
                    b"acde": 5,
                },
                b"abcacde",
            ),
            [0, 3],
        )
        self.assertEqual(
            byte_pair_merge(
                {
                    b"ab": 0,
                    b"ac": 1,
                    b"ad": 2,
                    b"abc": 3,
                    b"de": 4,
                    b"acde": 5,
                    b"abcacde": 6,
                },
                b"abcacde",
            ),
            [0],
        )

    @ai_marker(human_checked=True)
    def test_find_most_frequent_pair(self):
        # Test 1: Basic functionality - single sample with clear most frequent pair
        samples = [[1, 2, 1, 2]]
        result = find_most_frequent_pair(samples)
        self.assertEqual(result, (1, 2, 2))  # pair (1,2) appears twice

        # Test 2: Multiple samples - verify frequency accumulation across samples
        samples = [[1, 2], [1, 2], [2, 3]]
        result = find_most_frequent_pair(samples)
        self.assertEqual(result, (1, 2, 2))  # appears in two samples

        # Test 3: Tie handling - multiple pairs with same frequency
        # Note: max() will return the first occurrence in case of ties
        samples = [[1, 2, 3, 4]]
        result = find_most_frequent_pair(samples)
        # Should return one of the pairs with frequency 1
        self.assertEqual(result[2], 1)  # frequency should be 1
        self.assertIn(
            result[:2], [(1, 2), (2, 3), (3, 4)]
        )  # should be one of these pairs

        # Test 4: Single-element sequences - sequences that don't contribute pairs
        samples = [[1], [2, 3]]
        result = find_most_frequent_pair(samples)
        self.assertEqual(result, (2, 3, 1))  # only one pair exists

        # Test 5: Empty sequences - should be handled gracefully
        samples = [[], [1, 2]]
        result = find_most_frequent_pair(samples)
        self.assertEqual(result, (1, 2, 1))  # only one pair exists

        # Test 6: Complex case - multiple pairs with varying frequencies
        samples = [[1, 2, 3, 2, 1], [1, 2, 4], [3, 1, 1, 2]]
        result = find_most_frequent_pair(samples)
        self.assertEqual(result, (1, 2, 3))  # appears three times total

        # Test 7: Larger numbers and negative numbers
        samples = [[100, 200, 100, 200, -1, -2]]
        result = find_most_frequent_pair(samples)
        self.assertEqual(result, (100, 200, 2))  # pair (100,200) appears twice

        # Test 8: All empty samples
        samples = [[], [], []]
        result = find_most_frequent_pair(samples)
        self.assertEqual(result[2], 0)  # no pairs exist in any sample

        # Test 9: Mixed empty and single-element sequences
        samples = [[], [1], [2]]
        result = find_most_frequent_pair(samples)
        self.assertEqual(result[2], 0)  # no pairs exist in any sample


if __name__ == "__main__":
    unittest.main()
