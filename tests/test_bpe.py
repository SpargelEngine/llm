import unittest

from spargel_llm.bpe import byte_pair_merge


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


if __name__ == "__main__":
    unittest.main()
