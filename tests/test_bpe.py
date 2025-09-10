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
                },
                b"abcacde",
            ),
            [0, 2, 3, 5, 6],
        )


if __name__ == "__main__":
    unittest.main()
