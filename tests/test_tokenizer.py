import unittest

from spargel_llm.tokenizer import UnicodeTokenizer


class TestUnicodeTokenizer(unittest.TestCase):
    def test_basic(self):
        t = UnicodeTokenizer([])
        self.assertEqual(t.vocab_size, 0)

        t = UnicodeTokenizer(["a", "b", "c"])
        self.assertEqual(t.vocab_size, 3)

    def test_train(self):
        t = UnicodeTokenizer.train_from_text("123 abc")
        self.assertEqual(t.vocab_size, 7)

        t = UnicodeTokenizer.train_from_text("你好, 测试")
        self.assertEqual(t.vocab_size, 6)

        t = UnicodeTokenizer.train_from_text("おはよう")
        self.assertEqual(t.vocab_size, 4)

        t = UnicodeTokenizer.train_from_text("☺️")
        self.assertEqual(t.vocab_size, 2)

    def test_encode(self):
        t = UnicodeTokenizer.train_from_text("0123456789")
        self.assertEqual(t.encode("24680"), [2, 4, 6, 8, 0])

    def test_decode(self):
        t = UnicodeTokenizer.train_from_text("0123456789")
        self.assertEqual(t.decode([1, 3, 5, 7, 9]), "13579")

    def test_sort(self):
        t = UnicodeTokenizer.train_from_text("cba", sort=False)
        self.assertEqual(t.vocab_size, 3)
        self.assertEqual(t.encode("bca"), [1, 0, 2])
        self.assertEqual(t.decode([2, 1, 0]), "abc")


if __name__ == "__main__":
    unittest.main()
