import unittest

from spargel_llm.tokenizer import ByteTokenizer, UnicodeTokenizer


class TestByteTokenizer(unittest.TestCase):
    def test_vocab_size(self):
        t = ByteTokenizer()
        self.assertEqual(t.vocab_size, 256)

    def test_encode(self):
        t = ByteTokenizer()
        self.assertEqual(t.encode("hello"), [104, 101, 108, 108, 111])
        self.assertEqual(t.encode("测试"), [0xE6, 0xB5, 0x8B, 0xE8, 0xAF, 0x95])

    def test_decode(self):
        t = ByteTokenizer()
        self.assertEqual(t.decode([48, 32, 126]), "0 ~")
        self.assertEqual(t.decode([0xE6, 0xB5, 0x8B, 0xE8, 0xAF, 0x95]), "测试")


class TestUnicodeTokenizer(unittest.TestCase):
    def test_basic(self):
        t = UnicodeTokenizer([])
        self.assertEqual(t.vocab_size, 0)

        t = UnicodeTokenizer(["a", "b", "c"])
        self.assertEqual(t.vocab_size, 3)

        t = UnicodeTokenizer(["a", "b", "c"], unknown=1)
        self.assertEqual(t.vocab_size, 3)

    def test_train(self):
        t = UnicodeTokenizer.train_from_text("123 abc")
        self.assertEqual(t.vocab_size, 7)

        t = UnicodeTokenizer.train_from_text(
            "123 abc", ["<|pad|>", "<|unk|>", "<|sos|>", "<|eos|>"], unknown=1
        )
        self.assertEqual(t.vocab_size, 7 + 4)

        t = UnicodeTokenizer.train_from_text("你好, 测试")
        self.assertEqual(t.vocab_size, 6)

        t = UnicodeTokenizer.train_from_text("おはよう")
        self.assertEqual(t.vocab_size, 4)

        t = UnicodeTokenizer.train_from_text("☺️")
        self.assertEqual(t.vocab_size, 2)

    def test_encode(self):
        t = UnicodeTokenizer.train_from_text("0123456789")
        self.assertEqual(t.encode("24680"), [2, 4, 6, 8, 0])

        t = UnicodeTokenizer.train_from_text("0123456789", unknown=3)
        self.assertEqual(t.encode("2468A"), [2, 4, 6, 8, 3])

    def test_decode(self):
        t = UnicodeTokenizer.train_from_text("0123456789")
        self.assertEqual(t.decode([1, 3, 5, 7, 9]), "13579")

        t = UnicodeTokenizer.train_from_text("0123456789", unknown=0)
        self.assertEqual(t.decode([1, 3, 5, 7, 123]), "13570")


if __name__ == "__main__":
    unittest.main()
