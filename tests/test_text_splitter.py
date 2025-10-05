import unittest

from spargel_llm.meta import ai_marker
from spargel_llm.text_splitter import (
    FixedLengthSplitter,
    RegexSplitter,
    TrivialSplitter,
)


@ai_marker(human_checked=True)
class TestTrivialSplitter(unittest.TestCase):
    def test_empty_string(self):
        self.assertEqual(TrivialSplitter().split(""), [])

    def test_single_character(self):
        self.assertEqual(TrivialSplitter().split("a"), [0])

    def test_long_text(self):
        self.assertEqual(
            TrivialSplitter().split("This is a longer text that should not be split"),
            [0],
        )


@ai_marker(human_checked=True)
class TestFixedLengthSplitter(unittest.TestCase):
    def test_length_1(self):
        self.assertEqual(FixedLengthSplitter(1).split("abc"), [0, 1, 2])

    def test_length_3(self):
        self.assertEqual(FixedLengthSplitter(3).split("abcdef"), [0, 3])

    def test_length_5_with_remainder(self):
        self.assertEqual(FixedLengthSplitter(5).split("abcdefgh"), [0, 5])

    def test_text_shorter_than_length(self):
        self.assertEqual(FixedLengthSplitter(10).split("abc"), [0])

    def test_text_exactly_length(self):
        self.assertEqual(FixedLengthSplitter(3).split("abc"), [0])

    def test_empty_string(self):
        self.assertEqual(FixedLengthSplitter(5).split(""), [])


@ai_marker(human_checked=True)
class TestRegexSplitter(unittest.TestCase):
    def test_space_pattern(self):
        self.assertEqual(
            RegexSplitter(r"\s+").split("hello world this is a test"),
            [0, 5, 6, 11, 12, 16, 17, 19, 20, 21, 22],
        )

    def test_sentence_pattern(self):
        self.assertEqual(
            RegexSplitter(r"[.!?]").split("Hello world. How are you? I'm fine!"),
            [0, 11, 12, 24, 25, 34],
        )

    def test_no_matches(self):
        self.assertEqual(RegexSplitter(r"xyz").split("hello world"), [0])

    def test_multiple_consecutive_matches(self):
        self.assertEqual(RegexSplitter(r"a").split("baaab"), [0, 1, 2, 3, 4])

    def test_empty_string(self):
        self.assertEqual(RegexSplitter(r"\s+").split(""), [])

    def test_only_matches(self):
        self.assertEqual(RegexSplitter(r"a").split("aaa"), [0, 1, 2])

    def test_comma_separated_values(self):
        self.assertEqual(
            RegexSplitter(r",").split("apple,banana,cherry"), [0, 5, 6, 12, 13]
        )


if __name__ == "__main__":
    unittest.main()
