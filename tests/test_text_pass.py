"""Unit tests for lm.text_pass."""

import gzip
import json
from pathlib import Path

import pytest

from spargel_llm.text_pass import (
    FilterPass,
    FindPass,
    JoinPass,
    PlainTextPass,
    ReadFilePass,
    ReferencePass,
    ReplacePass,
    SplitLinesPass,
    SplitPass,
    StripPass,
    TextPassList,
    _Chain,
    _resolve_parent,
    load_texts,
    process_texts,
)


class TestResolveParent:
    def test_returns_directory_unchanged(self, tmp_path):
        assert _resolve_parent(tmp_path) == tmp_path.resolve()

    def test_returns_parent_when_path_is_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.touch()
        assert _resolve_parent(f) == tmp_path.resolve()

    def test_handles_str_path(self, tmp_path):
        result = _resolve_parent(str(tmp_path))
        assert isinstance(result, Path)

    def test_non_existent_path_returns_unchanged(self, tmp_path):
        p = tmp_path / "nope"
        assert _resolve_parent(p) == p.resolve()


class TestPlainTextPass:
    def test_model_validation(self):
        p = PlainTextPass.model_validate({"name": "text", "texts": ["a", "b"]})
        assert p.name == "text"
        assert p.texts == ["a", "b"]

    def test_process_yields_given_texts(self):
        inst = PlainTextPass(name="text", texts=["hello", "world"]).build(".")
        result = list(inst.process([]))
        assert result == ["hello", "world"]

    def test_process_appends_to_existing_texts(self):
        inst = PlainTextPass(name="text", texts=["b", "c"]).build(".")
        result = list(inst.process(["a"]))
        assert result == ["a", "b", "c"]

    def test_process_empty_input(self):
        inst = PlainTextPass(name="text", texts=[]).build(".")
        assert list(inst.process(["x"])) == ["x"]
        assert list(inst.process([])) == []


class TestSplitLinesPass:
    def test_model_validation(self):
        p = SplitLinesPass.model_validate({"name": "split_lines"})
        assert p.name == "split_lines"
        assert p.keep_ends is False

    def test_model_validation_with_keep_ends(self):
        p = SplitLinesPass.model_validate({"name": "split_lines", "keep_ends": True})
        assert p.keep_ends is True

    def test_splits_single_text(self):
        inst = SplitLinesPass(name="split_lines").build(".")
        result = list(inst.process(["a\nb\nc"]))
        assert result == ["a", "b", "c"]

    def test_splits_multiple_texts(self):
        inst = SplitLinesPass(name="split_lines").build(".")
        result = list(inst.process(["a\nb", "c\nd"]))
        assert result == ["a", "b", "c", "d"]

    def test_keep_ends(self):
        inst = SplitLinesPass(name="split_lines", keep_ends=True).build(".")
        result = list(inst.process(["a\nb\nc"]))
        assert result == ["a\n", "b\n", "c"]

    def test_empty_text(self):
        # "".splitlines() returns [], so empty string yields nothing
        inst = SplitLinesPass(name="split_lines").build(".")
        result = list(inst.process([""]))
        assert result == []

    def test_empty_input(self):
        inst = SplitLinesPass(name="split_lines").build(".")
        assert list(inst.process([])) == []


class TestSplitPass:
    @staticmethod
    def _build(**kwargs) -> SplitPass._Instance:
        defaults: dict = {"name": "split", "separator": ","}
        return SplitPass.model_validate({**defaults, **kwargs}).build(".")

    def test_model_validation(self):
        p = SplitPass.model_validate({"name": "split", "separator": ","})
        assert p.name == "split"
        assert p.separator == ","
        assert p.regex is False
        assert p.maxsplit == 0
        assert p.behavior == "removed"

    def test_model_validation_with_all_options(self):
        p = SplitPass.model_validate(
            {
                "name": "split",
                "separator": r"\s+",
                "regex": True,
                "maxsplit": 3,
                "behavior": "isolated",
            }
        )
        assert p.separator == r"\s+"
        assert p.regex is True
        assert p.maxsplit == 3
        assert p.behavior == "isolated"

    # --- plain split (str.split) ---

    def test_splits_single_text_by_comma(self):
        inst = self._build(separator=",")
        result = list(inst.process(["a,b,c"]))
        assert result == ["a", "b", "c"]

    def test_splits_multiple_texts(self):
        inst = self._build(separator=",")
        result = list(inst.process(["a,b", "c,d"]))
        assert result == ["a", "b", "c", "d"]

    def test_split_by_whitespace(self):
        inst = self._build(separator=" ")
        result = list(inst.process(["hello world foo"]))
        assert result == ["hello", "world", "foo"]

    def test_separator_not_found(self):
        inst = self._build(separator=",")
        result = list(inst.process(["no commas here"]))
        assert result == ["no commas here"]

    def test_multi_char_separator(self):
        inst = self._build(separator="::")
        result = list(inst.process(["a::b::c"]))
        assert result == ["a", "b", "c"]

    def test_empty_text_plain(self):
        inst = self._build(separator=",")
        result = list(inst.process([""]))
        assert result == [""]

    def test_empty_input(self):
        inst = self._build(separator=",")
        assert list(inst.process([])) == []

    # --- regex split ---

    def test_regex_split_by_whitespace(self):
        inst = self._build(regex=True, separator=r"\s+")
        result = list(inst.process(["a  b\tc"]))
        assert result == ["a", "b", "c"]

    def test_regex_split_multiple_delimiters(self):
        inst = self._build(regex=True, separator=r"[,;]")
        result = list(inst.process(["a,b;c"]))
        assert result == ["a", "b", "c"]

    def test_regex_split_with_capturing_group(self):
        inst = self._build(regex=True, separator=r"(,)")
        result = list(inst.process(["a,b,c"]))
        assert result == ["a", ",", "b", ",", "c"]

    def test_regex_split_no_match(self):
        inst = self._build(regex=True, separator=r"\d+")
        result = list(inst.process(["abc def"]))
        assert result == ["abc def"]

    def test_regex_split_empty_text(self):
        inst = self._build(regex=True, separator=r",")
        result = list(inst.process([""]))
        assert result == [""]

    # --- maxsplit ---

    def test_maxsplit_plain(self):
        inst = self._build(separator=",", maxsplit=2)
        result = list(inst.process(["a,b,c,d,e"]))
        assert result == ["a", "b", "c,d,e"]

    def test_maxsplit_regex(self):
        inst = self._build(regex=True, separator=r"\s+", maxsplit=2)
        result = list(inst.process(["a b c d e"]))
        assert result == ["a", "b", "c d e"]

    def test_maxsplit_one(self):
        inst = self._build(separator=",", maxsplit=1)
        result = list(inst.process(["a,b,c"]))
        assert result == ["a", "b,c"]

    def test_maxsplit_zero_means_unlimited(self):
        inst = self._build(separator=",", maxsplit=0)
        result = list(inst.process(["a,b,c,d"]))
        assert result == ["a", "b", "c", "d"]

    # --- end-to-end via process_texts ---

    def test_end_to_end_split(self):
        passes = TextPassList.model_validate(
            {"passes": [{"name": "split", "separator": ","}]}
        ).passes
        result = list(process_texts(["a,b,c"], passes))
        assert result == ["a", "b", "c"]

    def test_end_to_end_split_regex(self):
        passes = TextPassList.model_validate(
            {"passes": [{"name": "split", "regex": True, "separator": r"[,;]"}]}
        ).passes
        result = list(process_texts(["a,b;c"], passes))
        assert result == ["a", "b", "c"]

    # --- integration: split + strip + filter pattern ---

    def test_split_strip_filter_pipeline(self):
        """Split by double-newline, strip each piece, drop empties — article boundary pattern."""
        passes = TextPassList.model_validate(
            {
                "passes": [
                    {"name": "split", "regex": True, "separator": r"\n\n+"},
                    {"name": "strip"},
                    {"name": "filter", "pattern": r"^$", "invert": True},
                ]
            }
        ).passes
        result = list(
            process_texts(["  = A = \n\n  text a  \n\n\n  = B = \n  text b  "], passes)
        )
        assert result == ["= A =", "text a", "= B = \n  text b"]

    # --- behavior modes ---

    def test_behavior_removed_is_default(self):
        """Default behavior is 'removed' — same as str.split / regex.split."""
        inst = self._build(separator=",", behavior="removed")
        result = list(inst.process(["a,b,c"]))
        assert result == ["a", "b", "c"]

    def test_behavior_removed_regex(self):
        inst = self._build(regex=True, separator=r"\s+", behavior="removed")
        result = list(inst.process(["a  b\tc"]))
        assert result == ["a", "b", "c"]

    # --- isolated ---

    def test_behavior_isolated_plain(self):
        inst = self._build(separator=",", behavior="isolated")
        result = list(inst.process(["a,b,c"]))
        assert result == ["a", ",", "b", ",", "c"]

    def test_behavior_isolated_regex(self):
        inst = self._build(regex=True, separator=r"\s+", behavior="isolated")
        result = list(inst.process(["a  b\tc"]))
        assert result == ["a", "  ", "b", "\t", "c"]

    def test_behavior_isolated_no_match(self):
        inst = self._build(separator=",", behavior="isolated")
        result = list(inst.process(["abc"]))
        assert result == ["abc"]

    def test_behavior_isolated_match_at_start(self):
        inst = self._build(separator=",", behavior="isolated")
        result = list(inst.process([",a,b"]))
        assert result == ["", ",", "a", ",", "b"]

    def test_behavior_isolated_match_at_end(self):
        inst = self._build(separator=",", behavior="isolated")
        result = list(inst.process(["a,b,"]))
        assert result == ["a", ",", "b", ",", ""]

    # --- merged_with_previous ---

    def test_behavior_merged_with_previous_plain(self):
        inst = self._build(separator=",", behavior="merged_with_previous")
        result = list(inst.process(["a,b,c"]))
        assert result == ["a,", "b,", "c"]

    def test_behavior_merged_with_previous_regex(self):
        inst = self._build(
            regex=True, separator=r"\s+", behavior="merged_with_previous"
        )
        result = list(inst.process(["a  b\tc"]))
        assert result == ["a  ", "b\t", "c"]

    def test_behavior_merged_with_previous_no_match(self):
        inst = self._build(separator=",", behavior="merged_with_previous")
        result = list(inst.process(["abc"]))
        assert result == ["abc"]

    # --- merged_with_next ---

    def test_behavior_merged_with_next_plain(self):
        inst = self._build(separator=",", behavior="merged_with_next")
        result = list(inst.process(["a,b,c"]))
        assert result == ["a", ",b", ",c"]

    def test_behavior_merged_with_next_regex(self):
        inst = self._build(regex=True, separator=r"\s+", behavior="merged_with_next")
        result = list(inst.process(["a  b\tc"]))
        assert result == ["a", "  b", "\tc"]

    def test_behavior_merged_with_next_no_match(self):
        inst = self._build(separator=",", behavior="merged_with_next")
        result = list(inst.process(["abc"]))
        assert result == ["abc"]

    # --- behavior + maxsplit ---

    def test_behavior_isolated_with_maxsplit(self):
        inst = self._build(separator=",", behavior="isolated", maxsplit=2)
        result = list(inst.process(["a,b,c,d"]))
        assert result == ["a", ",", "b", ",", "c,d"]

    def test_behavior_merged_with_previous_with_maxsplit(self):
        inst = self._build(separator=",", behavior="merged_with_previous", maxsplit=2)
        result = list(inst.process(["a,b,c,d"]))
        assert result == ["a,", "b,", "c,d"]

    def test_behavior_merged_with_next_with_maxsplit(self):
        inst = self._build(separator=",", behavior="merged_with_next", maxsplit=2)
        result = list(inst.process(["a,b,c,d"]))
        assert result == ["a", ",b", ",c,d"]

    # --- end-to-end behavior via process_texts ---

    def test_end_to_end_behavior_merged_with_next(self):
        passes = TextPassList.model_validate(
            {
                "passes": [
                    {"name": "split", "separator": ",", "behavior": "merged_with_next"}
                ]
            }
        ).passes
        result = list(process_texts(["a,b,c"], passes))
        assert result == ["a", ",b", ",c"]

    def test_end_to_end_behavior_isolated_regex(self):
        passes = TextPassList.model_validate(
            {
                "passes": [
                    {
                        "name": "split",
                        "regex": True,
                        "separator": r"[,;]",
                        "behavior": "isolated",
                    }
                ]
            }
        ).passes
        result = list(process_texts(["a,b;c"], passes))
        assert result == ["a", ",", "b", ";", "c"]


class TestJoinPass:
    def test_model_validation(self):
        p = JoinPass.model_validate({"name": "join"})
        assert p.name == "join"
        assert p.separator == ""

    def test_model_validation_with_separator(self):
        p = JoinPass.model_validate({"name": "join", "separator": "\n"})
        assert p.separator == "\n"

    def test_join_default_empty_separator(self):
        inst = JoinPass(name="join").build(".")
        result = list(inst.process(["a", "b", "c"]))
        assert result == ["abc"]

    def test_join_with_separator(self):
        inst = JoinPass(name="join", separator=", ").build(".")
        result = list(inst.process(["a", "b", "c"]))
        assert result == ["a, b, c"]

    def test_join_single_text(self):
        inst = JoinPass(name="join", separator="-").build(".")
        result = list(inst.process(["only"]))
        assert result == ["only"]

    def test_join_empty_input(self):
        inst = JoinPass(name="join", separator="-").build(".")
        result = list(inst.process([]))
        assert result == [""]


class TestStripPass:
    def test_model_validation(self):
        p = StripPass.model_validate({"name": "strip"})
        assert p.name == "strip"
        assert p.chars is None
        assert p.right is False

    def test_model_validation_with_chars_and_right(self):
        p = StripPass.model_validate({"name": "strip", "chars": ".,!", "right": True})
        assert p.chars == ".,!"
        assert p.right is True

    def test_strips_whitespace(self):
        inst = StripPass(name="strip").build(".")
        result = list(inst.process(["  hello  ", "\tworld\n"]))
        assert result == ["hello", "world"]

    def test_strips_custom_chars(self):
        inst = StripPass(name="strip", chars=".,!").build(".")
        result = list(inst.process(["...hello...", "!!!world!!!"]))
        assert result == ["hello", "world"]

    def test_strip_no_chars(self):
        inst = StripPass(name="strip", chars=None).build(".")
        result = list(inst.process(["  x  "]))
        assert result == ["x"]

    def test_strip_empty_input(self):
        inst = StripPass(name="strip").build(".")
        assert list(inst.process([])) == []

    def test_rstrip_right_side_only(self):
        inst = StripPass(name="strip", right=True).build(".")
        result = list(inst.process(["  hello  ", "\tworld\n"]))
        # rstrip only removes trailing whitespace
        assert result == ["  hello", "\tworld"]

    def test_rstrip_with_custom_chars(self):
        inst = StripPass(name="strip", right=True, chars=".!").build(".")
        result = list(inst.process(["hello...", "world!!!"]))
        assert result == ["hello", "world"]

    def test_empty_chars_string_strips_nothing(self):
        inst = StripPass(name="strip", chars="").build(".")
        result = list(inst.process(["  hello  "]))
        assert result == ["  hello  "]


class TestReplacePass:
    @staticmethod
    def _build(**kwargs) -> ReplacePass._Instance:
        defaults = {"name": "replace", "old": "foo", "new": "bar"}
        return ReplacePass.model_validate({**defaults, **kwargs}).build(".")

    def test_model_validation(self):
        p = ReplacePass.model_validate({"name": "replace", "old": "foo", "new": "bar"})
        assert p.name == "replace"
        assert p.old == "foo"
        assert p.new == "bar"
        assert p.regex is False
        assert p.repeat is False

    # --- plain replace ---

    def test_plain_replace_single_text(self):
        inst = self._build(old="cat", new="dog")
        result = list(inst.process(["the cat sat"]))
        assert result == ["the dog sat"]

    def test_plain_replace_multiple_texts(self):
        inst = self._build(old="cat", new="dog")
        result = list(inst.process(["a cat", "another cat"]))
        assert result == ["a dog", "another dog"]

    def test_plain_replace_no_match(self):
        inst = self._build(old="xyz", new="abc")
        result = list(inst.process(["nothing here"]))
        assert result == ["nothing here"]

    def test_plain_replace_empty_text(self):
        inst = self._build(old="x", new="y")
        result = list(inst.process([""]))
        assert result == [""]

    def test_plain_replace_empty_input(self):
        inst = self._build(old="x", new="y")
        assert list(inst.process([])) == []

    # --- regex replace ---

    def test_regex_replace(self):
        inst = self._build(regex=True, old=r"\d+", new="NUM")
        result = list(inst.process(["abc 123 def 456"]))
        assert result == ["abc NUM def NUM"]

    def test_regex_replace_with_groups(self):
        inst = self._build(regex=True, old=r"(\w+)@(\w+)", new=r"\2@\1")
        result = list(inst.process(["alice@host"]))
        assert result == ["host@alice"]

    # --- repeat replace ---

    def test_repeat_replace_until_stable(self):
        inst = self._build(old="xx", new="x", repeat=True)
        # "xxxx" → "xx" (replace all non-overlapping) → "x" → "x" (stable)
        result = list(inst.process(["xxxx"]))
        assert result == ["x"]

    def test_repeat_replace_no_match(self):
        inst = self._build(old="foo", new="bar", repeat=True)
        result = list(inst.process(["nothing"]))
        assert result == ["nothing"]

    def test_repeat_with_regex(self):
        inst = self._build(regex=True, old=r"aa+", new="a", repeat=True)
        # "aaaa" → regex finds "aaaa" → "a" → stable
        result = list(inst.process(["aaaa"]))
        assert result == ["a"]

    def test_max_repeat_caps_divergent_replacement(self):
        # "a" → "aa" → "aaaa" → … never stabilizes; max_repeat stops it
        inst = self._build(old="a", new="aa", repeat=True, max_repeat=3)
        result = list(inst.process(["a"]))
        # iteration 1: "a" → "aa"; 2: "aa" → "aaaa"; 3: "aaaa" → "aaaaaaaa"
        assert result == ["aaaaaaaa"]

    def test_max_repeat_stabilizes_before_limit(self):
        # replacement stabilizes before hitting the limit
        inst = self._build(old="xx", new="x", repeat=True, max_repeat=1000)
        result = list(inst.process(["xxxx"]))
        assert result == ["x"]

    # --- end-to-end via process_texts ---

    def test_end_to_end_replace(self):
        passes = TextPassList.model_validate(
            {"passes": [{"name": "replace", "old": "foo", "new": "bar"}]}
        ).passes
        result = list(process_texts(["foo bar foo"], passes))
        assert result == ["bar bar bar"]

    def test_end_to_end_replace_regex(self):
        passes = TextPassList.model_validate(
            {
                "passes": [
                    {
                        "name": "replace",
                        "regex": True,
                        "old": r"\b\w{3}\b",
                        "new": "???",
                    }
                ]
            }
        ).passes
        result = list(process_texts(["abc def ghij"], passes))
        assert result == ["??? ??? ghij"]


class TestForEachPass:
    def test_model_validation(self):
        p = TextPassList.model_validate(
            {"passes": [{"name": "for_each", "passes": [{"name": "strip"}]}]}
        ).passes[0]
        assert p.name == "for_each"
        assert len(p.passes) == 1

    def test_model_validation_empty_passes(self):
        p = TextPassList.model_validate(
            {"passes": [{"name": "for_each", "passes": []}]}
        ).passes[0]
        assert p.passes == []

    @staticmethod
    def _build(wraps: list[dict]):
        """Build a ForEachPass via TextPassListModel to resolve circular refs."""
        return (
            TextPassList.model_validate(
                {
                    "passes": [{"name": "for_each", "passes": wraps}],
                }
            )
            .passes[0]
            .build(".")
        )

    def test_applies_strip_to_each_text(self):
        inst = self._build([{"name": "strip"}])
        result = list(inst.process(["  a  ", "  b  ", "  c  "]))
        assert result == ["a", "b", "c"]

    def test_applies_split_lines_to_each_text(self):
        inst = self._build([{"name": "split_lines"}])
        result = list(inst.process(["a\nb", "c\nd"]))
        assert result == ["a", "b", "c", "d"]

    def test_chains_multiple_passes(self):
        inst = self._build([{"name": "strip"}, {"name": "split_lines"}])
        result = list(inst.process([" a\nb ", " c "]))
        # " a\nb " → strip → "a\nb" → split_lines → "a", "b"
        # " c " → strip → "c" → split_lines → "c"
        assert result == ["a", "b", "c"]

    def test_empty_input(self):
        inst = self._build([{"name": "strip"}])
        assert list(inst.process([])) == []

    def test_empty_passes(self):
        inst = self._build([])
        result = list(inst.process(["hello", "world"]))
        assert result == ["hello", "world"]

    def test_split_strip_join_per_text(self):
        """for_each { split_lines + strip + join } — per-line strip, preserves text boundaries."""
        inst = self._build(
            [
                {"name": "split_lines"},
                {"name": "strip"},
                {"name": "join", "separator": "\n"},
            ]
        )
        result = list(inst.process(["  a  \n  b  ", "  c  \n  d  "]))
        # Each text: split → strip each line → rejoin; output count = input count
        assert result == ["a\nb", "c\nd"]

    def test_split_strip_join_single_text(self):
        """Single text through split+strip+join round-trip."""
        inst = self._build(
            [
                {"name": "split_lines"},
                {"name": "strip"},
                {"name": "join", "separator": "\n"},
            ]
        )
        result = list(inst.process(["  hello  \n  world  \n  !  "]))
        assert result == ["hello\nworld\n!"]

    def test_split_strip_join_empty_lines_become_empty_string(self):
        """Empty/whitespace-only lines become empty strings after strip+join."""
        inst = self._build(
            [
                {"name": "split_lines"},
                {"name": "strip"},
                {"name": "join", "separator": "\n"},
            ]
        )
        result = list(inst.process(["a\n   \nb"]))
        assert result == ["a\n\nb"]

    def test_nested_for_each(self):
        """for_each containing another for_each — each level isolates its input."""
        # Outer: for each text → inner: for each text, strip → join with comma
        # So each outer text gets inner for_each applied to it (as a single-element stream)
        inst = (
            TextPassList.model_validate(
                {
                    "passes": [
                        {
                            "name": "for_each",
                            "passes": [
                                {
                                    "name": "for_each",
                                    "passes": [
                                        {"name": "strip"},
                                    ],
                                },
                            ],
                        },
                    ],
                }
            )
            .passes[0]
            .build(".")
        )
        result = list(inst.process(["  a  ", "  b  ", "  c  "]))
        assert result == ["a", "b", "c"]


class TestFilterPass:
    @staticmethod
    def _build(**kwargs) -> FilterPass._Instance:
        defaults: dict = {"name": "filter", "pattern": r"foo"}
        return FilterPass.model_validate({**defaults, **kwargs}).build(".")

    def test_model_validation(self):
        p = FilterPass.model_validate({"name": "filter", "pattern": r"foo"})
        assert p.name == "filter"
        assert p.pattern == "foo"
        assert p.invert is False

    def test_model_validation_with_invert(self):
        p = FilterPass.model_validate(
            {"name": "filter", "pattern": r"bar", "invert": True}
        )
        assert p.invert is True

    # --- non-inverted (keep matching) ---

    def test_keeps_matching_texts(self):
        inst = self._build(pattern=r"foo")
        result = list(inst.process(["foo", "foobar", "a foo b", "bar"]))
        assert result == ["foo", "foobar", "a foo b"]

    def test_discards_non_matching_texts(self):
        inst = self._build(pattern=r"\d+")
        result = list(inst.process(["abc", "123", "x1y"]))
        assert result == ["123", "x1y"]

    def test_empty_input(self):
        inst = self._build()
        assert list(inst.process([])) == []

    def test_all_match(self):
        inst = self._build(pattern=r"a")
        result = list(inst.process(["a", "ab", "ba"]))
        assert result == ["a", "ab", "ba"]

    def test_none_match(self):
        inst = self._build(pattern=r"xyz")
        result = list(inst.process(["abc", "def"]))
        assert result == []

    # --- inverted (discard matching) ---

    def test_invert_discards_matching_texts(self):
        inst = self._build(pattern=r"foo", invert=True)
        result = list(inst.process(["foo", "foobar", "a foo b", "bar"]))
        assert result == ["bar"]

    def test_invert_keeps_non_matching_texts(self):
        inst = self._build(pattern=r"\d+", invert=True)
        result = list(inst.process(["abc", "123", "x1y"]))
        assert result == ["abc"]

    def test_invert_empty_input(self):
        inst = self._build(invert=True)
        assert list(inst.process([])) == []

    def test_invert_all_match(self):
        inst = self._build(pattern=r"a", invert=True)
        result = list(inst.process(["a", "ab", "ba"]))
        assert result == []

    def test_invert_none_match(self):
        inst = self._build(pattern=r"xyz", invert=True)
        result = list(inst.process(["abc", "def"]))
        assert result == ["abc", "def"]

    def test_pattern_with_word_boundaries(self):
        inst = self._build(pattern=r"\bfoo\b")
        result = list(inst.process(["foo", "foobar", "a foo b", "food"]))
        assert result == ["foo", "a foo b"]

    # --- end-to-end via process_texts ---

    def test_end_to_end_filter(self):
        passes = TextPassList.model_validate(
            {"passes": [{"name": "filter", "pattern": r"keep"}]}
        ).passes
        result = list(process_texts(["keep me", "discard", "also keep this"], passes))
        assert result == ["keep me", "also keep this"]

    def test_end_to_end_filter_invert(self):
        passes = TextPassList.model_validate(
            {"passes": [{"name": "filter", "pattern": r"drop", "invert": True}]}
        ).passes
        result = list(process_texts(["drop me", "keep this", "drop that too"], passes))
        assert result == ["keep this"]


class TestReadFilePass:
    def test_model_validation(self):
        p = ReadFilePass.model_validate({"name": "read_file"})
        assert p.name == "read_file"
        assert p.base == "."
        assert p.encoding is None
        assert p.compression is None

    def test_model_validation_with_all_options(self):
        p = ReadFilePass.model_validate(
            {
                "name": "read_file",
                "base": "./data",
                "encoding": "utf-16",
                "compression": "gzip",
            }
        )
        assert p.base == "./data"
        assert p.encoding == "utf-16"
        assert p.compression == "gzip"

    def test_reads_single_file(self, tmp_path):
        f = tmp_path / "data" / "file.txt"
        f.parent.mkdir()
        f.write_text("content", encoding="utf-8")

        p = ReadFilePass(name="read_file", base=str(tmp_path / "data"))
        inst = p.build(".")
        result = list(inst.process(["file.txt"]))
        assert result == ["content"]

    def test_reads_multiple_files(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        (d / "a.txt").write_text("A")
        (d / "b.txt").write_text("B")

        p = ReadFilePass(name="read_file", base=str(d))
        inst = p.build(".")
        result = list(inst.process(["a.txt", "b.txt"]))
        assert result == ["A", "B"]

    def test_resolves_base_relative_to_config_path(self, tmp_path):
        config = tmp_path / "config.json"
        config.write_text("{}")
        d = tmp_path / "data"
        d.mkdir()
        (d / "file.txt").write_text("hello")

        p = ReadFilePass(name="read_file", base="data")
        inst = p.build(config)
        result = list(inst.process(["file.txt"]))
        assert result == ["hello"]

    def test_encoding(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        fake_utf16 = "hello".encode("utf-16")
        (d / "file.txt").write_bytes(fake_utf16)

        p = ReadFilePass(name="read_file", base=str(d), encoding="utf-16")
        inst = p.build(".")
        result = list(inst.process(["file.txt"]))
        assert result == ["hello"]

    def test_gzip_compression(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        path = d / "file.txt.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write("compressed content")

        p = ReadFilePass(name="read_file", base=str(d), compression="gzip")
        inst = p.build(".")
        result = list(inst.process(["file.txt.gz"]))
        assert result == ["compressed content"]

    def test_unicode_decode_error_is_logged_not_raised(self, tmp_path, caplog):
        d = tmp_path / "data"
        d.mkdir()
        (d / "bad.txt").write_bytes(b"\xff\xfe\x00\x01")

        p = ReadFilePass(name="read_file", base=str(d))
        inst = p.build(".")
        result = list(inst.process(["bad.txt"]))
        assert result == []
        assert "UnicodeDecodeError" in caplog.text

    def test_file_not_found_raises(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        p = ReadFilePass(name="read_file", base=str(d))
        inst = p.build(".")
        with pytest.raises(FileNotFoundError):
            list(inst.process(["nonexistent.txt"]))

    def test_gzip_with_encoding(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        path = d / "file.txt.gz"
        content = "héllo wörld"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(content)

        p = ReadFilePass(
            name="read_file", base=str(d), compression="gzip", encoding="utf-8"
        )
        inst = p.build(".")
        result = list(inst.process(["file.txt.gz"]))
        assert result == [content]


class TestFindPass:
    def test_model_validation(self):
        p = FindPass.model_validate({"name": "find"})
        assert p.name == "find"
        assert p.base == "."
        assert p.paths == ["."]
        assert p.file_pattern is None
        assert p.dir_pattern is None

    def test_model_validation_with_patterns(self):
        p = FindPass.model_validate(
            {
                "name": "find",
                "base": "./data",
                "paths": ["a", "b"],
                "file_pattern": r".*\.txt",
                "dir_pattern": r"sub",
            }
        )
        assert p.base == "./data"
        assert p.paths == ["a", "b"]
        assert p.file_pattern == r".*\.txt"
        assert p.dir_pattern == "sub"

    def _make_tree(self, tmp_path):
        base = tmp_path / "root"
        (base / "sub").mkdir(parents=True)
        (base / "a.txt").write_text("a")
        (base / "b.log").write_text("b")
        (base / "sub" / "c.txt").write_text("c")
        (base / "sub" / "d.log").write_text("d")
        (base / "sub" / "deep").mkdir()
        (base / "sub" / "deep" / "e.txt").write_text("e")
        return base

    def test_finds_all_files(self, tmp_path):
        base = self._make_tree(tmp_path)
        p = FindPass(name="find", base=str(base))
        inst = p.build(".")
        paths = sorted(inst.process([]))
        assert len(paths) == 5

    def test_filter_by_file_pattern(self, tmp_path):
        base = self._make_tree(tmp_path)
        p = FindPass(name="find", base=str(base), file_pattern=r".*\.txt")
        inst = p.build(".")
        paths = sorted(inst.process([]))
        assert len(paths) == 3
        assert all(p.endswith(".txt") for p in paths)

    def test_filter_by_dir_pattern(self, tmp_path):
        base = self._make_tree(tmp_path)
        p = FindPass(name="find", base=str(base), dir_pattern=r"(sub|\.)")
        inst = p.build(".")
        paths = sorted(inst.process([]))
        assert len(paths) == 4
        assert not any("deep" in p for p in paths)

    def test_file_and_dir_pattern_together(self, tmp_path):
        base = self._make_tree(tmp_path)
        p = FindPass(
            name="find",
            base=str(base),
            file_pattern=r".*\.txt",
            dir_pattern=r"(sub|\.)",
        )
        inst = p.build(".")
        paths = sorted(inst.process([]))
        assert len(paths) == 2
        assert all(p.endswith(".txt") for p in paths)

    def test_yields_from_multiple_start_dirs(self, tmp_path):
        base = tmp_path / "root"
        (base / "x").mkdir(parents=True)
        (base / "y").mkdir(parents=True)
        (base / "x" / "a.txt").write_text("a")
        (base / "y" / "b.txt").write_text("b")

        p = FindPass(name="find", base=str(base), paths=["x", "y"])
        inst = p.build(".")
        paths = sorted(inst.process([]))
        assert len(paths) == 2

    def test_empty_dir(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        p = FindPass(name="find", base=str(d))
        inst = p.build(".")
        assert list(inst.process([])) == []

    def test_base_resolves_relative_to_config_path(self, tmp_path):
        config = tmp_path / "config.json"
        config.write_text("{}")
        base = tmp_path / "data"
        base.mkdir()
        (base / "a.txt").write_text("a")

        p = FindPass(name="find", base="data", file_pattern=r".*\.txt")
        inst = p.build(config)
        paths = sorted(inst.process([]))
        assert len(paths) == 1
        assert paths[0].endswith("a.txt")

    def test_dir_pattern_only_no_file_pattern(self, tmp_path):
        base = self._make_tree(tmp_path)
        # Only traverse directories named "deep", prune everything else from root.
        # Root dir has sub/ → pruned; sub/ is never entered.
        p = FindPass(name="find", base=str(base), dir_pattern=r"deep")
        inst = p.build(".")
        paths = sorted(inst.process([]))
        # Only root-level files remain
        assert len(paths) == 2  # a.txt, b.log
        assert not any("sub" in p for p in paths)


class TestReferencePass:
    def test_model_validation(self):
        p = ReferencePass.model_validate({"name": "ref", "paths": ["ops.json"]})
        assert p.name == "ref"
        assert p.base == "."
        assert p.paths == ["ops.json"]

    def test_model_validation_with_base(self):
        p = ReferencePass.model_validate(
            {"name": "ref", "base": "./configs", "paths": ["a.json", "b.json"]}
        )
        assert p.base == "./configs"
        assert p.paths == ["a.json", "b.json"]

    def test_references_external_passes(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        ref_file = tmp_path / "refs.json"
        ref_file.write_text(json.dumps({"passes": [{"name": "strip"}]}))
        text_file = d / "file.txt"
        text_file.write_text("  hello  ")

        p = ReferencePass(name="ref", base=str(tmp_path), paths=["refs.json"])
        inst = p.build(".")

        result = list(inst.process(["  hello  ", "  world  "]))
        assert result == ["hello", "world"]

    def test_references_multiple_passes(self, tmp_path):
        ref_file = tmp_path / "refs.json"
        ref_file.write_text(
            json.dumps(
                {
                    "passes": [
                        {"name": "strip"},
                        {"name": "split_lines"},
                    ]
                }
            )
        )

        p = ReferencePass(name="ref", base=str(tmp_path), paths=["refs.json"])
        inst = p.build(".")
        result = list(inst.process([" a\nb "]))
        assert result == ["a", "b"]

    def test_references_multiple_ref_files(self, tmp_path):
        (tmp_path / "refs1.json").write_text(
            json.dumps({"passes": [{"name": "strip"}]})
        )
        (tmp_path / "refs2.json").write_text(
            json.dumps({"passes": [{"name": "join", "separator": ","}]})
        )

        p = ReferencePass(
            name="ref", base=str(tmp_path), paths=["refs1.json", "refs2.json"]
        )
        inst = p.build(".")
        result = list(inst.process([" a ", " b "]))
        assert result == ["a,b"]

    def test_base_resolves_relative_to_config_path(self, tmp_path):
        config = tmp_path / "config.json"
        config.write_text("{}")
        refs_dir = tmp_path / "refs"
        refs_dir.mkdir()
        (refs_dir / "ops.json").write_text(json.dumps({"passes": [{"name": "strip"}]}))

        p = ReferencePass(name="ref", base="refs", paths=["ops.json"])
        inst = p.build(config)
        result = list(inst.process(["  hello  "]))
        assert result == ["hello"]

    def test_nested_reference(self, tmp_path):
        """A ref file can itself reference another ref file, resolved relative to that file."""
        (tmp_path / "inner.json").write_text(
            json.dumps({"passes": [{"name": "strip"}]})
        )
        (tmp_path / "outer.json").write_text(
            json.dumps({"passes": [{"name": "ref", "paths": ["inner.json"]}]})
        )

        p = ReferencePass(name="ref", base=str(tmp_path), paths=["outer.json"])
        inst = p.build(".")
        result = list(inst.process(["  nested  "]))
        assert result == ["nested"]

    def test_missing_ref_file_raises(self, tmp_path):
        p = ReferencePass(name="ref", base=str(tmp_path), paths=["nope.json"])
        with pytest.raises(FileNotFoundError):
            p.build(".")


class TestChain:
    def test_chains_two_instances(self):
        a = StripPass(name="strip").build(".")
        b = SplitLinesPass(name="split_lines").build(".")
        chain = _Chain([a, b])
        result = list(chain.process([" a\nb ", " c "]))
        assert result == ["a", "b", "c"]

    def test_chain_single_instance(self):
        a = StripPass(name="strip").build(".")
        chain = _Chain([a])
        result = list(chain.process(["  x  "]))
        assert result == ["x"]

    def test_chain_empty(self):
        chain = _Chain([])
        result = list(chain.process(["a", "b"]))
        assert result == ["a", "b"]

    def test_three_pass_chain(self):
        a = PlainTextPass(name="text", texts=["extra"]).build(".")
        b = StripPass(name="strip").build(".")
        c = SplitLinesPass(name="split_lines").build(".")
        chain = _Chain([a, b, c])
        result = list(chain.process(["  hello\n  "]))
        assert result == ["hello", "extra"]


class TestTextPassList:
    def test_validates_multiple_passes(self):
        m = TextPassList.model_validate(
            {
                "passes": [
                    {"name": "text", "texts": ["a"]},
                    {"name": "strip"},
                    {"name": "join", "separator": "\n"},
                ],
            }
        )
        assert len(m.passes) == 3

    def test_rejects_invalid_pass(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TextPassList.model_validate(
                {
                    "passes": [{"name": "nonexistent"}],
                }
            )

    def test_empty_passes(self):
        m = TextPassList.model_validate({"passes": []})
        assert m.passes == []


class TestProcessTexts:
    def test_end_to_end_strip(self):
        passes = TextPassList.model_validate({"passes": [{"name": "strip"}]}).passes
        result = list(process_texts(["  hello  ", "  world  "], passes))
        assert result == ["hello", "world"]

    def test_end_to_end_for_each_strip(self):
        passes = TextPassList.model_validate(
            {"passes": [{"name": "for_each", "passes": [{"name": "strip"}]}]}
        ).passes
        result = list(process_texts(["  a  ", "  b  "], passes))
        assert result == ["a", "b"]

    def test_end_to_end_text_source(self):
        passes = TextPassList.model_validate(
            {"passes": [{"name": "text", "texts": ["additional"]}]}
        ).passes
        result = list(process_texts(["original"], passes))
        assert result == ["original", "additional"]

    def test_empty_passes(self):
        passes = TextPassList.model_validate({"passes": []}).passes
        result = list(process_texts(["hello"], passes))
        assert result == ["hello"]

    def test_empty_texts(self):
        passes = TextPassList.model_validate({"passes": [{"name": "strip"}]}).passes
        result = list(process_texts([], passes))
        assert result == []

    def test_join_multiple(self):
        passes = TextPassList.model_validate(
            {"passes": [{"name": "join", "separator": " "}]}
        ).passes
        result = list(process_texts(["hello", "world"], passes))
        assert result == ["hello world"]

    def test_path_parameter_affects_base_resolution(self, tmp_path):
        """The `path` parameter should be used as config_path for building passes."""
        config = tmp_path / "config.json"
        config.write_text("{}")
        data = tmp_path / "data"
        data.mkdir()
        (data / "f.txt").write_text("hello")

        passes = TextPassList.model_validate(
            {
                "passes": [
                    {
                        "name": "find",
                        "base": "data",
                        "paths": ["."],
                        "file_pattern": r".*\.txt",
                    },
                    {"name": "read_file", "base": "data"},
                ]
            }
        ).passes
        # path=config means bases resolve relative to tmp_path (config's parent)
        result = list(process_texts([], passes, path=config))
        assert result == ["hello"]


class TestLoadTexts:
    def test_loads_and_processes(self, tmp_path):
        config = tmp_path / "pipeline.json"
        config.write_text(json.dumps({"passes": [{"name": "strip"}]}))
        result = list(load_texts(config))
        assert result == []

    def test_loads_with_initial_texts_not_possible(self, tmp_path):
        """load_texts starts with empty text stream; texts come from passes."""
        config = tmp_path / "pipeline.json"
        config.write_text(
            json.dumps({"passes": [{"name": "text", "texts": ["a", "b"]}]})
        )
        result = list(load_texts(config))
        assert result == ["a", "b"]

    def test_loads_with_find_and_read(self, tmp_path):
        config = tmp_path / "pipeline.json"
        data = tmp_path / "data"
        data.mkdir()
        (data / "f.txt").write_text("content")
        config.write_text(
            json.dumps(
                {
                    "passes": [
                        {
                            "name": "find",
                            "base": "data",
                            "paths": ["."],
                            "file_pattern": r".*\.txt",
                        },
                        {"name": "read_file", "base": "data"},
                    ]
                }
            )
        )
        result = list(load_texts(config))
        assert result == ["content"]

    def test_loads_empty_config(self, tmp_path):
        config = tmp_path / "pipeline.json"
        config.write_text(json.dumps({"passes": []}))
        result = list(load_texts(config))
        assert result == []

    def test_loads_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            list(load_texts(tmp_path / "nope.json"))


class TestIntegration:
    def test_find_read_strip_pipeline(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        (d / "a.txt").write_text("  hello  ")
        (d / "b.txt").write_text("  world  ")

        passes = TextPassList.model_validate(
            {
                "passes": [
                    {
                        "name": "find",
                        "base": str(d),
                        "paths": ["."],
                        "file_pattern": r".*\.txt",
                    },
                    {"name": "read_file", "base": str(d)},
                    {"name": "strip"},
                ]
            }
        ).passes
        result = list(process_texts([], passes, path="."))
        assert sorted(result) == ["hello", "world"]

    def test_find_multiple_paths_with_read(self, tmp_path):
        (tmp_path / "A").mkdir()
        (tmp_path / "B").mkdir()
        (tmp_path / "A" / "a.txt").write_text("aaa")
        (tmp_path / "B" / "b.txt").write_text("bbb")

        passes = TextPassList.model_validate(
            {
                "passes": [
                    {
                        "name": "find",
                        "base": str(tmp_path),
                        "paths": ["A", "B"],
                        "file_pattern": r".*\.txt",
                    },
                    {"name": "read_file", "base": str(tmp_path)},
                ]
            }
        ).passes
        result = list(process_texts([], passes))
        assert sorted(result) == ["aaa", "bbb"]

    def test_ref_with_find_and_read(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        (d / "hello.txt").write_text("hello ref")

        ref_file = tmp_path / "ops.json"
        ref_file.write_text(
            json.dumps(
                {
                    "passes": [
                        {
                            "name": "find",
                            "base": str(d),
                            "paths": ["."],
                            "file_pattern": r".*\.txt",
                        },
                        {"name": "read_file", "base": str(d)},
                    ]
                }
            )
        )

        passes = TextPassList.model_validate(
            {"passes": [{"name": "ref", "base": str(tmp_path), "paths": ["ops.json"]}]}
        ).passes
        result = list(process_texts([], passes))
        assert result == ["hello ref"]

    def test_for_each_split_strip_join_preserves_texts(self):
        """Per-line strip via for_each { split_lines + strip + join }, keeping text count."""
        passes = TextPassList.model_validate(
            {
                "passes": [
                    {
                        "name": "for_each",
                        "passes": [
                            {"name": "split_lines"},
                            {"name": "strip"},
                            {"name": "join", "separator": "\n"},
                        ],
                    }
                ]
            }
        ).passes
        result = list(process_texts(["  a  \n  b  ", "  c  \n  d  "], passes))
        assert result == ["a\nb", "c\nd"]

    def test_for_each_split_strip_join_with_files(self, tmp_path):
        """Read files → per-line strip each → output cleaned texts."""
        d = tmp_path / "data"
        d.mkdir()
        (d / "a.txt").write_text("  hello  \n  world  ")
        (d / "b.txt").write_text("  foo  \n  bar  ")

        passes = TextPassList.model_validate(
            {
                "passes": [
                    {
                        "name": "find",
                        "base": str(d),
                        "paths": ["."],
                        "file_pattern": r".*\.txt",
                    },
                    {"name": "read_file", "base": str(d)},
                    {
                        "name": "for_each",
                        "passes": [
                            {"name": "split_lines"},
                            {"name": "strip"},
                            {"name": "join", "separator": "\n"},
                        ],
                    },
                ]
            }
        ).passes
        result = list(process_texts([], passes, path="."))
        assert sorted(result) == ["foo\nbar", "hello\nworld"]
