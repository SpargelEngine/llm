"""Unit tests for lm.text_pass."""

import gzip
import json
from compression import zstd
from pathlib import Path

import pytest

from spargel_llm.text_pass import (
    ChainPassInstance,
    CombinePass,
    CombinePassModel,
    FilterPass,
    FindPass,
    ForEachPass,
    ForEachPassModel,
    JoinPass,
    JSONPass,
    PlainTextPass,
    ReadFilePass,
    ReferencePass,
    ReplacePass,
    SplitPass,
    StripPass,
    TextPassList,
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


class TestCombinePass:
    @staticmethod
    def _build(passes):
        return CombinePass(passes=passes).build(".")

    def test_each_pass_sees_all_texts(self):
        """Each pass runs on the full upstream iterator independently."""
        inst = self._build(
            [
                PlainTextPass(texts=["a", "b"]),
                PlainTextPass(texts=["c"]),
            ]
        )
        result = set(inst.process(["x", "y"]))
        # Each PlainTextPass appends its texts to the stream.
        # Both passes see "x", "y" from upstream (and ignore them),
        # then append their own texts.
        assert result >= {"a", "b", "c"}

    def test_combines_strip_and_split(self):
        """Strip pass and split pass each operate on the same input."""
        inst = self._build(
            [
                StripPass(),
                SplitPass(separator="\n"),
            ]
        )
        result = list(inst.process(["  a\nb  ", "  c  "]))
        # strip: "a\nb", "c"
        # split: "  a", "b  ", "  c  "
        assert "a\nb" in result
        assert "c" in result
        assert "  a" in result
        assert "b  " in result
        assert "  c  " in result

    def test_empty_passes_yields_nothing(self):
        inst = self._build([])
        result = list(inst.process(["hello", "world"]))
        assert result == []

    def test_empty_input(self):
        inst = self._build([StripPass()])
        result = list(inst.process([]))
        assert result == []

    def test_single_pass_is_identity_equivalent(self):
        """Single pass in combine yields same as running pass directly."""
        inst = self._build([StripPass()])
        result = list(inst.process(["  a  ", "  b  "]))
        assert set(result) == {"a", "b"}

    def test_counts_match_total_outputs(self):
        """Each pass produces 1:1 output, so total count = passes * inputs."""
        inst = self._build(
            [
                StripPass(),
                StripPass(),
            ]
        )
        result = list(inst.process(["  a  ", "  b  "]))
        # 2 passes × 2 inputs = 4 outputs
        assert len(result) == 4
        assert set(result) == {"a", "b"}


class TestCombinePassModel:
    @staticmethod
    def _build(wraps: list[dict]):
        return (
            TextPassList.model_validate(
                {"passes": [{"name": "combine", "passes": wraps}]}
            )
            .passes[0]
            .build(".")
        )

    def test_each_pass_sees_all_texts(self):
        inst = self._build(
            [
                {"name": "text", "texts": ["a", "b"]},
                {"name": "text", "texts": ["c"]},
            ]
        )
        result = set(inst.process(["x", "y"]))
        assert result >= {"a", "b", "c"}

    def test_combines_strip_and_split(self):
        inst = self._build(
            [
                {"name": "strip"},
                {"name": "split", "separator": "\n"},
            ]
        )
        result = list(inst.process(["  a\nb  ", "  c  "]))
        assert "a\nb" in result
        assert "c" in result
        assert "  a" in result
        assert "b  " in result
        assert "  c  " in result

    def test_empty_passes_yields_nothing(self):
        inst = self._build([])
        result = list(inst.process(["hello", "world"]))
        assert result == []

    def test_empty_input(self):
        inst = self._build([{"name": "strip"}])
        result = list(inst.process([]))
        assert result == []

    def test_model_validate(self):
        p = TextPassList.model_validate(
            {"passes": [{"name": "combine", "passes": [{"name": "strip"}]}]}
        ).passes[0]
        assert p.name == "combine"
        assert len(p.passes) == 1
        assert isinstance(p.passes[0], StripPass)

    def test_model_validate_empty_passes(self):
        p = TextPassList.model_validate(
            {"passes": [{"name": "combine", "passes": []}]}
        ).passes[0]
        assert isinstance(p, CombinePassModel) and p.passes == []


class TestFilterPass:
    @staticmethod
    def _build(**kwargs):
        defaults = {"name": "filter", "pattern": r"foo"}
        return FilterPass(**{**defaults, **kwargs}).build(".")

    # keep matching

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

    # inverted

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

    def test_end_to_end_filter(self):
        passes = [FilterPass(pattern=r"keep")]
        result = list(process_texts(["keep me", "discard", "also keep this"], passes))
        assert result == ["keep me", "also keep this"]

    def test_end_to_end_filter_invert(self):
        passes = [FilterPass(pattern=r"drop", invert=True)]
        result = list(process_texts(["drop me", "keep this", "drop that too"], passes))
        assert result == ["keep this"]

    def test_model_validate(self):
        p = FilterPass.model_validate(
            {"name": "filter", "pattern": r"bar", "invert": True}
        )
        assert p.name == "filter"
        assert p.pattern == "bar"
        assert p.invert is True


class TestFindPass:
    @staticmethod
    def _build(**kwargs):
        return FindPass(**kwargs).build(".")

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
        inst = self._build(base=str(base))
        paths = sorted(inst.process([]))
        assert len(paths) == 5

    def test_filter_by_file_pattern(self, tmp_path):
        base = self._make_tree(tmp_path)
        inst = self._build(base=str(base), file_pattern=r".*\.txt")
        paths = sorted(inst.process([]))
        assert len(paths) == 3
        assert all(p.endswith(".txt") for p in paths)

    def test_filter_by_dir_pattern(self, tmp_path):
        base = self._make_tree(tmp_path)
        inst = self._build(base=str(base), dir_pattern=r"(sub|\.)")
        paths = sorted(inst.process([]))
        assert len(paths) == 4
        assert not any("deep" in p for p in paths)

    def test_file_and_dir_pattern_together(self, tmp_path):
        base = self._make_tree(tmp_path)
        inst = self._build(
            base=str(base), file_pattern=r".*\.txt", dir_pattern=r"(sub|\.)"
        )
        paths = sorted(inst.process([]))
        assert len(paths) == 2
        assert all(p.endswith(".txt") for p in paths)

    def test_yields_from_multiple_start_dirs(self, tmp_path):
        base = tmp_path / "root"
        (base / "x").mkdir(parents=True)
        (base / "y").mkdir(parents=True)
        (base / "x" / "a.txt").write_text("a")
        (base / "y" / "b.txt").write_text("b")

        inst = self._build(base=str(base), paths=["x", "y"])
        paths = sorted(inst.process([]))
        assert len(paths) == 2

    def test_empty_dir(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        inst = self._build(base=str(d))
        assert list(inst.process([])) == []

    def test_base_resolves_relative_to_config_path(self, tmp_path):
        config = tmp_path / "config.json"
        config.write_text("{}")
        base = tmp_path / "data"
        base.mkdir()
        (base / "a.txt").write_text("a")

        p = FindPass(base="data", file_pattern=r".*\.txt")
        inst = p.build(config)
        paths = sorted(inst.process([]))
        assert len(paths) == 1
        assert paths[0].endswith("a.txt")

    def test_dir_pattern_only_no_file_pattern(self, tmp_path):
        base = self._make_tree(tmp_path)
        inst = self._build(base=str(base), dir_pattern=r"deep")
        paths = sorted(inst.process([]))
        assert len(paths) == 2
        assert not any("sub" in p for p in paths)

    def test_model_validate(self):
        p = FindPass.model_validate(
            {
                "name": "find",
                "base": "./data",
                "paths": ["a", "b"],
                "file_pattern": r".*\.txt",
                "dir_pattern": "sub",
            }
        )
        assert p.base == "./data"
        assert p.paths == ["a", "b"]
        assert p.file_pattern == r".*\.txt"
        assert p.dir_pattern == "sub"


class TestForEachPass:
    @staticmethod
    def _build(passes):
        return ForEachPass(passes=passes).build(".")

    def test_applies_single_pass_to_each_text(self):
        inst = self._build([StripPass()])
        result = list(inst.process(["  a  ", "  b  ", "  c  "]))
        assert result == ["a", "b", "c"]

    def test_applies_split_to_each_text(self):
        inst = self._build([SplitPass(separator="\n")])
        result = list(inst.process(["a\nb", "c\nd"]))
        assert result == ["a", "b", "c", "d"]

    def test_chains_multiple_passes_per_text(self):
        inst = self._build([StripPass(), SplitPass(separator="\n")])
        result = list(inst.process([" a\nb ", " c "]))
        assert result == ["a", "b", "c"]

    def test_empty_passes_is_identity(self):
        inst = self._build([])
        result = list(inst.process(["hello", "world"]))
        assert result == ["hello", "world"]

    def test_empty_input(self):
        inst = self._build([StripPass()])
        assert list(inst.process([])) == []

    def test_split_strip_join_preserves_text_count(self):
        inst = self._build(
            [
                SplitPass(separator="\n"),
                StripPass(),
                JoinPass(separator="\n"),
            ]
        )
        result = list(inst.process(["  a  \n  b  ", "  c  \n  d  "]))
        assert result == ["a\nb", "c\nd"]

    def test_split_strip_join_single_text(self):
        inst = self._build(
            [
                SplitPass(separator="\n"),
                StripPass(),
                JoinPass(separator="\n"),
            ]
        )
        result = list(inst.process(["  hello  \n  world  \n  !  "]))
        assert result == ["hello\nworld\n!"]

    def test_split_strip_join_empty_lines_become_empty_string(self):
        inst = self._build(
            [
                SplitPass(separator="\n"),
                StripPass(),
                JoinPass(separator="\n"),
            ]
        )
        result = list(inst.process(["a\n   \nb"]))
        assert result == ["a\n\nb"]

    # nesting — ForEachPass can contain ForEachPass

    def test_nests_for_each_inside_for_each(self):
        inst = self._build([ForEachPass(passes=[StripPass()])])
        result = list(inst.process(["  a  ", "  b  ", "  c  "]))
        assert result == ["a", "b", "c"]

    def test_nests_deeply(self):
        inst = self._build([ForEachPass(passes=[ForEachPass(passes=[StripPass()])])])
        result = list(inst.process(["  a  ", "  b  "]))
        assert result == ["a", "b"]

    def test_nested_split_strip_join(self):
        inst = self._build(
            [
                ForEachPass(
                    passes=[
                        SplitPass(separator="\n"),
                        StripPass(),
                        JoinPass(separator="\n"),
                    ]
                ),
                StripPass(),
            ]
        )
        result = list(inst.process(["  a  \n  b  ", "  c  \n  d  "]))
        assert result == ["a\nb", "c\nd"]

    def test_nested_for_each_with_filter(self):
        inst = self._build(
            [
                ForEachPass(
                    passes=[
                        SplitPass(separator="\n"),
                        FilterPass(pattern=r"keep"),
                    ]
                ),
            ]
        )
        result = list(inst.process(["keep\nskip\nkeep2"]))
        assert result == ["keep", "keep2"]

    def test_via_process_texts(self):
        passes = [ForEachPass(passes=[StripPass()])]
        result = list(process_texts(["  a  ", "  b  "], passes))
        assert result == ["a", "b"]

    def test_via_process_texts_with_nesting(self):
        passes = [
            ForEachPass(
                passes=[
                    SplitPass(separator="\n"),
                    StripPass(),
                    JoinPass(separator="\n"),
                ]
            )
        ]
        result = list(process_texts(["  a  \n  b  ", "  c  \n  d  "], passes))
        assert result == ["a\nb", "c\nd"]

    def test_via_process_texts_mixed_flat_and_for_each(self):
        passes = [
            StripPass(),
            ForEachPass(passes=[SplitPass(separator="\n")]),
            StripPass(),
        ]
        result = list(process_texts(["  a\nb  "], passes))
        assert result == ["a", "b"]


class TestForEachPassModel:
    @staticmethod
    def _build(wraps: list[dict]):
        """Build via TextPassList because ForEachPassModel has a circular
        reference through DiscriminatedTextPass that can only be resolved
        when the full TextPassList schema is built."""
        return (
            TextPassList.model_validate(
                {"passes": [{"name": "for_each", "passes": wraps}]}
            )
            .passes[0]
            .build(".")
        )

    def test_applies_strip_to_each_text(self):
        inst = self._build([{"name": "strip"}])
        result = list(inst.process(["  a  ", "  b  ", "  c  "]))
        assert result == ["a", "b", "c"]

    def test_chains_multiple_passes(self):
        inst = self._build([{"name": "strip"}, {"name": "split", "separator": "\n"}])
        result = list(inst.process([" a\nb ", " c "]))
        assert result == ["a", "b", "c"]

    def test_empty_input(self):
        inst = self._build([{"name": "strip"}])
        assert list(inst.process([])) == []

    def test_empty_passes(self):
        inst = self._build([])
        result = list(inst.process(["hello", "world"]))
        assert result == ["hello", "world"]

    def test_split_strip_join_per_text(self):
        inst = self._build(
            [
                {"name": "split", "separator": "\n"},
                {"name": "strip"},
                {"name": "join", "separator": "\n"},
            ]
        )
        result = list(inst.process(["  a  \n  b  ", "  c  \n  d  "]))
        assert result == ["a\nb", "c\nd"]

    def test_split_strip_join_single_text(self):
        inst = self._build(
            [
                {"name": "split", "separator": "\n"},
                {"name": "strip"},
                {"name": "join", "separator": "\n"},
            ]
        )
        result = list(inst.process(["  hello  \n  world  \n  !  "]))
        assert result == ["hello\nworld\n!"]

    def test_split_strip_join_empty_lines_become_empty_string(self):
        inst = self._build(
            [
                {"name": "split", "separator": "\n"},
                {"name": "strip"},
                {"name": "join", "separator": "\n"},
            ]
        )
        result = list(inst.process(["a\n   \nb"]))
        assert result == ["a\n\nb"]

    def test_nested_for_each(self):
        inst = (
            TextPassList.model_validate(
                {
                    "passes": [
                        {
                            "name": "for_each",
                            "passes": [
                                {
                                    "name": "for_each",
                                    "passes": [{"name": "strip"}],
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

    def test_model_validate(self):
        p = TextPassList.model_validate(
            {"passes": [{"name": "for_each", "passes": [{"name": "strip"}]}]}
        ).passes[0]
        assert p.name == "for_each"
        assert len(p.passes) == 1
        assert isinstance(p.passes[0], StripPass)

    def test_model_validate_empty_passes(self):
        p = TextPassList.model_validate(
            {"passes": [{"name": "for_each", "passes": []}]}
        ).passes[0]
        assert isinstance(p, ForEachPassModel) and p.passes == []


class TestJSONPass:
    @staticmethod
    def _build(**kwargs):
        return JSONPass(**kwargs).build(".")

    def test_extracts_string_field(self):
        inst = self._build(key="content")
        result = list(inst.process(['{"content": "hello", "id": 1}']))
        assert result == ["hello"]

    def test_extracts_from_multiple_texts(self):
        inst = self._build(key="msg")
        result = list(inst.process(['{"msg": "a", "id": 1}', '{"msg": "b", "id": 2}']))
        assert result == ["a", "b"]

    def test_key_not_found_raises_key_error(self):
        inst = self._build(key="missing")
        with pytest.raises(KeyError):
            list(inst.process(['{"content": "hello"}']))

    def test_value_not_string_raises_type_error(self):
        inst = self._build(key="content")
        with pytest.raises(TypeError):
            list(inst.process(['{"content": 123}']))

    def test_value_is_null_raises_type_error(self):
        inst = self._build(key="content")
        with pytest.raises(TypeError):
            list(inst.process(['{"content": null}']))

    def test_invalid_json_raises_error(self):
        inst = self._build(key="content")
        with pytest.raises(json.decoder.JSONDecodeError):
            list(inst.process(["not json"]))

    def test_empty_input(self):
        inst = self._build(key="content")
        assert list(inst.process([])) == []

    def test_model_validate(self):
        p = JSONPass.model_validate({"name": "json", "key": "title"})
        assert p.name == "json"
        assert p.key == "title"

    def test_end_to_end_json(self):
        passes = TextPassList.model_validate(
            {"passes": [{"name": "json", "key": "text"}]}
        ).passes
        result = list(process_texts(['{"text": "hello world"}'], passes))
        assert result == ["hello world"]

    def test_end_to_end_jsonl_pipeline(self, tmp_path):
        """Read a JSONL file, extract field from each line."""
        d = tmp_path / "data"
        d.mkdir()
        jsonl_path = d / "items.jsonl"
        jsonl_path.write_text(
            '{"text": "hello"}\n{"text": "world"}\n{"text": "foo"}\n',
            encoding="utf-8",
        )

        read_inst = ReadFilePass(base=str(d), lines=True).build(".")
        strip_inst = StripPass().build(".")
        filter_inst = FilterPass(pattern=r"^$", invert=True).build(".")
        json_inst = JSONPass(key="text").build(".")

        chain = ChainPassInstance([read_inst, strip_inst, filter_inst, json_inst])
        result = list(chain.process([jsonl_path.name]))
        assert result == ["hello", "world", "foo"]


class TestJoinPass:
    @staticmethod
    def _build(**kwargs):
        return JoinPass(**kwargs).build(".")

    def test_join_default_empty_separator(self):
        inst = self._build()
        result = list(inst.process(["a", "b", "c"]))
        assert result == ["abc"]

    def test_join_with_separator(self):
        inst = self._build(separator=", ")
        result = list(inst.process(["a", "b", "c"]))
        assert result == ["a, b, c"]

    def test_join_single_text(self):
        inst = self._build(separator="-")
        result = list(inst.process(["only"]))
        assert result == ["only"]

    def test_join_empty_input(self):
        inst = self._build(separator="-")
        result = list(inst.process([]))
        assert result == [""]

    def test_model_validate(self):
        p = JoinPass.model_validate({"name": "join", "separator": "\n"})
        assert p.name == "join"
        assert p.separator == "\n"


class TestPlainTextPass:
    @staticmethod
    def _build(**kwargs):
        return PlainTextPass(**kwargs).build(".")

    def test_process_yields_given_texts(self):
        inst = self._build(texts=["hello", "world"])
        result = list(inst.process([]))
        assert result == ["hello", "world"]

    def test_process_appends_to_existing_texts(self):
        inst = self._build(texts=["b", "c"])
        result = list(inst.process(["a"]))
        assert result == ["a", "b", "c"]

    def test_process_empty_input(self):
        inst = self._build(texts=[])
        assert list(inst.process(["x"])) == ["x"]
        assert list(inst.process([])) == []

    def test_model_validate(self):
        p = PlainTextPass.model_validate({"name": "text", "texts": ["a", "b"]})
        assert p.name == "text"
        assert p.texts == ["a", "b"]


class TestReadFilePass:
    @staticmethod
    def _build(**kwargs):
        return ReadFilePass(**kwargs).build(".")

    def test_reads_single_file(self, tmp_path):
        f = tmp_path / "data" / "file.txt"
        f.parent.mkdir()
        f.write_text("content", encoding="utf-8")

        inst = self._build(base=str(tmp_path / "data"))
        result = list(inst.process(["file.txt"]))
        assert result == ["content"]

    def test_reads_multiple_files(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        (d / "a.txt").write_text("A")
        (d / "b.txt").write_text("B")

        inst = self._build(base=str(d))
        result = list(inst.process(["a.txt", "b.txt"]))
        assert result == ["A", "B"]

    def test_resolves_base_relative_to_config_path(self, tmp_path):
        config = tmp_path / "config.json"
        config.write_text("{}")
        d = tmp_path / "data"
        d.mkdir()
        (d / "file.txt").write_text("hello")

        p = ReadFilePass(base="data")
        inst = p.build(config)
        result = list(inst.process(["file.txt"]))
        assert result == ["hello"]

    def test_encoding(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        fake_utf16 = "hello".encode("utf-16")
        (d / "file.txt").write_bytes(fake_utf16)

        inst = self._build(base=str(d), encoding="utf-16")
        result = list(inst.process(["file.txt"]))
        assert result == ["hello"]

    def test_gzip_compression(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        path = d / "file.txt.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write("compressed content")

        inst = self._build(base=str(d), compression="gzip")
        result = list(inst.process(["file.txt.gz"]))
        assert result == ["compressed content"]

    def test_unicode_decode_error_is_logged_not_raised(self, tmp_path, caplog):
        d = tmp_path / "data"
        d.mkdir()
        (d / "bad.txt").write_bytes(b"\xff\xfe\x00\x01")

        inst = self._build(base=str(d))
        result = list(inst.process(["bad.txt"]))
        assert result == []
        assert "UnicodeDecodeError" in caplog.text

    def test_file_not_found_raises(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        inst = self._build(base=str(d))
        with pytest.raises(FileNotFoundError):
            list(inst.process(["nonexistent.txt"]))

    def test_gzip_with_encoding(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        path = d / "file.txt.gz"
        content = "héllo wörld"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(content)

        inst = self._build(base=str(d), compression="gzip", encoding="utf-8")
        result = list(inst.process(["file.txt.gz"]))
        assert result == [content]

    def test_model_validate(self):
        p = ReadFilePass.model_validate(
            {
                "name": "read_file",
                "base": "./data",
                "encoding": "utf-16",
                "compression": "gzip",
                "lines": True,
            }
        )
        assert p.base == "./data"
        assert p.encoding == "utf-16"
        assert p.compression == "gzip"
        assert p.lines is True

    def test_lines_reads_line_by_line(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        (d / "file.txt").write_text("a\nb\nc\n", encoding="utf-8")

        inst = self._build(base=str(d), lines=True)
        result = list(inst.process(["file.txt"]))
        assert result == ["a\n", "b\n", "c\n"]

    def test_lines_with_gzip(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        path = d / "file.txt.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write("x\ny\n")

        inst = self._build(base=str(d), compression="gzip", lines=True)
        result = list(inst.process(["file.txt.gz"]))
        assert result == ["x\n", "y\n"]

    def test_zstd_compression(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        path = d / "file.txt.zst"
        with zstd.open(path, "wt", encoding="utf-8") as f:
            f.write("compressed content")

        inst = self._build(base=str(d), compression="zstd")
        result = list(inst.process(["file.txt.zst"]))
        assert result == ["compressed content"]

    def test_zstd_with_encoding(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        path = d / "file.txt.zst"
        content = "héllo wörld"
        with zstd.open(path, "wt", encoding="utf-8") as f:
            f.write(content)

        inst = self._build(base=str(d), compression="zstd", encoding="utf-8")
        result = list(inst.process(["file.txt.zst"]))
        assert result == [content]

    def test_lines_with_zstd(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        path = d / "file.txt.zst"
        with zstd.open(path, "wt", encoding="utf-8") as f:
            f.write("x\ny\n")

        inst = self._build(base=str(d), compression="zstd", lines=True)
        result = list(inst.process(["file.txt.zst"]))
        assert result == ["x\n", "y\n"]


class TestReferencePass:
    @staticmethod
    def _build(**kwargs):
        return ReferencePass(**kwargs).build(".")

    def test_references_external_passes(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        ref_file = tmp_path / "refs.json"
        ref_file.write_text(json.dumps({"passes": [{"name": "strip"}]}))
        text_file = d / "file.txt"
        text_file.write_text("  hello  ")

        inst = self._build(base=str(tmp_path), paths=["refs.json"])
        result = list(inst.process(["  hello  ", "  world  "]))
        assert result == ["hello", "world"]

    def test_references_multiple_passes(self, tmp_path):
        ref_file = tmp_path / "refs.json"
        ref_file.write_text(
            json.dumps(
                {
                    "passes": [
                        {"name": "strip"},
                        {"name": "split", "separator": "\n"},
                    ]
                }
            )
        )

        inst = self._build(base=str(tmp_path), paths=["refs.json"])
        result = list(inst.process([" a\nb "]))
        assert result == ["a", "b"]

    def test_references_multiple_ref_files(self, tmp_path):
        (tmp_path / "refs1.json").write_text(
            json.dumps({"passes": [{"name": "strip"}]})
        )
        (tmp_path / "refs2.json").write_text(
            json.dumps({"passes": [{"name": "join", "separator": ","}]})
        )

        inst = self._build(base=str(tmp_path), paths=["refs1.json", "refs2.json"])
        result = list(inst.process([" a ", " b "]))
        assert result == ["a,b"]

    def test_base_resolves_relative_to_config_path(self, tmp_path):
        config = tmp_path / "config.json"
        config.write_text("{}")
        refs_dir = tmp_path / "refs"
        refs_dir.mkdir()
        (refs_dir / "ops.json").write_text(json.dumps({"passes": [{"name": "strip"}]}))

        p = ReferencePass(base="refs", paths=["ops.json"])
        inst = p.build(config)
        result = list(inst.process(["  hello  "]))
        assert result == ["hello"]

    def test_nested_reference(self, tmp_path):
        (tmp_path / "inner.json").write_text(
            json.dumps({"passes": [{"name": "strip"}]})
        )
        (tmp_path / "outer.json").write_text(
            json.dumps({"passes": [{"name": "ref", "paths": ["inner.json"]}]})
        )

        inst = self._build(base=str(tmp_path), paths=["outer.json"])
        result = list(inst.process(["  nested  "]))
        assert result == ["nested"]

    def test_missing_ref_file_raises(self, tmp_path):
        p = ReferencePass(base=str(tmp_path), paths=["nope.json"])
        with pytest.raises(FileNotFoundError):
            p.build(".")

    def test_model_validate(self):
        p = ReferencePass.model_validate(
            {"name": "ref", "base": "./configs", "paths": ["a.json", "b.json"]}
        )
        assert p.base == "./configs"
        assert p.paths == ["a.json", "b.json"]


class TestReplacePass:
    @staticmethod
    def _build(**kwargs):
        defaults = {"name": "replace", "old": "foo", "new": "bar"}
        return ReplacePass(**{**defaults, **kwargs}).build(".")

    # plain replace

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

    # regex replace

    def test_regex_replace(self):
        inst = self._build(regex=True, old=r"\d+", new="NUM")
        result = list(inst.process(["abc 123 def 456"]))
        assert result == ["abc NUM def NUM"]

    def test_regex_replace_with_groups(self):
        inst = self._build(regex=True, old=r"(\w+)@(\w+)", new=r"\2@\1")
        result = list(inst.process(["alice@host"]))
        assert result == ["host@alice"]

    # repeat replace

    def test_repeat_replace_until_stable(self):
        inst = self._build(old="xx", new="x", repeat=True)
        result = list(inst.process(["xxxx"]))
        assert result == ["x"]

    def test_repeat_replace_no_match(self):
        inst = self._build(old="foo", new="bar", repeat=True)
        result = list(inst.process(["nothing"]))
        assert result == ["nothing"]

    def test_repeat_with_regex(self):
        inst = self._build(regex=True, old=r"aa+", new="a", repeat=True)
        result = list(inst.process(["aaaa"]))
        assert result == ["a"]

    def test_max_repeat_caps_divergent_replacement(self):
        inst = self._build(old="a", new="aa", repeat=True, max_repeat=3)
        result = list(inst.process(["a"]))
        assert result == ["aaaaaaaa"]

    def test_max_repeat_stabilizes_before_limit(self):
        inst = self._build(old="xx", new="x", repeat=True, max_repeat=1000)
        result = list(inst.process(["xxxx"]))
        assert result == ["x"]

    def test_end_to_end_replace(self):
        passes = [ReplacePass(old="foo", new="bar")]
        result = list(process_texts(["foo bar foo"], passes))
        assert result == ["bar bar bar"]

    def test_end_to_end_replace_regex(self):
        passes = [ReplacePass(regex=True, old=r"\b\w{3}\b", new="???")]
        result = list(process_texts(["abc def ghij"], passes))
        assert result == ["??? ??? ghij"]

    def test_model_validate(self):
        p = ReplacePass.model_validate(
            {
                "name": "replace",
                "regex": True,
                "old": r"\d+",
                "new": "NUM",
                "repeat": True,
                "max_repeat": 5,
            }
        )
        assert p.name == "replace"
        assert p.regex is True
        assert p.old == r"\d+"
        assert p.new == "NUM"
        assert p.repeat is True
        assert p.max_repeat == 5


class TestSplitPass:
    @staticmethod
    def _build(**kwargs):
        defaults = {"name": "split", "separator": ","}
        return SplitPass(**{**defaults, **kwargs}).build(".")

    # plain split

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

    # regex split

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

    # maxsplit

    def test_maxsplit_plain(self):
        inst = self._build(separator=",", max_split=2)
        result = list(inst.process(["a,b,c,d,e"]))
        assert result == ["a", "b", "c,d,e"]

    def test_maxsplit_regex(self):
        inst = self._build(regex=True, separator=r"\s+", max_split=2)
        result = list(inst.process(["a b c d e"]))
        assert result == ["a", "b", "c d e"]

    def test_maxsplit_one(self):
        inst = self._build(separator=",", max_split=1)
        result = list(inst.process(["a,b,c"]))
        assert result == ["a", "b,c"]

    def test_maxsplit_zero_means_unlimited(self):
        inst = self._build(separator=",", max_split=0)
        result = list(inst.process(["a,b,c,d"]))
        assert result == ["a", "b", "c", "d"]

    def test_split_strip_filter_pipeline(self):
        """Split by double-newline, strip each piece, drop empties."""
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

    # behavior: removed

    def test_behavior_removed_is_default(self):
        inst = self._build(separator=",", behavior="removed")
        result = list(inst.process(["a,b,c"]))
        assert result == ["a", "b", "c"]

    def test_behavior_removed_regex(self):
        inst = self._build(regex=True, separator=r"\s+", behavior="removed")
        result = list(inst.process(["a  b\tc"]))
        assert result == ["a", "b", "c"]

    # behavior: isolated

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

    # behavior: merged_with_previous

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

    # behavior: merged_with_next

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

    # behavior + maxsplit

    def test_behavior_isolated_with_maxsplit(self):
        inst = self._build(separator=",", behavior="isolated", max_split=2)
        result = list(inst.process(["a,b,c,d"]))
        assert result == ["a", ",", "b", ",", "c,d"]

    def test_behavior_merged_with_previous_with_maxsplit(self):
        inst = self._build(separator=",", behavior="merged_with_previous", max_split=2)
        result = list(inst.process(["a,b,c,d"]))
        assert result == ["a,", "b,", "c,d"]

    def test_behavior_merged_with_next_with_maxsplit(self):
        inst = self._build(separator=",", behavior="merged_with_next", max_split=2)
        result = list(inst.process(["a,b,c,d"]))
        assert result == ["a", ",b", ",c,d"]

    # model validate

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

    def test_model_validate(self):
        p = TextPassList.model_validate(
            {
                "passes": [
                    {
                        "name": "split",
                        "separator": r"\s+",
                        "regex": True,
                        "max_split": 3,
                        "behavior": "isolated",
                    }
                ]
            }
        ).passes[0]
        assert isinstance(p, SplitPass)
        assert p.separator == r"\s+"
        assert p.regex is True
        assert p.max_split == 3
        assert p.behavior == "isolated"


class TestStripPass:
    @staticmethod
    def _build(**kwargs):
        return StripPass(**kwargs).build(".")

    def test_strips_whitespace(self):
        inst = self._build()
        result = list(inst.process(["  hello  ", "\tworld\n"]))
        assert result == ["hello", "world"]

    def test_strips_custom_chars(self):
        inst = self._build(chars=".,!")
        result = list(inst.process(["...hello...", "!!!world!!!"]))
        assert result == ["hello", "world"]

    def test_strip_no_chars(self):
        inst = self._build(chars=None)
        result = list(inst.process(["  x  "]))
        assert result == ["x"]

    def test_strip_empty_input(self):
        inst = self._build()
        assert list(inst.process([])) == []

    def test_rstrip_right_side_only(self):
        inst = self._build(right=True)
        result = list(inst.process(["  hello  ", "\tworld\n"]))
        assert result == ["  hello", "\tworld"]

    def test_rstrip_with_custom_chars(self):
        inst = self._build(right=True, chars=".!")
        result = list(inst.process(["hello...", "world!!!"]))
        assert result == ["hello", "world"]

    def test_empty_chars_string_strips_nothing(self):
        inst = self._build(chars="")
        result = list(inst.process(["  hello  "]))
        assert result == ["  hello  "]

    def test_model_validate(self):
        p = StripPass.model_validate({"name": "strip", "chars": ".,!", "right": True})
        assert p.name == "strip"
        assert p.chars == ".,!"
        assert p.right is True


class TestChainPassInstance:
    def test_chains_two_instances(self):
        a = StripPass().build(".")
        b = SplitPass(separator="\n").build(".")
        chain = ChainPassInstance([a, b])
        result = list(chain.process([" a\nb ", " c "]))
        assert result == ["a", "b", "c"]

    def test_chain_single_instance(self):
        a = StripPass().build(".")
        chain = ChainPassInstance([a])
        result = list(chain.process(["  x  "]))
        assert result == ["x"]

    def test_chain_empty(self):
        chain = ChainPassInstance([])
        result = list(chain.process(["a", "b"]))
        assert result == ["a", "b"]

    def test_three_pass_chain(self):
        a = PlainTextPass(texts=["extra"]).build(".")
        b = StripPass().build(".")
        c = SplitPass(separator="\n").build(".")
        chain = ChainPassInstance([a, b, c])
        result = list(chain.process(["  hello\n  "]))
        assert result == ["hello", "extra"]

    def test_chain_with_for_each_instance(self):
        fe = ForEachPass(passes=[StripPass()]).build(".")
        j = JoinPass(separator="|").build(".")
        chain = ChainPassInstance([fe, j])
        result = list(chain.process(["  a  ", "  b  ", "  c  "]))
        assert result == ["a|b|c"]

    def test_chain_mixed_sources(self):
        a = StripPass().build(".")
        b = (
            TextPassList.model_validate(
                {"passes": [{"name": "split", "separator": "\n"}]}
            )
            .passes[0]
            .build(".")
        )
        chain = ChainPassInstance([a, b])
        result = list(chain.process([" a\nb ", " c "]))
        assert result == ["a", "b", "c"]

    def test_chain_for_each_followed_by_filter(self):
        fe = ForEachPass(passes=[SplitPass(separator="\n")]).build(".")
        f = FilterPass(pattern=r"keep").build(".")
        chain = ChainPassInstance([fe, f])
        result = list(chain.process(["keep\nskip", "also keep"]))
        assert result == ["keep", "also keep"]

    def test_chain_filter_followed_by_for_each(self):
        f = FilterPass(pattern=r"good").build(".")
        fe = ForEachPass(passes=[StripPass()]).build(".")
        chain = ChainPassInstance([f, fe])
        result = list(chain.process(["  good  ", "  bad  ", "  good2  "]))
        assert result == ["good", "good2"]


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
            TextPassList.model_validate({"passes": [{"name": "nonexistent"}]})

    def test_empty_passes(self):
        m = TextPassList.model_validate({"passes": []})
        assert m.passes == []

    def test_validates_all_pass_types(self):
        m = TextPassList.model_validate(
            {
                "passes": [
                    {"name": "filter", "pattern": r"."},
                    {"name": "find", "base": "."},
                    {"name": "for_each", "passes": []},
                    {"name": "join"},
                    {"name": "json", "key": "content"},
                    {"name": "text", "texts": []},
                    {"name": "read_file", "base": "."},
                    {"name": "ref", "paths": ["x.json"]},
                    {"name": "replace", "old": "", "new": ""},
                    {"name": "split", "separator": "\n"},
                    {"name": "split", "separator": ""},
                    {"name": "strip"},
                ]
            }
        )
        assert len(m.passes) == 12


class TestProcessTexts:
    def test_code_constructed_strip(self):
        passes = [StripPass()]
        result = list(process_texts(["  hello  ", "  world  "], passes))
        assert result == ["hello", "world"]

    def test_code_constructed_multi_pass_pipeline(self):
        passes = [
            PlainTextPass(texts=["extra"]),
            StripPass(),
            JoinPass(separator=" "),
        ]
        result = list(process_texts(["  hello  "], passes))
        assert result == ["hello extra"]

    def test_code_constructed_for_each(self):
        passes = [
            ForEachPass(
                passes=[
                    SplitPass(separator="\n"),
                    StripPass(),
                ]
            )
        ]
        result = list(process_texts([" a\nb ", " c\nd "], passes))
        assert result == ["a", "b", "c", "d"]

    def test_mixed_code_and_model_pass(self):
        model_passes = TextPassList.model_validate(
            {"passes": [{"name": "strip"}]}
        ).passes
        code_for_each = ForEachPass(passes=[SplitPass(separator="\n")])
        combined = [code_for_each, *model_passes]
        result = list(process_texts([" a\nb "], combined))
        assert result == ["a", "b"]

    def test_path_resolves_with_code_constructed_passes(self, tmp_path):
        config = tmp_path / "config.json"
        config.write_text("{}")
        data = tmp_path / "data"
        data.mkdir()
        (data / "f.txt").write_text("hello")

        passes = [
            FindPass(base="data", paths=["."], file_pattern=r".*\.txt"),
            ReadFilePass(base="data"),
        ]
        result = list(process_texts([], passes, path=config))
        assert result == ["hello"]

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
        result = list(process_texts([], passes, path=config))
        assert result == ["hello"]


class TestLoadTexts:
    def test_loads_and_processes(self, tmp_path):
        config = tmp_path / "pipeline.json"
        config.write_text(json.dumps({"passes": [{"name": "strip"}]}))
        result = list(load_texts(config))
        assert result == []

    def test_loads_with_initial_texts_not_possible(self, tmp_path):
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
    def test_find_read_for_each_strip_join(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        (d / "a.txt").write_text("  hello  \n  world  ")
        (d / "b.txt").write_text("  foo  \n  bar  ")

        passes = [
            FindPass(
                base=str(d),
                paths=["."],
                file_pattern=r".*\.txt",
            ),
            ReadFilePass(base=str(d)),
            ForEachPass(
                passes=[
                    SplitPass(separator="\n"),
                    StripPass(),
                    JoinPass(separator="\n"),
                ]
            ),
        ]
        result = list(process_texts([], passes, path="."))
        assert sorted(result) == ["foo\nbar", "hello\nworld"]

    def test_find_read_strip(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        (d / "a.txt").write_text("  hello  ")
        (d / "b.txt").write_text("  world  ")

        passes = [
            FindPass(base=str(d), paths=["."], file_pattern=r".*\.txt"),
            ReadFilePass(base=str(d)),
            StripPass(),
        ]
        result = list(process_texts([], passes, path="."))
        assert sorted(result) == ["hello", "world"]

    def test_find_multiple_paths_with_read(self, tmp_path):
        (tmp_path / "A").mkdir()
        (tmp_path / "B").mkdir()
        (tmp_path / "A" / "a.txt").write_text("aaa")
        (tmp_path / "B" / "b.txt").write_text("bbb")

        passes = [
            FindPass(
                base=str(tmp_path),
                paths=["A", "B"],
                file_pattern=r".*\.txt",
            ),
            ReadFilePass(base=str(tmp_path)),
        ]
        result = list(process_texts([], passes))
        assert sorted(result) == ["aaa", "bbb"]

    def test_for_each_split_strip_join(self):
        passes = [
            ForEachPass(
                passes=[
                    SplitPass(separator="\n"),
                    StripPass(),
                    JoinPass(separator="\n"),
                ]
            )
        ]
        result = list(process_texts(["  a  \n  b  ", "  c  \n  d  "], passes))
        assert result == ["a\nb", "c\nd"]

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
            {
                "passes": [
                    {
                        "name": "ref",
                        "base": str(tmp_path),
                        "paths": ["ops.json"],
                    }
                ]
            }
        ).passes
        result = list(process_texts([], passes))
        assert result == ["hello ref"]
