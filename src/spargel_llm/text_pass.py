# Don't forget to update `docs/text-pass.md`.

import gzip
import itertools
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Iterable, Iterator, Literal, Sequence, override

import regex
from pydantic import BaseModel, Discriminator
from regex import Pattern

from spargel_llm.typing import StrOrPath

logger = logging.getLogger(__name__)


def _resolve_parent(path: StrOrPath):
    p = Path(path).resolve()
    return p.parent if p.is_file() else p


# Base


class TextPassInstance:
    """Runtime text pass."""

    def process(self, texts: Iterable[str]) -> Iterator[str]:
        raise NotImplementedError


class TextPass:
    """Text pass factory."""

    def build(self, config_path: StrOrPath) -> TextPassInstance:
        raise NotImplementedError


class TextPassModel(TextPass, BaseModel):
    """Text pass for Pydantic validation."""

    description: str | None = None


# Passes


@dataclass
class ChainPassInstance(TextPassInstance):
    """Chain multiple instances sequentially."""

    instances: Sequence[TextPassInstance]

    @override
    def process(self, texts):
        current = iter(texts)
        for inst in self.instances:
            current = inst.process(current)
        yield from current


@dataclass
class CombinePassInstance(TextPassInstance):
    instances: Sequence[TextPassInstance]

    @override
    def process(self, texts):
        if not self.instances:
            return
        copies = itertools.tee(texts, len(self.instances))
        iterators = [inst.process(c) for inst, c in zip(self.instances, copies)]
        # Round-robin interleave — order among passes is not significant.
        while iterators:
            remaining = []
            for it in iterators:
                try:
                    yield next(it)
                    remaining.append(it)
                except StopIteration:
                    pass
            iterators = remaining


@dataclass
class CombinePass(TextPass):
    """Apply passes independently and concatenate results."""

    passes: Sequence[TextPass]

    @override
    def build(self, config_path):
        instances = [p.build(config_path) for p in self.passes]
        return CombinePassInstance(instances)


class CombinePassModel(TextPassModel):
    """Apply passes independently and concatenate results (Model)."""

    name: Literal["combine"] = "combine"
    passes: list[DiscriminatedTextPass]

    @override
    def build(self, config_path):
        instances = [p.build(config_path) for p in self.passes]
        return CombinePassInstance(instances)


class FilterPass(TextPassModel):
    """Filter texts by regex pattern."""

    name: Literal["filter"] = "filter"
    pattern: str
    invert: bool = False

    @override
    def build(self, config_path):
        pat = regex.compile(self.pattern)
        return self._Instance(pat, self.invert)

    @dataclass
    class _Instance(TextPassInstance):
        pattern: Pattern
        invert: bool

        @override
        def process(self, texts):
            for text in texts:
                matched = bool(self.pattern.search(text))
                if matched != self.invert:
                    yield text


class FindPass(TextPassModel):
    """Find files in a directory."""

    name: Literal["find"] = "find"
    base: str = "."
    paths: list[str] = ["."]
    file_pattern: str | None = None
    dir_pattern: str | None = None

    @override
    def build(self, config_path):
        base = _resolve_parent(config_path) / self.base
        file_pat = regex.compile(self.file_pattern) if self.file_pattern else None
        dir_pat = regex.compile(self.dir_pattern) if self.dir_pattern else None
        return self._Instance(base, self.paths, file_pat, dir_pat)

    @dataclass
    class _Instance(TextPassInstance):
        base: Path
        paths: list[str]
        file_pattern: Pattern | None
        dir_pattern: Pattern | None

        @override
        def process(self, texts):
            for path in self.paths:
                yield from (str(p) for p in self._search_dir(self.base / path))
            yield from texts

        def _search_dir(self, dir: StrOrPath) -> Iterator[Path]:
            for cur_dir, dirs, files in os.walk(dir):
                if self.dir_pattern is not None:
                    dirs[:] = [d for d in dirs if self.dir_pattern.fullmatch(d)]
                for file in files:
                    if (
                        self.file_pattern is not None
                        and not self.file_pattern.fullmatch(file)
                    ):
                        continue
                    yield Path(cur_dir) / file


@dataclass
class ForEachPassInstance(TextPassInstance):
    instances: Sequence[TextPassInstance]

    @override
    def process(self, texts):
        for text in texts:
            current = (text,)
            for inst in self.instances:
                current = inst.process(current)
            yield from current
            del current, text  # release before fetching next text


@dataclass
class ForEachPass(TextPass):
    """Apply passes to each text."""

    passes: Sequence[TextPass]

    @override
    def build(self, config_path):
        instances = [p.build(config_path) for p in self.passes]
        return ForEachPassInstance(instances)


class ForEachPassModel(TextPassModel):
    """Apply passes to each text (Model)."""

    name: Literal["for_each"] = "for_each"
    passes: list[DiscriminatedTextPass]

    @override
    def build(self, config_path):
        instances = [p.build(config_path) for p in self.passes]
        return ForEachPassInstance(instances)


class JSONPass(TextPassModel):
    """Parse each text as JSON and extract a key field."""

    name: Literal["json"] = "json"
    key: str

    @override
    def build(self, config_path):
        return self._Instance(self.key)

    @dataclass
    class _Instance(TextPassInstance):
        key: str

        @override
        def process(self, texts):
            for text in texts:
                obj = json.loads(text)
                value = obj[self.key]
                if not isinstance(value, str):
                    raise TypeError(
                        f'JSON key "{self.key}" is not a string: {type(value).__name__}'
                    )
                yield value


class JoinPass(TextPassModel):
    """Join texts."""

    name: Literal["join"] = "join"
    separator: str = ""

    @override
    def build(self, config_path):
        return self._Instance(self.separator)

    @dataclass
    class _Instance(TextPassInstance):
        separator: str

        @override
        def process(self, texts):
            yield self.separator.join(texts)


class PlainTextPass(TextPassModel):
    """Provide a list of plain texts directly."""

    name: Literal["text"] = "text"
    texts: list[str]

    @override
    def build(self, config_path):
        return self._Instance(self.texts)

    @dataclass
    class _Instance(TextPassInstance):
        texts: list[str]

        @override
        def process(self, texts):
            return itertools.chain(texts, self.texts)


class ReadFilePass(TextPassModel):
    """Read file content.

    When ``lines`` is true, yields individual lines (via ``readline()`` loop).
    """

    name: Literal["read_file"] = "read_file"
    base: str = "."
    encoding: str | None = None
    compression: Literal["gzip"] | None = None
    lines: bool = False

    @override
    def build(self, config_path):
        base = _resolve_parent(config_path) / self.base
        return self._Instance(base, self.encoding, self.compression, self.lines)

    @dataclass
    class _Instance(TextPassInstance):
        base: Path
        encoding: str | None
        compression: Literal["gzip"] | None
        lines: bool

        @override
        def process(self, texts):
            for text in texts:
                path = self.base / text
                try:
                    match self.compression:
                        case "gzip":
                            with gzip.open(path, "rt", encoding=self.encoding) as f:
                                yield from self._read(f)
                        case _:
                            with open(path, "r", encoding=self.encoding) as f:
                                yield from self._read(f)
                except UnicodeDecodeError:
                    logger.warning(f'UnicodeDecodeError in file: "{path}"')

        def _read(self, f):
            if self.lines:
                while line := f.readline():
                    yield line
            else:
                yield f.read()


class ReferencePass(TextPassModel):
    """Reference external passes."""

    name: Literal["ref"] = "ref"
    base: str = "."
    paths: list[str]

    @override
    def build(self, config_path):
        base = _resolve_parent(config_path) / self.base
        instances: list[TextPassInstance] = []
        for rel in self.paths:
            ref_path = base / rel
            with open(ref_path, encoding="utf-8") as f:
                obj = json.load(f)
            ref_passes = TextPassList.model_validate(obj).passes
            for p in ref_passes:
                instances.append(p.build(ref_path))
        return ChainPassInstance(instances)


class ReplacePass(TextPassModel):
    """Replace All"""

    name: Literal["replace"] = "replace"
    regex: bool = False
    old: str
    new: str
    repeat: bool = False
    max_repeat: int = 1000

    @override
    def build(self, config_path):
        return self._Instance(
            self.regex, self.old, self.new, self.repeat, self.max_repeat
        )

    @dataclass
    class _Instance(TextPassInstance):
        regex: bool
        old: str
        new: str
        repeat: bool
        max_repeat: int

        @override
        def process(self, texts):
            for text in texts:
                yield self._apply_replace(text)

        def _apply_replace(self, text: str) -> str:
            if self.repeat:
                for _ in range(self.max_repeat):
                    last_text = text
                    if self.regex:
                        text = regex.sub(self.old, self.new, text)
                    else:
                        text = text.replace(self.old, self.new)

                    if text == last_text:
                        return text
                return text
            else:
                if self.regex:
                    return regex.sub(self.old, self.new, text)
                else:
                    return text.replace(self.old, self.new)


def _literal_positions(text: str, sep: str) -> Iterator[tuple[int, int, None]]:
    """Yield (start, end, None) of each occurrence of *sep* in *text*."""
    if not sep:
        return
    start = 0
    sep_len = len(sep)
    while (end := text.find(sep, start)) != -1:
        yield (end, end + sep_len, None)
        start = end + sep_len


def _split_iter(
    positions: Iterator[tuple[int, int, tuple[str, ...] | None]],
    text: str,
    max_split: int,
    behavior: str,
) -> Iterator[str]:
    """Yield slices of *text* split at *positions*, lazy, per *behavior*."""
    start = 0
    limit = max_split if max_split > 0 else None
    prev_start = None

    for count, (p_start, p_end, groups) in enumerate(positions):
        if limit is not None and count >= limit:
            break
        if behavior == "removed":
            yield text[start:p_start]
            if groups:
                yield from groups
        elif behavior == "isolated":
            yield text[start:p_start]
            yield text[p_start:p_end]
        elif behavior == "merged_with_previous":
            yield text[start:p_end]
        else:  # merged_with_next
            yield text[prev_start if prev_start is not None else start : p_start]
            prev_start = p_start
        start = p_end

    if behavior == "merged_with_next":
        yield text[prev_start if prev_start is not None else start :]
    else:
        yield text[start:]


class SplitPass(TextPassModel):
    """Split text by a separator or regex pattern."""

    name: Literal["split"] = "split"
    separator: str
    regex: bool = False
    max_split: int = 0
    behavior: Literal[
        "removed", "isolated", "merged_with_previous", "merged_with_next"
    ] = "removed"

    @override
    def build(self, config_path):
        if self.regex:
            pat = regex.compile(self.separator)
        else:
            pat = None

        if self.behavior != "removed":
            if self.regex and pat:
                cap_pat = regex.compile(f"({self.separator})", pat.flags)
            else:
                cap_pat = regex.compile(f"({regex.escape(self.separator)})")
        else:
            cap_pat = None

        return self._Instance(
            self.regex, pat, cap_pat, self.separator, self.max_split, self.behavior
        )

    @dataclass
    class _Instance(TextPassInstance):
        regex: bool
        pattern: Pattern | None
        cap_pattern: Pattern | None
        separator: str
        max_split: int
        behavior: str

        @override
        def process(self, texts):
            for text in texts:
                if self.behavior == "removed":
                    if self.regex and self.pattern:
                        positions = (
                            (m.start(), m.end(), m.groups())
                            for m in self.pattern.finditer(text)
                        )
                    else:
                        positions = _literal_positions(text, self.separator)
                elif self.cap_pattern:
                    positions = (
                        (m.start(), m.end(), None)
                        for m in self.cap_pattern.finditer(text)
                    )
                else:
                    raise RuntimeError("no self.cap_pattern")
                yield from _split_iter(positions, text, self.max_split, self.behavior)


class StripPass(TextPassModel):
    """Strip text (remove leading and trailing whitespace)."""

    name: Literal["strip"] = "strip"
    chars: str | None = None
    right: bool = False

    @override
    def build(self, config_path):
        return self._Instance(self.chars, self.right)

    @dataclass
    class _Instance(TextPassInstance):
        chars: str | None
        right: bool

        @override
        def process(self, texts):
            for text in texts:
                if self.right:
                    yield text.rstrip(self.chars)
                else:
                    yield text.strip(self.chars)


# Types

type TextPassModelUnion = (
    CombinePassModel
    | FilterPass
    | FindPass
    | ForEachPassModel
    | JoinPass
    | JSONPass
    | PlainTextPass
    | ReadFilePass
    | ReferencePass
    | ReplacePass
    | SplitPass
    | StripPass
)
type DiscriminatedTextPass = Annotated[TextPassModelUnion, Discriminator("name")]


class TextPassList(BaseModel):
    passes: list[DiscriminatedTextPass]


# Public API


def process_texts(
    texts: Iterable[str], passes: Sequence[TextPass], path: StrOrPath = "."
) -> Iterator[str]:
    instances = [p.build(path) for p in passes]
    yield from ChainPassInstance(instances).process(texts)


def load_texts(path: StrOrPath) -> Iterator[str]:
    with open(path, "r") as f:
        obj = json.load(f)
    passes = TextPassList.model_validate(obj).passes
    return process_texts([], passes, path)
