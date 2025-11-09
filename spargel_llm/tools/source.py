import json
from os.path import dirname
from pathlib import Path
from typing import Iterable, Literal, Optional, override

import regex
from pydantic import BaseModel, Field
from regex import Pattern

from spargel_llm.meta import ai_marker

from .typing import StrOrPath


def resolve_parent(path: StrOrPath):
    return Path(path).resolve().parent


class SourceModel(BaseModel):
    """Base class for all source models"""

    comment: str = ""

    def get_texts(self, this_path: StrOrPath) -> Iterable[str]: ...


class FileListSource(SourceModel):
    """Get texts from listed files.

    Each file's content is treated as a single text.
    If a path starts with '@', it is treated as a file containing a list of file paths.
    """

    type: Literal["file_list"]
    base: str = "."
    paths: list[str]

    @override
    def get_texts(self, this_path):
        for path in self.paths:
            if path.startswith("@"):
                list_file_path = resolve_parent(this_path) / self.base / path[1:]
                with open(list_file_path, "r") as f:
                    for line in f:
                        file_path = line.strip()
                        if file_path:
                            real_path = Path(
                                dirname(list_file_path), file_path
                            ).resolve()
                            with open(real_path, "r") as file_f:
                                yield file_f.read()
            else:
                real_path = resolve_parent(this_path) / self.base / path
                with open(real_path, "r") as f:
                    yield f.read()


class FindSource(SourceModel):
    """Find files in a directory.

    Similar to FileList. Here the file list is automatically discovered.
    """

    type: Literal["find"]
    file_pattern: Optional[str] = None
    dir_pattern: Optional[str] = None
    bases: list[str]

    @override
    def get_texts(self, this_path):
        dir_pattern = regex.compile(self.dir_pattern) if self.dir_pattern else None
        file_pattern = regex.compile(self.file_pattern) if self.file_pattern else None

        for base in self.bases:
            search_dir = resolve_parent(this_path) / base

            yield from self._search_dir(
                search_dir, file_pattern=file_pattern, dir_pattern=dir_pattern
            )

    def _search_dir(
        self,
        dir: StrOrPath,
        *,
        file_pattern: Optional[Pattern[str]] = None,
        dir_pattern: Optional[Pattern[str]] = None,
    ):
        for path in Path(dir).iterdir():
            if path.is_file():
                if file_pattern is not None and not file_pattern.fullmatch(path.name):
                    continue

                with open(path, "r") as f:
                    yield f.read()

            elif path.is_dir():
                if dir_pattern is not None and not dir_pattern.fullmatch(path.name):
                    continue

                yield from self._search_dir(
                    path, file_pattern=file_pattern, dir_pattern=dir_pattern
                )


class ReferenceSource(SourceModel):
    """Reference to a list of source description files."""

    type: Literal["reference"]
    base: str = "."
    paths: list[str]

    @override
    def get_texts(self, this_path):
        for path in self.paths:
            ref_path = resolve_parent(this_path) / self.base / path
            yield from get_texts(ref_path)


class LinesSource(SourceModel):
    """Get lines.

    Each line is a text.
    """

    type: Literal["lines"]
    sources: list["Source"]

    @override
    def get_texts(self, this_path):
        for source in self.sources:
            for text in source.get_texts(this_path):
                yield from text.splitlines()


class RegexSource(SourceModel):
    """Extract all matched sub-texts in each text of each source using a regular expression."""

    type: Literal["regex"]
    pattern: str
    sources: list["Source"]

    @override
    def get_texts(self, this_path):
        for source in self.sources:
            for text in source.get_texts(this_path):
                yield from regex.findall(self.pattern, text)


class LengthFilterSource(SourceModel):
    """Keep only the texts whose lengths are in the range."""

    type: Literal["length_filter"]
    min_length: int = 1
    max_length: int = 0
    sources: list["Source"]

    @override
    def get_texts(self, this_path):
        for source in self.sources:
            for text in source.get_texts(this_path):
                if len(text) >= self.min_length:
                    if self.max_length <= 0 or len(text) <= self.max_length:
                        yield text


class ProcessSource(SourceModel):
    """Apply operations in order to each text."""

    type: Literal["process"]
    operations: list["Operation"]
    sources: list["Source"]

    @override
    def get_texts(self, this_path):
        for source in self.sources:
            for text in source.get_texts(this_path):
                for operation in self.operations:
                    text = operation.process(text, this_path)
                yield text


@ai_marker(human_checked=True)
class DuplicateFilterSource(SourceModel):
    """Filter out duplicate texts based on content hash"""

    type: Literal["duplicate_filter"]
    sources: list["Source"]

    @override
    def get_texts(self, this_path):
        seen_hashes = set()
        for source in self.sources:
            for text in source.get_texts(this_path):
                text_hash = hash(text)
                if text_hash not in seen_hashes:
                    seen_hashes.add(text_hash)
                    yield text


class SplitSource(SourceModel):
    """Split texts"""

    type: Literal["split"]
    separator: Optional[str] = None
    sources: list["Source"]

    @override
    def get_texts(self, this_path):
        for source in self.sources:
            for text in source.get_texts(this_path):
                yield from text.split(self.separator)


type Source = FileListSource | FindSource | ReferenceSource | LinesSource | RegexSource | LengthFilterSource | ProcessSource | DuplicateFilterSource | SplitSource


class SourceWrapperModel(BaseModel):
    source: Source = Field(discriminator="type")


class OperationModel(BaseModel):
    """Base class for operation models"""

    comment: str = ""

    def process(self, text: str, this_path: StrOrPath) -> str: ...


class ReferenceOperation(OperationModel):
    """Reference to a list of operation description files

    Each file contains a list of operations.
    """

    type: Literal["reference"]
    base: str = "."
    paths: list[str]

    @override
    def process(self, text, this_path):
        for path in self.paths:
            ref_path = resolve_parent(this_path) / self.base / path
            for operation in self._get_operations(ref_path):
                text = operation.process(text, ref_path)
        return text

    def _get_operations(self, path: StrOrPath) -> Iterable["Operation"]:
        with open(path, "r") as f:
            array = json.load(f)

        for obj in array:
            yield OperationWrapperModel.model_validate({"operation": obj}).operation


class StripOperation(OperationModel):
    """Strip whitespace from text"""

    type: Literal["strip"]
    chars: Optional[str] = None
    per_line: bool = False

    @override
    def process(self, text, this_path):
        if self.per_line:
            return "\n".join(line.strip(self.chars) for line in text.splitlines())
        else:
            return text.strip(self.chars)


@ai_marker(human_checked=True)
class RegexReplaceOperation(OperationModel):
    """Replace all matches of a regex pattern with replacement string"""

    type: Literal["regex_replace"]
    pattern: str
    replacement: str
    repeat: bool = False
    per_line: bool = False

    @override
    def process(self, text, this_path):
        if self.per_line:
            return "\n".join(self._apply_replace(line) for line in text.splitlines())
        else:
            return self._apply_replace(text)

    def _apply_replace(self, text: str) -> str:
        if self.repeat:
            last_text = text
            while True:
                text = regex.sub(self.pattern, self.replacement, text)
                if text == last_text:
                    break
                last_text = text
            return text
        else:
            return regex.sub(self.pattern, self.replacement, text)


class RemoveShortLinesOperation(OperationModel):
    """Remove lines shorter than the given minimum length"""

    type: Literal["remove_short_lines"]
    min_length: int = 1

    @override
    def process(self, text, this_path):
        return "\n".join(
            line for line in text.splitlines() if len(line) >= self.min_length
        )


type Operation = ReferenceOperation | StripOperation | RegexReplaceOperation | RemoveShortLinesOperation


class OperationWrapperModel(BaseModel):
    operation: Operation = Field(discriminator="type")


def get_texts(path: StrOrPath) -> Iterable[str]:
    with open(path, "r") as f:
        obj = json.load(f)

    source = SourceWrapperModel.model_validate({"source": obj}).source

    return source.get_texts(path)
