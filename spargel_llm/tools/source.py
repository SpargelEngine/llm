import json
from os.path import dirname
from pathlib import Path
from typing import Literal, Optional

import regex
from regex import Pattern
from pydantic import BaseModel, Field

from .typing import StrOrPath


class FileList(BaseModel):
    """Get texts from listed files.

    Each file's content is treated as a single text.
    """

    type: Literal["file_list"]
    base: str = "."
    paths: list[str]


class Find(BaseModel):
    """Find files in a directory.

    Similar to FileList. Here the file list is automatically discovered.
    """

    type: Literal["find"]
    base: str
    file_pattern: Optional[str]
    dir_pattern: Optional[str]


class Reference(BaseModel):
    """Reference to a list of source description files."""

    type: Literal["reference"]
    base: str = "."
    paths: list[str]


class Lines(BaseModel):
    """Get lines.

    Each line is a text.
    """

    type: Literal["lines"]
    sources: list["Source"]


class Regex(BaseModel):
    """Get all matched sub-texts in each text of each source using a regular expression."""

    type: Literal["regex"]
    pattern: str
    sources: list["Source"]


type Source = FileList | Find | Reference | Lines | Regex


class Model(BaseModel):
    source: Source = Field(discriminator="type")


def _search_dir(
    texts: list[str],
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
                texts.append(f.read())
        elif path.is_dir():
            if dir_pattern is not None and not dir_pattern.fullmatch(path.name):
                continue

            _search_dir(texts, path, file_pattern=file_pattern, dir_pattern=dir_pattern)


def _get_texts(source: Source, path: StrOrPath) -> list[str]:
    texts = []

    match source.type:
        case "file_list":
            for _path in source.paths:
                real_path = Path(dirname(path), source.base, _path)
                with open(real_path, "r") as f:
                    texts.append(f.read())

        case "find":
            search_dir = Path(dirname(path), source.base)

            dir_pattern = (
                regex.compile(source.dir_pattern) if source.dir_pattern else None
            )
            file_pattern = (
                regex.compile(source.file_pattern) if source.file_pattern else None
            )

            _search_dir(
                texts, search_dir, file_pattern=file_pattern, dir_pattern=dir_pattern
            )

        case "reference":
            for _path in source.paths:
                real_path = Path(dirname(path), source.base, _path)
                texts.extend(get_texts(real_path))

        case "lines":
            for child_source in source.sources:
                for text in _get_texts(child_source, path):
                    texts.extend(text.splitlines())

        case "regex":
            for child_source in source.sources:
                for text in _get_texts(child_source, path):
                    texts.extend(regex.findall(source.pattern, text))

    return texts


def get_texts(path: StrOrPath) -> list[str]:
    with open(path, "r") as f:
        obj = json.load(f)

    source = Model.model_validate({"source": obj}).source

    return _get_texts(source, path)
