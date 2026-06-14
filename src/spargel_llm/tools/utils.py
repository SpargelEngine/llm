import gzip
import pickle
from os.path import isdir, isfile
from typing import Container

from .logging import log_error
from .typing import StrOrPath


def load_gzip_pickle(path: StrOrPath):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def save_gzip_pickle(path: StrOrPath, obj):
    with gzip.open(path, "wb") as f:
        pickle.dump(obj, f)


class PromptAbortError(Exception):
    pass


YES_STRINGS = ["y", "yes", "ok"]
NO_STRINGS = ["n", "no"]


def prompt_overwrite(
    path: StrOrPath,
    yes: bool = False,
    *,
    yes_strings: Container[str] = YES_STRINGS,
):
    if isdir(path):
        log_error(f"Is a directory: {path}")
        raise IsADirectoryError
    elif not yes and isfile(path):
        response = input(f"File {path} already exists. Overwrite? (y/n): ")
        if response.strip().lower() not in yes_strings:
            raise PromptAbortError
