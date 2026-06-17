from os.path import isdir, isfile
from typing import Container

from spargel_llm.logging import log_error
from spargel_llm.typing import StrOrPath


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
