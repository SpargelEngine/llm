#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path

from spargel_llm.logging import log_error


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def run_git(args: list[str]) -> str | None:
    result = subprocess.run(
        ["git"] + args,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def check_git_status() -> bool:
    """Return True if the working tree is up to date with remote."""
    output = run_git(["status", "--porcelain"])
    if output is None or output != "":
        return False
    output = run_git(
        ["log", "@{u}..HEAD", "--oneline"],
    )
    if output is None or output != "":
        return False
    return True


def main():
    if not check_git_status():
        log_error("Commit and push your changes before running an experiment.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
