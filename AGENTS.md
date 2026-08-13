# AGENTS.md

## Workflow: write locally, execute remotely

- **Code, tests, and commits happen on the local machine.**
- **Execution happens on the remote server** (`autodl`) through `scripts/remote.sh`,
  which syncs the repo over SSH with `rsync` and then runs the given command there.
- **Tests run on the remote server too**:

  ```bash
  ./scripts/remote.sh ~/miniconda3/bin/python -m pytest -q
  ```

There is no manual code upload: `scripts/remote.sh` is the only way to get code onto
the server. Always run it after changing code that needs to run remotely.

## Remote server (`autodl`)

- SSH host: `autodl` (see `~/.ssh/config`); repo lives at `~/autodl-tmp/llm`.
- GPU: NVIDIA RTX 4080 SUPER (32 GB VRAM).
- The server provides a conda base environment (`~/miniconda3/bin/python`, Python 3.12.3)
  with `torch 2.12.1+cu130` (CUDA available) and the project installed in editable mode
  (installed with `pip install --ignore-requires-python -e .` since the server Python is
  older than the `>=3.14` requirement).
- Run remote Python with `~/miniconda3/bin/python`.

## Usage

```bash
# Sync code, then run a command on the server
./scripts/remote.sh <command...>

# Sync only (no command)
./scripts/remote.sh
```

Environment overrides: `SERVER_HOST` (default `autodl`), `SERVER_DIR`
(default `~/autodl-tmp/llm`).

## Notes

- The sync excludes `__pycache__`, `.venv`, `.pytest_cache`, and `*.pyc`, and uses
  `--delete`, so do not keep files on the server inside the repo directory that are
  not in the local tree.
- The project targets Python >= 3.14, but the server runs 3.12.3; a few 3.14-only
  features are shimmed for 3.12 compatibility (`spargel_llm/zstd.py` provides the
  `compression.zstd` stdlib module via the `zstandard` package, and `text_pass.py`
  uses `from __future__ import annotations` to support forward references under
  eager annotation evaluation).
