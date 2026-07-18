"""Convert TensorBoard logs: use cumulative elapsed time as the x-axis (global step).

Reads TensorBoard event files from a source directory, computes the cumulative
elapsed time from ``metric/time/elapsed`` scalars (already cumulative in the
training loop), and writes loss and perplexity metrics to a destination
directory with that elapsed time (in milliseconds) as the global step.

If *src* is a TensorBoard directory (contains event files), it is converted
directly.  If *src* contains nested subdirectories that are TensorBoard runs,
the directory structure is mirrored under *dst* and each run is converted in
parallel.

Usage::

    python scripts/retime_tb.py <src_dir> <dst_dir>
"""

import os
import struct
from argparse import ArgumentParser
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from torch.utils.tensorboard import SummaryWriter

_TB_FILE_PREFIX = "events.out.tfevents"
_ELAPSED_TAG = "metric/time/elapsed"
_KEPT_PREFIXES = ("loss/", "perplexity/")


def _is_tb_dir(path: Path) -> bool:
    """Return True if *path* is a directory containing TensorBoard event files."""
    if not path.is_dir():
        return False
    return any(
        entry.is_file() and entry.name.startswith(_TB_FILE_PREFIX)
        for entry in path.iterdir()
    )


def _find_tb_dirs(root: Path) -> list[Path]:
    """Walk *root* and return subdirectories that contain event files.

    Directories with event files are treated as leaf runs — the walk does not
    descend into them.
    """
    tb_dirs: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        if any(f.startswith(_TB_FILE_PREFIX) for f in filenames):
            tb_dirs.append(Path(dirpath))
            dirnames.clear()
    return tb_dirs


def _list_event_files(src_dir: str) -> list[str]:
    """Return sorted paths to TensorBoard event files in *src_dir*."""
    files: list[str] = []
    for entry in os.scandir(src_dir):
        if entry.is_file() and entry.name.startswith(_TB_FILE_PREFIX):
            files.append(entry.path)
    return sorted(files)


def _extract_scalar(value) -> float | None:
    """Extract a float from a Summary.Value proto.

    Handles the modern ``tensor.float_val`` encoding as well as the legacy
    ``simple_value`` field.
    """
    if value.HasField("simple_value"):
        return value.simple_value
    tensor = value.tensor
    if len(tensor.float_val) > 0:
        return tensor.float_val[0]
    if len(tensor.tensor_content) >= 4:
        return struct.unpack("<f", tensor.tensor_content[:4])[0]
    return None


def _load_scalars(src_dir: str) -> dict[str, list[tuple[int, float]]]:
    """Load loss and perplexity scalars (plus elapsed time for the x-axis).

    Iterates event files directly via ``EventFileLoader`` — no eager indexing
    overhead from ``EventAccumulator``.

    Returns:
        dict mapping tag name -> list of ``(step, value)``, in file order.
    """
    event_files = _list_event_files(src_dir)
    if not event_files:
        return {}

    scalars: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for path in event_files:
        loader = EventFileLoader(path)
        for event in loader.Load():
            if not event.HasField("summary"):
                continue
            for value in event.summary.value:
                tag = value.tag
                wanted = (tag == _ELAPSED_TAG or tag.startswith(_KEPT_PREFIXES))
                if not wanted:
                    continue
                scalar = _extract_scalar(value)
                if scalar is not None:
                    scalars[tag].append((event.step, scalar))

    return dict(scalars)


def _build_timeline(
    elapsed_events: list[tuple[int, float]],
) -> dict[int, float]:
    """Return a mapping from original step -> cumulative elapsed time in seconds.

    ``metric/time/elapsed`` values are already cumulative — ``TrainInfo.time``
    is updated via ``+= step_time`` in the training loop — so we simply index
    them by step.
    """
    # Keep the last value for any duplicate step.
    return dict(elapsed_events)


def _write_scalars(
    dst_dir: str,
    scalars: dict[str, list[tuple[int, float]]],
    step_to_cumtime: dict[int, float],
) -> int:
    """Write loss and perplexity scalars to *dst_dir* with cumulative elapsed time as the step.

    Returns the number of events written.
    """
    os.makedirs(dst_dir, exist_ok=True)
    writer = SummaryWriter(dst_dir)

    n_written = 0
    for tag, events in scalars.items():
        if tag == _ELAPSED_TAG:
            continue
        for step, value in events:
            cumtime = step_to_cumtime.get(step)
            if cumtime is None:
                continue
            # TensorBoard stores steps as int64 — use milliseconds to retain
            # sub-second precision.
            step_ms = int(round(cumtime * 1000))
            writer.add_scalar(tag, value, step_ms)
            n_written += 1

    writer.close()
    return n_written


def _convert_one(
    src_dir: str,
    dst_dir: str,
) -> tuple[int, int, int, float]:
    """Convert a single TB directory.  Worker entry-point for parallel execution.

    Returns:
        ``(n_written, n_total, n_tags, total_elapsed_seconds)``.
    """
    scalars = _load_scalars(src_dir)
    if not scalars:
        return (0, 0, 0, 0.0)

    elapsed = scalars.get(_ELAPSED_TAG, [])
    step_to_cumtime = _build_timeline(elapsed)

    n_written = _write_scalars(dst_dir, scalars, step_to_cumtime)
    n_kept_tags = sum(1 for t in scalars if t != _ELAPSED_TAG)
    n_total = sum(len(v) for t, v in scalars.items() if t != _ELAPSED_TAG)

    total = step_to_cumtime[max(step_to_cumtime)] if step_to_cumtime else 0.0

    return (n_written, n_total, n_kept_tags, total)


def _print_result(
    rel_path: str,
    n_written: int,
    n_total: int,
    n_tags: int,
    total_elapsed: float,
) -> None:
    """Print a single conversion result."""
    if n_tags == 0:
        print(f"  Skipped (no scalars): {rel_path}")
        return
    print(
        f"  {rel_path}: {n_written}/{n_total} events, "
        f"{n_tags} tags, {total_elapsed:.1f}s"
    )


def convert_tensorboard(src_dir: str, dst_dir: str) -> None:
    """Read scalars from *src_dir*, re-write with cumulative elapsed time as step to *dst_dir*.

    Convenience wrapper for single-run programmatic use.
    """
    result = _convert_one(src_dir, dst_dir)
    _print_result(Path(src_dir).name, *result)


def main() -> None:
    parser = ArgumentParser(
        description="Convert TensorBoard logs: use cumulative elapsed time as x-axis"
    )
    parser.add_argument("src", help="Source TensorBoard directory or a parent of TB runs")
    parser.add_argument("dst", help="Destination directory")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    # --- single run ---
    if _is_tb_dir(src):
        print(f"Converting single run: {src}")
        convert_tensorboard(str(src), str(dst))
        print(f"Done: {dst}")
        return

    # --- nested runs ---
    tb_dirs = _find_tb_dirs(src)
    if not tb_dirs:
        print(f"No TensorBoard event files found under: {src}")
        return

    work: list[tuple[str, Path, str]] = []
    for tb_dir in tb_dirs:
        rel = tb_dir.relative_to(src)
        work.append((str(tb_dir), dst / rel, str(rel)))

    print(f"Found {len(work)} TensorBoard run(s) under: {src}")

    n_errors = 0
    with ProcessPoolExecutor() as executor:
        future_to_rel = {
            executor.submit(_convert_one, w[0], str(w[1])): w[2]
            for w in work
        }
        for future in as_completed(future_to_rel):
            rel_path = future_to_rel[future]
            try:
                result = future.result()
                _print_result(rel_path, *result)
            except Exception as exc:
                print(f"  Failed: {rel_path} ({exc})")
                n_errors += 1

    suffix = f" ({n_errors} error(s))" if n_errors else ""
    print(f"Done: {dst}{suffix}")


if __name__ == "__main__":
    main()
