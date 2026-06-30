from argparse import ArgumentParser
from pathlib import Path


BACKUP_PATTERNS = (
    ".info.json.*",
    ".model_state.pth.*",
    ".optimizer_state.pth.*",
)


def iter_backups(root: Path):
    seen: set[Path] = set()
    for pattern in BACKUP_PATTERNS:
        for path in root.rglob(pattern):
            if path in seen:
                continue
            seen.add(path)
            if path.is_file() or path.is_symlink():
                yield path


def format_bytes(size: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{size} B"
        value /= 1024.0
    raise AssertionError("unreachable")


def main():
    parser = ArgumentParser(
        description="Remove generated checkpoint backup files under ./exps."
    )
    parser.add_argument(
        "--root",
        default="exps",
        type=Path,
        help="experiment root to scan (default: exps)",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="delete matching files; without this flag the script only lists them",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="only print the summary",
    )
    args = parser.parse_args()

    root = args.root
    if not root.exists():
        raise SystemExit(f"{root} does not exist")
    if not root.is_dir():
        raise SystemExit(f"{root} is not a directory")

    backups = sorted(iter_backups(root))
    total_bytes = sum(path.stat().st_size for path in backups)

    if not args.quiet:
        action = "deleting" if args.delete else "would delete"
        for path in backups:
            print(f"{action}: {path}")

    if args.delete:
        for path in backups:
            path.unlink()

    action = "deleted" if args.delete else "matched"
    print(f"{action} {len(backups)} files ({format_bytes(total_bytes)})")
    if not args.delete:
        print("dry run only; pass --delete to remove these files")


if __name__ == "__main__":
    main()
