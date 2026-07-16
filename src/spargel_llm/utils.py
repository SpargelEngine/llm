def escape_whitespace(s: str) -> str:
    """Replace invisible whitespace characters with visible escape sequences."""
    return (
        s.replace("\t", "\\t")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\x0c", "\\f")
    )


def format_flops(v: int | float) -> str:
    """Format a FLOP count into a human-readable string with SI prefix.

    >>> format_flops(1500)
    '1,500 FLOPs'
    >>> format_flops(2.5e9)
    '2.50 GFLOPs'
    """
    if v >= 1e18:
        return f"{v / 1e18:,.2f} EFLOPs"
    if v >= 1e15:
        return f"{v / 1e15:,.2f} PFLOPs"
    if v >= 1e12:
        return f"{v / 1e12:,.2f} TFLOPs"
    if v >= 1e9:
        return f"{v / 1e9:,.2f} GFLOPs"
    if v >= 1e6:
        return f"{v / 1e6:,.2f} MFLOPs"
    return f"{v:,} FLOPs"


def format_bytes(v: int) -> str:
    """Format a byte count with an appropriate binary unit.

    >>> format_bytes(42_016_768)
    '40.07 MiB'
    >>> format_bytes(26_662_092_800)
    '24.83 GiB'
    """
    if v >= 1024**4:
        return f"{v / (1024**4):,.2f} TiB"
    if v >= 1024**3:
        return f"{v / (1024**3):,.2f} GiB"
    if v >= 1024**2:
        return f"{v / (1024**2):,.2f} MiB"
    if v >= 1024:
        return f"{v / 1024:,.2f} KiB"
    return f"{v:,} bytes"
