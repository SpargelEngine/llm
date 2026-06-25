def escape_whitespace(s: str) -> str:
    """Replace invisible whitespace characters with visible escape sequences."""
    return (
        s.replace("\t", "\\t")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\x0c", "\\f")
    )
