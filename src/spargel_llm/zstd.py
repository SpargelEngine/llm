"""Zstandard compatibility shim.

Python >= 3.14 ships the `compression.zstd` module in the standard library;
on older Pythons the third-party `zstandard` package provides the same API.
"""

try:
    from compression import zstd  # Python >= 3.14
except ImportError:  # Python < 3.14: use the zstandard package instead
    import importlib

    zstd = importlib.import_module("zstandard")
