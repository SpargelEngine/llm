import math

import numpy as np


def _generate_position_encoding(
    ctx_len: int, dim: int, frequencies: np.ndarray
) -> np.ndarray:
    """
    Args:
        ctx_len (int): expected max length of context
        dim (int): dimension of vector
    """

    assert ctx_len > 0 and dim > 0
    assert dim % 2 == 0

    position_encoding = np.zeros((ctx_len, dim), dtype=float)

    positions = np.arange(0, ctx_len, dtype=float).reshape((ctx_len, 1))

    position_encoding[:, 0::2] = np.sin(positions * frequencies)
    position_encoding[:, 1::2] = np.cos(positions * frequencies)

    return position_encoding


def generate_position_encoding0(ctx_len: int, dim: int) -> np.ndarray:
    """
    Args:
        ctx_len (int): expected max length of context
        dim (int): dimension of vector
    """

    return _generate_position_encoding(
        ctx_len, dim, ctx_len ** (-np.arange(0, dim // 2) * 2 / dim)
    )


def generate_position_encoding1(ctx_len: int, dim: int) -> np.ndarray:
    """
    Args:
        ctx_len (int): expected max length of context
        dim (int): dimension of vector
    """

    return _generate_position_encoding(
        ctx_len, dim, np.arange(1, dim // 2 + 1, dtype=float) * math.pi / ctx_len
    )
