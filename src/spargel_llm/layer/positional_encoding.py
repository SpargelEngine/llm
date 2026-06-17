from typing import override

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    """
    Positional Encoding
    """

    def __init__(self, max_seq_len: int, dim: int):
        super().__init__()
        assert dim % 2 == 0, f"dim must be even, got {dim}"
        self.max_seq_len = max_seq_len
        self.dim = dim

        self.pe = nn.Buffer(torch.empty(max_seq_len, dim))

        positions = torch.arange(0, max_seq_len).reshape(max_seq_len, 1)
        frequencies = torch.arange(1, dim // 2 + 1) * torch.pi / max_seq_len

        self.pe[:, 0::2] = torch.sin(positions * frequencies)
        self.pe[:, 1::2] = torch.cos(positions * frequencies)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (..., seq_len, dim)

        Returns:
            (..., seq_len, dim)
        """

        seq_len = x.size(-2)
        x += self.pe[:seq_len, :]

        return x

    @override
    def extra_repr(self) -> str:
        return f"max_seq_len={self.max_seq_len}, dim={self.dim}"
