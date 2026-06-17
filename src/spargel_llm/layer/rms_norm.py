# Reference: Root Mean Square Layer Normalization (arXiv:1910.07467)

from typing import override

import torch
import torch.nn as nn
from torch import Tensor


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    """

    def __init__(self, dim: int, epsilon: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.epsilon = epsilon

        self.weight = nn.Parameter(torch.ones(dim))

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (..., dim)

        Returns:
            (..., dim)
        """

        var = x.square().mean(-1, keepdim=True)
        y = x * torch.rsqrt(var + self.epsilon)
        y = self.weight * y
        return y

    @override
    def extra_repr(self) -> str:
        return f"dim={self.dim}, epsilon={self.epsilon}"
