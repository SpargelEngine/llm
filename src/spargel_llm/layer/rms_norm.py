# Reference: Root Mean Square Layer Normalization (arXiv:1910.07467)

from typing import override

import torch
import torch.nn as nn
from torch import Tensor


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    """

    def __init__(self, dim: int, epsilon: float = 1e-6, use_fp32: bool = True):
        super().__init__()
        self.dim = dim
        self.epsilon = epsilon
        self.use_fp32 = use_fp32

        # NOTE(tianjiao):
        # 1. Experiment shows that learnable weights don't improve loss.
        # 2. With learnable weights we lose variance control.
        self.weight = 1
        # self.weight = nn.Parameter(torch.ones(dim))

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (..., dim)

        Returns:
            (..., dim)
        """

        if self.use_fp32:
            x = x.float()

        var = x.square().mean(-1, keepdim=True)
        y = x * torch.rsqrt(var + self.epsilon)
        y = self.weight * y
        return y

    @override
    def extra_repr(self) -> str:
        return f"dim={self.dim}, epsilon={self.epsilon}"
