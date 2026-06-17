from typing import override

import torch.nn as nn
from torch import Tensor


class FeedForward(nn.Module):
    """
    Feed Forward
    """

    def __init__(self, dim: int, dim_hidden: int):
        super().__init__()
        self.dim = dim
        self.dim_hidden = dim_hidden

        self.layers = nn.Sequential(
            nn.Linear(in_features=dim, out_features=dim_hidden),
            nn.ReLU(),
            nn.Linear(in_features=dim_hidden, out_features=dim),
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

    @override
    def extra_repr(self) -> str:
        return f"dim={self.dim}, dim_hidden={self.dim_hidden}"
