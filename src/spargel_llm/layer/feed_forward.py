from typing import Literal, override

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FeedForward(nn.Module):
    """
    Feed Forward
    """

    type Activation = Literal["relu", "relu2"]

    def __init__(self, dim: int, dim_hidden: int, activation: Activation):
        super().__init__()
        self.activation = activation

        self.up = nn.Linear(dim, dim_hidden)
        self.down = nn.Linear(dim_hidden, dim)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.up(x)
        if self.activation == "relu":
            x = F.relu(x)
        elif self.activation == "relu2":
            x = F.relu(x).square()
        else:
            raise ValueError("unknown activation: " + self.activation)
        x = self.down(x)
        return x

    @override
    def extra_repr(self) -> str:
        return f"dim={self.dim}, hidden={self.hidden}, activation={self.activation!r}"
