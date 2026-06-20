from typing import Literal, override

import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from pydantic import BaseModel

class FeedForwardConfig(BaseModel):
    dim: int
    hidden: int
    activation: Literal["relu", "relu2"]

class FeedForward(nn.Module):
    """
    Feed Forward
    """

    def __init__(self, config: FeedForwardConfig):
        super().__init__()
        self.config = config

        self.up = nn.Linear(self.config.dim, self.config.hidden)
        self.down = nn.Linear(self.config.hidden, self.config.dim)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.up(x)
        if self.config.activation == "relu":
            x = F.relu(x)
        elif self.config.activation == "relu2":
            x = F.relu(x).square()
        else:
            raise ValueError("unknown activation: " + self.config.activation)
        x = self.down(x)
        return x

    @override
    def extra_repr(self) -> str:
        return f"dim={self.config.dim}, hidden={self.config.hidden}, activation={self.config.activation!r}"
