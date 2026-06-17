from typing import override

import torch
import torch.nn as nn
from pydantic import BaseModel, PositiveInt
from torch import Tensor

from spargel_llm.layers.torch import (
    Attention,
    FeedForward,
    PositionalEncoding,
    RMSNorm,
)


class Config(BaseModel):
    vocab_size: PositiveInt
    max_seq_len: PositiveInt
    num_layer: PositiveInt
    num_head: PositiveInt
    dim: PositiveInt
    dim_key: PositiveInt
    dim_value: PositiveInt
    dim_feed_forward: PositiveInt


class TransformerBlock(nn.Module):
    """
    One Transformer Block
    """

    def __init__(self, config: Config):
        super().__init__()

        self.norm1 = RMSNorm(config.dim)
        self.norm2 = RMSNorm(config.dim)

        self.attention = Attention(
            num_head=config.num_head,
            dim_in=config.dim,
            dim_out=config.dim,
            dim_k=config.dim_key,
            dim_v=config.dim_value,
        )

        self.feed_forward = FeedForward(config.dim, config.dim_feed_forward)

    @override
    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        y = x + self.attention(self.norm1(x), causal=True, mask=mask)
        y = y + self.feed_forward(self.norm2(y))
        return y


class Model(nn.Module):
    """
    The Language Model
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.positional_encoding = PositionalEncoding(config.max_seq_len, config.dim)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.num_layer)
        )
        self.out = nn.Linear(config.dim, config.vocab_size)

    @override
    def forward(self, tokens: Tensor, mask: Tensor | None = None) -> Tensor:
        x = self.embedding(tokens)
        x = self.positional_encoding(x)
        for block in self.blocks:
            x = block(x, mask)
        x = self.out(x)
        return x

    @override
    def extra_repr(self) -> str:
        return str(self.config)

    def loss(
        self, input_ids: Tensor, mask: Tensor, target_ids: Tensor, pad_index: int
    ) -> Tensor:
        """
        Args:
            input_ids (..., seq_len) dtype=int
            mask (..., seq_len) dtype=bool
            target_ids (..., seq_len) dtype=int
        """

        logits: Tensor = self(input_ids, mask)  # (..., seq_len, vocab_size)
        loss = nn.functional.cross_entropy(
            logits.flatten(0, -2),
            target_ids.flatten(0, -1).to(torch.long),
            ignore_index=pad_index,
        )

        return loss
