from dataclasses import dataclass
from typing import Optional, override

import torch.nn as nn
from torch import Tensor

from spargel_llm.layers.torch import (
    PositionalEncoding,
    LayerNorm,
    Attention,
    FeedForward,
)


@dataclass
class Config:
    vocab_size: int
    max_seq_len: int
    cnt_layer: int
    cnt_head: int
    dim: int
    d_key: int
    d_value: int
    d_feed_forward: int


class TransformerBlock(nn.Module):
    """One transformer block

    This module consists of self-attention and feed-forward layers.

    Args:
        x: (..., seq_len, dim)
        mask: (..., seq_len), dtype=bool
    Returns:
        (..., seq_len, dim)
    """

    attention: Attention
    feed_forward: FeedForward

    norm1: LayerNorm
    norm2: LayerNorm

    def __init__(self, config: Config):
        super().__init__()

        self.attention = Attention(
            cnt_head=config.cnt_head,
            d_in=config.dim,
            d_out=config.dim,
            d_key=config.d_key,
            d_value=config.d_value,
            is_scaled=False,
            is_causal=True,
        )
        self.feed_forward = FeedForward(config.dim, d_hidden=config.d_feed_forward)

        self.norm1 = LayerNorm(config.dim)
        self.norm2 = LayerNorm(config.dim)

    @override
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        y = x
        x = self.norm1(x)
        x = self.attention(x, mask)
        x += y

        y = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x += y

        return x


class Transformer(nn.Module):
    blocks: nn.ModuleList

    def __init__(self, config: Config):
        super().__init__()

        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.cnt_layer)
        )

    @override
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        for block in self.blocks:
            x = block(x, mask)

        return x


class LLM(nn.Module):
    """The full LLM

    Args:
        tokens: (..., seq_len), dtype=int
        mask: (..., seq_len), dtype=bool
    Returns:
        (..., vocab_size)
    """

    token_embedding: nn.Embedding
    positional_encoding: PositionalEncoding
    transformer: Transformer
    out: nn.Linear

    def __init__(self, config: Config):
        super().__init__()

        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.positional_encoding = PositionalEncoding(config.max_seq_len, config.dim)
        self.transformer = Transformer(config)
        self.out = nn.Linear(config.dim, config.vocab_size)

    @override
    def forward(self, tokens: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = self.token_embedding(tokens)
        x = self.positional_encoding(x)

        x = self.transformer(x, mask)

        x = self.out(x)

        return x
