from typing import Optional, override

import torch
import torch.nn as nn
from pydantic import BaseModel, NonNegativeFloat, PositiveInt
from torch import Tensor

from spargel_llm.layers.torch import (
    Attention,
    FeedForward,
    LayerNorm,
    PositionalEncoding,
)


class Config(BaseModel):
    vocab_size: PositiveInt
    max_seq_len: PositiveInt
    cnt_layer: PositiveInt
    cnt_head: PositiveInt
    dim: PositiveInt
    d_key: PositiveInt
    d_value: PositiveInt
    d_feed_forward: PositiveInt
    dropout_p: NonNegativeFloat = 0.1


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

    dropout1: nn.Dropout
    dropout2: nn.Dropout

    def __init__(self, config: Config):
        super().__init__()

        self.attention = Attention(
            cnt_head=config.cnt_head,
            d_in=config.dim,
            d_out=config.dim,
            d_key=config.d_key,
            d_value=config.d_value,
        )
        self.feed_forward = FeedForward(config.dim, d_hidden=config.d_feed_forward)

        self.norm1 = LayerNorm(config.dim)
        self.norm2 = LayerNorm(config.dim)

        self.dropout1 = nn.Dropout(config.dropout_p)
        self.dropout2 = nn.Dropout(config.dropout_p)

    @override
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        y = x
        x = self.norm1(x)
        x = self.attention(x, mask)
        x = self.dropout1(x)
        x += y

        y = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout2(x)
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

    config: Config

    token_embedding: nn.Embedding
    positional_encoding: PositionalEncoding
    transformer: Transformer
    out: nn.Linear

    def __init__(self, config: Config):
        super().__init__()

        self.config = config

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

    @override
    def extra_repr(self) -> str:
        return str(self.config)

    def embed(self, token: Tensor) -> Tensor:
        return self.token_embedding(token)

    def loss(
        self, input: Tensor, mask: Tensor, target: Tensor, pad_index: int
    ) -> Tensor:
        """
        Args:
            input: (..., seq_len), dtype=int
            mask: (..., seq_len), dtype=bool
            target: (..., seq_len), dtype=int
        """

        torch._assert(input.shape == target.shape, "shape")

        logits: Tensor = self(input, mask)  # (..., seq_len, vocab_size)
        loss = nn.functional.cross_entropy(
            logits.flatten(0, -2),
            target.flatten(0, -1).type(torch.long),
            ignore_index=pad_index,
        )

        return loss
