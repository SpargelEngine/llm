from typing import override

import torch.nn as nn
from pydantic import BaseModel, PositiveInt
from torch import Tensor

from spargel_llm.layer.attention import Attention
from spargel_llm.layer.feed_forward import FeedForward
from spargel_llm.layer.positional_encoding import PositionalEncoding
from spargel_llm.layer.rms_norm import RMSNorm


class Config(BaseModel):
    vocab_size: PositiveInt
    max_seq_len: PositiveInt
    num_layer: PositiveInt
    num_head: PositiveInt
    dim: PositiveInt
    dim_key: PositiveInt
    dim_value: PositiveInt
    dim_ff_hidden: PositiveInt
    use_rope: bool
    ff_activation: FeedForward.Activation


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
            use_rope=config.use_rope,
        )

        self.feed_forward = FeedForward(
            config.dim, config.dim_ff_hidden, config.ff_activation
        )

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
        if not self.config.use_rope:
            self.positional_encoding = PositionalEncoding(
                config.max_seq_len, config.dim
            )
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.num_layer)
        )
        self.final_norm = RMSNorm(config.dim)
        self.out = nn.Linear(config.dim, config.vocab_size)

    @override
    def forward(self, tokens: Tensor, mask: Tensor | None = None) -> Tensor:
        x = self.embedding(tokens)
        if not self.config.use_rope:
            x = self.positional_encoding(x)
        for block in self.blocks:
            x = block(x, mask)
        x = self.final_norm(x)
        x = self.out(x)
        return x

    @override
    def extra_repr(self) -> str:
        return str(self.config)
