import math
from typing import Optional, override

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncodingLearned(nn.Module):
    """Positional encoding that needs to be learned

    Args:
        x: (..., seq_len, dim)
    Returns:
        (..., seq_len, dim)
    """

    _ctx_len: int
    _dim: int

    _pe: nn.Parameter

    def __init__(self, ctx_len: int, dim: int):
        super().__init__()

        self._ctx_len = ctx_len
        self._dim = dim
        self._pe = nn.Parameter(torch.rand(ctx_len, dim))

    @override
    def forward(self, x: Tensor) -> Tensor:
        assert x.size(-1) == self._dim
        assert x.size(-2) <= self._ctx_len  # seq_len

        seq_len = x.size(-2)
        x += self._pe[:seq_len, :]

        return x


class PositionalEncoding(nn.Module):
    """Positional encoding which is specified maually

    Args:
        x: (..., seq_len, dim)
    Returns:
        (..., seq_len, dim)
    """

    _ctx_len: int
    _dim: int

    _pe: nn.Buffer

    def __init__(self, ctx_len: int, dim: int):
        super().__init__()

        self._ctx_len = ctx_len
        self._dim = dim
        self._pe = nn.Buffer(torch.empty(ctx_len, dim))

        positions = torch.arange(0, ctx_len, dtype=torch.float).reshape(ctx_len, 1)
        # frequencies = ctx_len ** (-torch.arange(0, dim // 2) * 2 / dim)
        frequencies = (
            torch.arange(1, dim // 2 + 1, dtype=torch.float) * torch.pi / ctx_len
        )

        self._pe[:, 0::2] = torch.sin(positions * frequencies)
        self._pe[:, 1::2] = torch.cos(positions * frequencies)

    @override
    def forward(self, x: Tensor) -> Tensor:
        assert x.size(-1) == self._dim
        assert x.size(-2) <= self._ctx_len  # seq_len

        seq_len = x.size(-2)
        x += self._pe[:seq_len, :]

        return x


def scaled_dot_product(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    *,
    mask: Optional[Tensor] = None,
    scaled: bool = False,
) -> Tensor:
    """
    Args:
        Q: (..., cnt_q, d_key)
        K: (..., cnt_k, d_key)
        V: (..., cnt_k, d_value)

        mask: (..., cnt_q, cnt_k)
    Returns:
        (..., cnt_q, d_value)
    """

    assert Q.size(-1) == K.size(-1)
    assert K.size(-2) == V.size(-2)
    if mask is not None:
        assert mask.dtype == torch.bool
        assert mask.size(-2) == Q.size(-2) and mask.size(-1) == K.size(-2)

    d_key = K.size(-1)

    # (..., cnt_q, cnt_k)
    scores = torch.einsum("...ik, ...jk -> ...ij", Q, K)

    if mask is not None:
        scores.masked_fill_(mask, -torch.inf)

    if scaled:
        scores /= math.sqrt(d_key)

    weights = torch.softmax(scores, dim=-1)

    # print(weights)

    result = weights @ V

    return result


class Attention(nn.Module):
    """Scaled dot-product attention

    Args:
        x: (..., seq_len, d_in)
        mask (Optional): (..., seq_len, seq_len)
    Returns:
        (..., seq_len, d_out)
    """

    _scaled: bool

    _W_q: nn.Linear
    _W_k: nn.Linear
    _W_v: nn.Linear

    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_key: int,
        *,
        scaled: bool = False,
    ):
        super().__init__()

        self._scaled = scaled

        self._W_q = nn.Linear(d_in, d_key, bias=False)
        self._W_k = nn.Linear(d_in, d_key, bias=False)
        self._W_v = nn.Linear(d_in, d_out, bias=False)

    @override
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        seq_len = x.size(-2)
        if mask is not None:
            assert mask.size(-1) == mask.size(-2) == seq_len

        Q = self._W_q(x)
        K = self._W_k(x)
        V = self._W_v(x)

        if mask is not None:
            return scaled_dot_product(Q, K, V, mask=mask, scaled=self._scaled)
        else:
            return scaled_dot_product(Q, K, V, scaled=self._scaled)


class FeedForward(nn.Module):
    """Fully connected feed-forward layers

    Args:
        x: (..., dim)
    Returns:
        (..., dim)
    """

    _layers: nn.Sequential

    def __init__(self, dim: int, d_hidden: int):
        super().__init__()

        self._layers = nn.Sequential(
            nn.Linear(dim, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, dim),
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self._layers(x)


class TransformerBlock(nn.Module):
    """One transformer block

    This module consists of self-attention and feed-forward layers.

    Args:
        x: (..., seq_len, dim)
        mask: (..., seq_len, seq_len)
    Returns:
        (..., seq_len, dim)
    """

    _attention: Attention
    _feed_forward: FeedForward

    def __init__(self, dim: int, d_key: int):
        super().__init__()

        self._attention = Attention(d_in=dim, d_out=dim, d_key=d_key, scaled=False)
        self._feed_forward = FeedForward(dim, d_hidden=dim)

    @override
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        y = x
        x = self._attention(x, mask)
        x += y

        y = x
        x = self._feed_forward(x)
        x += y

        return x


class LLM(nn.Module):
    """The full LLM

    Args:
        tokens: (..., seq_len), dtype=int
        mask: (..., seq_len, seq_len)
    Returns:
        (..., vocab_size)
    """

    _token_embedding: nn.Embedding
    _positional_encoding: PositionalEncoding
    _transformer: TransformerBlock
    _head: nn.Linear

    def __init__(self, vocab_size: int, ctx_len: int, dim: int, d_key: int):
        super().__init__()

        self._token_embedding = nn.Embedding(vocab_size, dim)
        self._positional_encoding = PositionalEncoding(ctx_len, dim)
        self._transformer = TransformerBlock(dim, d_key)
        self._head = nn.Linear(dim, vocab_size)

    @override
    def forward(self, tokens: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = self._token_embedding(tokens)
        x = self._positional_encoding(x)

        x = self._transformer(x, mask)

        x = self._head(x)

        return x


def generate_causal_mask(d1: int, d2: int) -> Tensor:
    return torch.tril(torch.ones(d1, d2)) == 0


def calculate_loss(
    model: LLM,
    input: Tensor,
    mask: Tensor,
    target: Tensor,
    pad_index: int,
) -> Tensor:
    """
    Args:
        input: (..., seq_len), dtype=int
        target: (..., seq_len), dtype=int
    """

    assert input.shape == target.shape

    logits: Tensor = model(input, mask)  # (..., seq_len, vocab_size)
    loss = nn.functional.cross_entropy(
        logits.flatten(0, -2), target.flatten(0, -1), ignore_index=pad_index
    )

    return loss
