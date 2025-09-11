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

    _max_seq_len: int
    _dim: int

    _pe: nn.Parameter

    def __init__(self, max_seq_len: int, dim: int):
        super().__init__()

        self._max_seq_len = max_seq_len
        self._dim = dim
        self._pe = nn.Parameter(torch.rand(max_seq_len, dim))

    @override
    def forward(self, x: Tensor) -> Tensor:
        torch._assert(x.size(-1) == self._dim, "bad dim")
        torch._assert(x.size(-2) <= self._max_seq_len, "seq_len too large")

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

    _max_seq_len: int
    _dim: int

    _pe: nn.Buffer

    def __init__(self, max_seq_len: int, dim: int):
        super().__init__()

        self._max_seq_len = max_seq_len
        self._dim = dim
        self._pe = nn.Buffer(torch.empty(max_seq_len, dim))

        positions = torch.arange(0, max_seq_len, dtype=torch.float).reshape(
            max_seq_len, 1
        )
        # frequencies = max_seq_len ** (-torch.arange(0, dim // 2) * 2 / dim)
        frequencies = (
            torch.arange(1, dim // 2 + 1, dtype=torch.float) * torch.pi / max_seq_len
        )

        self._pe[:, 0::2] = torch.sin(positions * frequencies)
        self._pe[:, 1::2] = torch.cos(positions * frequencies)

    @override
    def forward(self, x: Tensor) -> Tensor:
        torch._assert(x.size(-1) == self._dim, "bad dim")
        torch._assert(x.size(-2) <= self._max_seq_len, "seq_len too large")  # seq_len

        seq_len = x.size(-2)
        x += self._pe[:seq_len, :]

        return x


def scaled_dot_product(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    *,
    mask: Optional[Tensor] = None,
    is_scaled: bool = False,
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

    torch._assert(Q.size(-1) == K.size(-1), "d_key not matched")
    torch._assert(K.size(-2) == V.size(-2), "cnt_k not matched")

    cnt_q, cnt_k = Q.size(-2), K.size(-2)
    d_key = K.size(-1)

    if mask is not None:
        torch._assert(mask.dtype == torch.bool, "mask dtype != bool")
        torch._assert(mask.size(-2) == cnt_q, "bad mask size (-2)")
        torch._assert(mask.size(-1) == cnt_k, "bad mask size (-1)")

    # (..., cnt_q, cnt_k)
    scores = torch.einsum("...ik, ...jk -> ...ij", Q, K)

    if mask is not None:
        scores = scores.masked_fill(mask, -torch.inf)

    if is_scaled:
        weights = torch.softmax(scores / math.sqrt(d_key), dim=-1)
    else:
        weights = torch.softmax(scores, dim=-1)

    # get rid of NaN
    if mask is not None:
        weights = weights.masked_fill(mask, 0.0)

    # print(weights)

    result = weights @ V

    return result


class Attention(nn.Module):
    """Scaled dot-product attention

    Args:
        x: (..., seq_len, d_in)
        mask (Optional): (..., seq_len), dtype=bool
    Returns:
        (..., seq_len, d_out)
    """

    _is_scaled: bool
    _is_causal: bool

    _W_q: nn.Linear
    _W_k: nn.Linear
    _W_v: nn.Linear

    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_key: int,
        *,
        is_scaled: bool = False,
        is_causal: bool = False,
    ):
        super().__init__()

        self._is_scaled = is_scaled
        self._is_causal = is_causal

        self._W_q = nn.Linear(d_in, d_key, bias=False)
        self._W_k = nn.Linear(d_in, d_key, bias=False)
        self._W_v = nn.Linear(d_in, d_out, bias=False)

    @override
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        seq_len = x.size(-2)
        if mask is not None:
            torch._assert(mask.size(-1) == seq_len, "bad mask size")

        Q = self._W_q(x)
        K = self._W_k(x)
        V = self._W_v(x)

        if self._is_causal:
            # (seq_len, seq_len)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1
            )
            if mask is not None:
                # (..., seq_len, seq_len)
                padding_mask = mask.unsqueeze(-2) | mask.unsqueeze(-1)
                return scaled_dot_product(
                    Q,
                    K,
                    V,
                    mask=(padding_mask | causal_mask),
                    is_scaled=self._is_scaled,
                )
            else:
                return scaled_dot_product(
                    Q, K, V, mask=causal_mask, is_scaled=self._is_scaled
                )
        else:
            if mask is not None:
                padding_mask = mask.unsqueeze(-2) | mask.unsqueeze(-1)
                return scaled_dot_product(Q, K, V, mask=mask, is_scaled=self._is_scaled)
            else:
                return scaled_dot_product(Q, K, V, is_scaled=self._is_scaled)


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


class LayerNorm(nn.Module):
    """
    Args:
        x: (..., dim)
    Returns:
        (..., dim)
    """

    eps = 1e-5

    def __init__(self):
        super().__init__()

    @override
    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps)


class TransformerBlock(nn.Module):
    """One transformer block

    This module consists of self-attention and feed-forward layers.

    Args:
        x: (..., seq_len, dim)
        mask: (..., seq_len), dtype=bool
    Returns:
        (..., seq_len, dim)
    """

    _attention: Attention
    _feed_forward: FeedForward

    # _norm1: LayerNorm

    def __init__(self, dim: int, d_key: int, d_hidden: int):
        super().__init__()

        self._attention = Attention(
            d_in=dim, d_out=dim, d_key=d_key, is_scaled=False, is_causal=True
        )
        self._feed_forward = FeedForward(dim, d_hidden=d_hidden)

        # self._norm1 = LayerNorm()

    @override
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        y = x
        x = self._attention(x, mask)
        x += y
        # x = self._norm1(x)

        y = x
        x = self._feed_forward(x)
        x += y

        return x


class LLM(nn.Module):
    """The full LLM

    Args:
        tokens: (..., seq_len), dtype=int
        mask: (..., seq_len), dtype=bool
    Returns:
        (..., vocab_size)
    """

    _token_embedding: nn.Embedding
    _positional_encoding: PositionalEncoding
    _transformer: TransformerBlock
    # _transformer2: TransformerBlock
    _head: nn.Linear

    def __init__(
        self, vocab_size: int, max_seq_len: int, dim: int, d_key: int, d_hidden: int
    ):
        super().__init__()

        self._token_embedding = nn.Embedding(vocab_size, dim)
        self._positional_encoding = PositionalEncoding(max_seq_len, dim)
        self._transformer = TransformerBlock(dim, d_key, d_hidden)
        # self._transformer2 = TransformerBlock(dim, d_key, d_hidden)
        self._head = nn.Linear(dim, vocab_size)

    @override
    def forward(self, tokens: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = self._token_embedding(tokens)
        x = self._positional_encoding(x)

        x = self._transformer(x, mask)
        # x = self._transformer2(x, mask)

        x = self._head(x)

        return x


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
        mask: (..., seq_len), dtype=bool
        target: (..., seq_len), dtype=int
    """

    torch._assert(input.shape == target.shape, "shape not matched")

    logits: Tensor = model(input, mask)  # (..., seq_len, vocab_size)
    loss = nn.functional.cross_entropy(
        logits.flatten(0, -2), target.flatten(0, -1), ignore_index=pad_index
    )

    return loss
