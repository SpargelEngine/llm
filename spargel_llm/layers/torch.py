# Reference: Attention Is All You Need (arXiv:1706.03762v7)
# Reference: Root Mean Square Layer Normalization (arXiv:1910.07467)

import math
from typing import override

import torch
import torch.nn as nn
from torch import Tensor


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: float,
    *,
    mask: Tensor | None = None,
) -> Tensor:
    """
    Args:
        q (..., num_q, dim_k)
        k (..., num_k, dim_k)
        v (..., num_k, dim_v)
        scale (float | None): the scale used in softmax

        mask (..., num_q, num_k)

    Returns:
        (..., num_q, dim_v)
    """

    assert q.size(-1) == k.size(-1), "dimensions of queries & keys do not match"
    assert k.size(-2) == v.size(-2), "numbers of keys & values do not match"

    num_q, num_k = q.size(-2), k.size(-2)

    if mask is not None:
        assert mask.dtype == torch.bool, "mask.dtype != bool"
        assert mask.size(-2) in {1, num_q}, "bad mask size (-2)"
        assert mask.size(-1) == num_k, "bad mask size (-1)"

    # (..., num_q, num_k)
    scores = q @ k.transpose(-2, -1)

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    scores = torch.softmax(scores * scale, dim=-1)

    result = scores @ v

    return result


class Attention(nn.Module):
    """
    Multi-Head Attention

    Attributes:
        num_head: number of heads
        dim_in: input dimension
        dim_out: output dimension
        dim_k: dimension of query/key (per head)
        dim_v: dimension of value (per head)
    """

    def __init__(
        self,
        num_head: int,
        dim_in: int,
        dim_out: int,
        dim_k: int,
        dim_v: int,
    ):
        super().__init__()
        self.num_head = num_head
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_k = dim_k
        self.dim_v = dim_v

        self.scale = 1 / math.sqrt(self.dim_k)

        self.W_q = nn.Parameter(torch.empty(num_head, dim_in, dim_k))
        self.W_k = nn.Parameter(torch.empty(num_head, dim_in, dim_k))
        self.W_v = nn.Parameter(torch.empty(num_head, dim_in, dim_v))
        self.W_o = nn.Parameter(torch.empty(num_head, dim_v, dim_out))

        def _xavier_init(w: Tensor, d1: int, d2: int):
            a = math.sqrt(6 / (d1 + d2))
            nn.init.uniform_(w, -a, +a)

        _xavier_init(self.W_q, dim_in, dim_k)
        _xavier_init(self.W_k, dim_in, dim_k)
        _xavier_init(self.W_v, dim_in, dim_v)
        _xavier_init(self.W_o, num_head * dim_v, dim_out)

    @override
    def forward(
        self,
        x: Tensor,
        causal: bool = True,
        mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x (..., seq_len, dim_in)
            causal (bool): apply causal mask
            mask (..., seq_len) dtype=bool: padding mask (`True` means masked)

        Returns:
            (..., seq_len, dim_out)
        """

        assert x.size(-1) == self.dim_in
        seq_len = x.size(-2)
        if mask is not None:
            assert mask.size(-1) == seq_len, "bad mask shape"

        # (..., num_head, seq_len, dim_k)
        q = torch.einsum("ijk, ...lj -> ...ilk", self.W_q, x)
        # (..., num_head, seq_len, dim_k)
        k = torch.einsum("ijk, ...lj -> ...ilk", self.W_k, x)
        # (..., num_head, seq_len, dim_v)
        v = torch.einsum("ijk, ...lj -> ...ilk", self.W_v, x)

        if causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
                diagonal=1,
            )
            if mask is not None:
                # (..., seq_len, seq_len)
                full_mask = mask.unsqueeze(-2) | causal_mask
            else:
                full_mask = causal_mask
        else:
            if mask is not None:
                full_mask = mask.unsqueeze(-2)
            else:
                full_mask = None

        # broadcast to all heads
        if full_mask is not None:
            full_mask = full_mask.unsqueeze(-3)

        # (..., num_head, seq_len, dim_v)
        results = attention(q, k, v, self.scale, mask=full_mask)

        return torch.einsum("ijk, ...ilj -> ...lk", self.W_o, results)

    @override
    def extra_repr(self) -> str:
        return f"num_head={self.num_head}, dim_in={self.dim_in}, dim_out={self.dim_out}, dim_k={self.dim_k}, dim_v={self.dim_v}"


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


class PositionalEncoding(nn.Module):
    """
    Positional Encoding
    """

    def __init__(self, max_seq_len: int, dim: int):
        super().__init__()
        assert dim % 2 == 0, f"dim must be even, got {dim}"
        self.max_seq_len = max_seq_len
        self.dim = dim

        self.pe = nn.Buffer(torch.empty(max_seq_len, dim))

        positions = torch.arange(0, max_seq_len).reshape(max_seq_len, 1)
        frequencies = torch.arange(1, dim // 2 + 1) * torch.pi / max_seq_len

        self.pe[:, 0::2] = torch.sin(positions * frequencies)
        self.pe[:, 1::2] = torch.cos(positions * frequencies)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (..., seq_len, dim)

        Returns:
            (..., seq_len, dim)
        """

        assert x.size(-1) == self.dim
        assert x.size(-2) <= self.max_seq_len, "seq_len too large"

        seq_len = x.size(-2)
        x += self.pe[:seq_len, :]

        return x

    @override
    def extra_repr(self) -> str:
        return f"max_seq_len={self.max_seq_len}, dim={self.dim}"


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    """

    def __init__(self, dim: int, epsilon: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.epsilon = epsilon

        self.weight = nn.Parameter(torch.ones(dim))

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (..., dim)

        Returns:
            (..., dim)
        """

        var = x.square().mean(-1, keepdim=True)
        y = x * torch.rsqrt(var + self.epsilon)
        y = self.weight * y
        return y

    @override
    def extra_repr(self) -> str:
        return f"dim={self.dim}, epsilon={self.epsilon}"
