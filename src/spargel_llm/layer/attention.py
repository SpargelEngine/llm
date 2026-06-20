# Reference: Attention Is All You Need (arXiv:1706.03762v7)

import math
from typing import override

import torch
import torch.nn as nn
from torch import Tensor


def rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """
    Compute: x R_k
    Args:
        x (..., seq_len, dim)       actually (batch, num_head, seq_len, dim_head)
        cos (seq_len, dim // 2)
        sin (seq_len, dim // 2)
    Returns:
        (..., seq_len, dim)
    """
    dim = x.size(-1)
    assert dim % 2 == 0
    half_dim = dim // 2
    x1, x2 = x[..., :half_dim], x[..., half_dim:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


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
        *,
        use_rope=False,
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

        self.use_rope = use_rope
        if self.use_rope:
            assert self.dim_k % 2 == 0, f"head dim must be even, got {self.dim_k}"
            self.thetas = nn.Buffer(
                1.0 / (10000.0 ** (torch.arange(0, self.dim_k, 2) / self.dim_k)),
                persistent=False,
            )
            self.cos_values = nn.Buffer(None, persistent=False)
            self.sin_values = nn.Buffer(None, persistent=False)
            self.seq_len = None

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

        seq_len = x.size(-2)

        # (..., num_head, seq_len, dim_k)
        q = torch.einsum("ijk, ...lj -> ...ilk", self.W_q, x)
        # (..., num_head, seq_len, dim_k)
        k = torch.einsum("ijk, ...lj -> ...ilk", self.W_k, x)
        # (..., num_head, seq_len, dim_v)
        v = torch.einsum("ijk, ...lj -> ...ilk", self.W_v, x)

        if self.use_rope:
            if self.seq_len is None or self.seq_len != seq_len:
                self.seq_len = seq_len
                positions = torch.arange(0, seq_len, device=x.device, dtype=q.dtype)
                thetas = self.thetas.to(dtype=q.dtype)
                freqs = torch.outer(positions, thetas)
                self.cos_values = freqs.cos()
                self.sin_values = freqs.sin()

            q = rope(q, self.cos_values, self.sin_values)
            k = rope(k, self.cos_values, self.sin_values)

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
