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

    max_seq_len: int
    dim: int

    pe: nn.Parameter

    def __init__(self, max_seq_len: int, dim: int):
        """
        Args:
            max_seq_len (int): expected max sequence length to distinguish
            dim (int): embed dimension
        """

        super().__init__()

        self.max_seq_len = max_seq_len
        self.dim = dim
        self.pe = nn.Parameter(torch.empty(max_seq_len, dim))
        nn.init.xavier_uniform_(self.pe)

    @override
    def forward(self, x: Tensor) -> Tensor:
        torch._assert(x.size(-1) == self.dim, "bad dim")
        torch._assert(x.size(-2) <= self.max_seq_len, "seq_len too large")

        seq_len = x.size(-2)
        x += self.pe[:seq_len, :]

        return x


class PositionalEncoding(nn.Module):
    """Positional encoding which is specified maually

    Args:
        x: (..., seq_len, dim)
    Returns:
        (..., seq_len, dim)
    """

    max_seq_len: int
    dim: int

    pe: nn.Buffer

    def __init__(self, max_seq_len: int, dim: int):
        """
        Args:
            max_seq_len (int): expected max sequence length to distinguish
            dim (int): embed dimension
        """

        super().__init__()

        self.max_seq_len = max_seq_len
        self.dim = dim
        self.pe = nn.Buffer(torch.empty(max_seq_len, dim))

        positions = torch.arange(0, max_seq_len, dtype=torch.float).reshape(
            max_seq_len, 1
        )
        # frequencies = max_seq_len ** (-torch.arange(0, dim // 2) * 2 / dim)
        frequencies = (
            torch.arange(1, dim // 2 + 1, dtype=torch.float) * torch.pi / max_seq_len
        )

        self.pe[:, 0::2] = torch.sin(positions * frequencies)
        self.pe[:, 1::2] = torch.cos(positions * frequencies)

    @override
    def forward(self, x: Tensor) -> Tensor:
        torch._assert(x.size(-1) == self.dim, "bad dim")
        torch._assert(x.size(-2) <= self.max_seq_len, "seq_len too large")  # seq_len

        seq_len = x.size(-2)
        x += self.pe[:seq_len, :]

        return x

    @override
    def extra_repr(self) -> str:
        return f"max_seq_len={self.max_seq_len}, dim={self.dim}"


class LayerNorm(nn.Module):
    """
    Args:
        x: (..., dim)
    Returns:
        (..., dim)
    """

    eps = 1e-5

    scale_and_shift: bool
    scale: Optional[nn.Parameter]
    shift: Optional[nn.Parameter]

    def __init__(self, dim: int, scale_and_shift: bool = True):
        """
        Args:
            dim: embed dimension
        """

        super().__init__()

        self.scale_and_shift = scale_and_shift

        if scale_and_shift:
            self.scale = nn.Parameter(torch.ones(dim))
            self.shift = nn.Parameter(torch.zeros(dim))
        else:
            self.scale = self.shift = None

    @override
    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        normalized = (x - mean) / torch.sqrt(var + self.eps)

        if self.scale_and_shift and self.scale is not None and self.shift is not None:
            return self.scale * normalized + self.shift
        else:
            return normalized

    @override
    def extra_repr(self) -> str:
        return f"scale_and_shift={self.scale_and_shift}"


def scaled_dot_product(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    *,
    mask: Optional[Tensor] = None,
    is_scaled: bool = True,
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
        scores = scores.masked_fill(mask, scores.new_tensor(float("-inf")))

    if is_scaled:
        weights = torch.softmax(scores / math.sqrt(d_key), dim=-1)
    else:
        weights = torch.softmax(scores, dim=-1)

    # LEGACY
    # Get rid of possible NaN,
    # since NaN * 0.0 = NaN and therefore cannot be eliminated.
    # if mask is not None:
    #     weights = weights.masked_fill(mask, 0.0)

    result = weights @ V

    return result


class Attention(nn.Module):
    """Multihead scaled dot-product attention

    Args:
        x: (..., seq_len, d_in)
        mask (Optional): (..., seq_len), dtype=bool
    Returns:
        (..., seq_len, d_out)
    """

    cnt = 0

    cnt_head: int
    d_in: int
    d_out: int
    d_key: int
    d_value: int
    is_scaled: bool
    is_causal: bool

    W_q: nn.Parameter  # (cnt_h, d_in, d_key)
    W_k: nn.Parameter  # (cnt_h, d_in, d_key)
    W_v: nn.Parameter  # (cnt_h, d_in, d_value)
    W_o: nn.Parameter  # (cnt_h, d_value, d_out)

    def __init__(
        self,
        cnt_head: int,
        d_in: int,
        d_out: int,
        d_key: int,
        d_value: int,
        *,
        is_scaled: bool = True,
        is_causal: bool = True,
    ):
        """
        Args:
            cnt_head: number of heads
            d_in: input dimension
            d_out: output dimension
            d_key: key dimension
            d_value: value dimension
            is_scaled (bool): whether to scale the dot-product before softmax
            is_causal (bool): whether to apply causal mask
        """
        super().__init__()

        self.cnt_head = cnt_head
        self.d_in = d_in
        self.d_out = d_out
        self.d_key = d_key
        self.d_value = d_value

        self.is_scaled = is_scaled
        self.is_causal = is_causal

        self.W_q = nn.Parameter(torch.empty(cnt_head, d_in, d_key))
        self.W_k = nn.Parameter(torch.empty(cnt_head, d_in, d_key))
        self.W_v = nn.Parameter(torch.empty(cnt_head, d_in, d_value))
        self.W_o = nn.Parameter(torch.empty(cnt_head, d_value, d_out))

        def _xavier_init(W: Tensor, d1: int, d2: int):
            a = math.sqrt(6 / (d1 + d2))
            nn.init._no_grad_uniform_(W, -a, +a)

        _xavier_init(self.W_q, d_in, cnt_head * d_key)
        _xavier_init(self.W_k, d_in, cnt_head * d_key)
        _xavier_init(self.W_v, d_in, cnt_head * d_value)
        _xavier_init(self.W_o, cnt_head * d_value, d_out)

    @override
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        seq_len = x.size(-2)
        if mask is not None:
            torch._assert(mask.size(-1) == seq_len, "bad mask size")

        # x: (..., seq_len, d_in)

        # W_q: (cnt_h, d_in, d_key)
        Q = torch.einsum(
            "ijk, ...lj -> ...ilk", self.W_q, x
        )  # (..., cnt_h, seq_len, d_key)

        # W_k: (cnt_h, d_in, d_key)
        K = torch.einsum(
            "ijk, ...lj -> ...ilk", self.W_k, x
        )  # (..., cnt_h, seq_len, d_key)

        # W_v: (cnt_h, d_in, d_value)
        V = torch.einsum(
            "ijk, ...lj -> ...ilk", self.W_v, x
        )  # (..., cnt_h, seq_len, d_value)

        if self.is_causal:
            # (seq_len, seq_len)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
                diagonal=1,
            )
            if mask is not None:
                # (..., seq_len, seq_len)
                mask = mask.unsqueeze(-2) | causal_mask
            else:
                mask = causal_mask
        else:
            if mask is not None:
                mask = mask.unsqueeze(-2)
            else:
                mask = None

        # mask: (..., seq_len, seq_len) | (seq_len, seq_len) | (..., 1, seq_len) | None
        if mask is not None:
            # broadcase to all heads
            mask = mask.unsqueeze(-3)
        # mask: (..., 1, seq_len, seq_len) | (1, seq_len, seq_len) | (..., 1, 1, seq_len) | None

        values = scaled_dot_product(Q, K, V, mask=mask, is_scaled=self.is_scaled)

        # values: (..., cnt_h, seq_len, d_value)
        # W_o: (cnt_h, d_value, d_out)
        return torch.einsum("ijk, ...ilj -> ...lk", self.W_o, values)

    @override
    def extra_repr(self) -> str:
        return f"cnt_head={self.cnt_head}, d_in={self.d_in}, d_out={self.d_out}, d_key={self.d_key}, d_value={self.d_value}, is_scaled={self.is_scaled}, is_causal={self.is_causal}"


class FeedForward(nn.Module):
    """Fully connected feed-forward layers

    Args:
        x: (..., dim)
    Returns:
        (..., dim)
    """

    dim: int
    d_hidden: int

    layers: nn.Sequential

    def __init__(self, dim: int, d_hidden: int):
        super().__init__()

        self.dim = dim
        self.d_hidden = d_hidden

        self.layers = nn.Sequential(
            nn.Linear(dim, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, dim),
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

    @override
    def extra_repr(self) -> str:
        return f"dim={self.dim}, d_hidden={self.d_hidden}"
