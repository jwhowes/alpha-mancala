import torch
import torch.nn.functional as F

from torch import nn, FloatTensor, Tensor
from typing import Optional
from math import sqrt
from einops import rearrange


class SinusoidalEmbedding(nn.Module):
    def __init__(self, d_model: int, base: float = 1e4):
        super(SinusoidalEmbedding, self).__init__()
        assert d_model % 2 == 0
        self.register_buffer(
            "theta",
            1.0 / (base ** (2 * torch.arange(d_model // 2) / d_model)),
            persistent=False
        )

    def forward(self, x: Tensor) -> FloatTensor:
        x = x.unsqueeze(-1).float() * self.theta

        return torch.stack((
            x.cos(),
            x.sin()
        ), dim=-1).flatten(-2)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_hidden: Optional[int] = None):
        super(SwiGLU, self).__init__()
        if d_hidden is None:
            d_hidden = 4 * d_model

        self.gate = nn.Linear(d_model, d_hidden, bias=False)
        self.hidden = nn.Linear(d_model, d_hidden, bias=False)
        self.out = nn.Linear(d_hidden, d_model)

    def forward(self, x: FloatTensor) -> FloatTensor:
        return self.out(
            F.silu(self.gate(x)) * self.hidden(x)
        )


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(d_model).normal_(mean=1.0, std=sqrt(1 / d_model)))

    def forward(self, x: FloatTensor) -> FloatTensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super(Attention, self).__init__()
        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.scale = sqrt(d_model // n_heads)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x: FloatTensor) -> FloatTensor:
        q = rearrange(self.W_q(x), "b l (n d) -> b n l d", n=self.n_heads)
        k = rearrange(self.W_k(x), "b l (n d) -> b n l d", n=self.n_heads)
        v = rearrange(self.W_v(x), "b l (n d) -> b n l d", n=self.n_heads)

        return self.W_o(
            rearrange(
                self.dropout(F.softmax((q @ k.transpose(-2, -1)) / self.scale, dim=-1)) @ v, "b n l d -> b l (n d)"
            )
        )


class Block(nn.Module):
    def __init__(
            self, d_model: int, n_heads: int,
            attn_dropout: float = 0.0, ffn_dropout: float = 0.0,
            d_hidden: Optional[int] = None, norm_eps: float = 1e-6
    ):
        super(Block, self).__init__()
        self.attn = Attention(d_model, n_heads, dropout=attn_dropout)
        self.attn_norm = RMSNorm(d_model, eps=norm_eps)

        self.ffn_dropout = nn.Dropout(ffn_dropout)
        self.ffn = SwiGLU(d_model, d_hidden)
        self.ffn_norm = RMSNorm(d_model, eps=norm_eps)

    def forward(self, x: FloatTensor) -> FloatTensor:
        x = x + self.attn(self.attn_norm(x))

        return x + self.ffn_dropout(self.ffn(self.ffn_norm(x)))
