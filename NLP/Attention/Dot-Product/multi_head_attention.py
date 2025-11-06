import math
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class MultiHeadAttentionCustom(nn.Module):
    """
    Custom implementation of Multi-Head Scaled Dot-Product Attention.

    Attributes:
        d_model (int): Dimension of the input embeddings.
        num_heads (int): Number of attention heads.
        d_k (int): Dimension of each attention head (d_model / num_heads).
        w_q (nn.Linear): Linear layer to project inputs to queries.
        w_k (nn.Linear): Linear layer to project inputs to keys.
        w_v (nn.Linear): Linear layer to project inputs to values.
        w_o (nn.Linear): Linear layer to combine output of attention heads.
        dropout (nn.Dropout): Dropout layer applied to attention weights.
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Constructor

        :param d_model: Dimension of input embeddings.
        :param num_heads: Number of attention heads.
        :param dropout: Dropout probability applied to attention weights.
        """

        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (num_heads, d_k) and transpose for attention calculation.

        :param x: Input tensor of shape (batch_size, seq_len, d_model)
        :return: Tensor of shape (batch_size, num_heads, seq_len, d_k)
        """

        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2).contiguous()
        batch_size, seq_len, _, _ = x.size()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of Multi-Head Attention.

        :param x_q: Query tensor of shape (batch_size, seq_len_q, d_model)
        :param x_kv: Key/Value tensor of shape (batch_size, seq_len_kv, d_model)
        :param mask: Optional mask tensor of shape (batch_size, 1, seq_len_q, seq_len_kv)
        :return: Tuple containing:
            - output tensor of shape (batch_size, seq_len_q, d_model)
            - attention weights tensor of shape (batch_size, num_heads, seq_len_q, seq_len_kv)
        """

        Q = self.w_q(x_q)
        K = self.w_k(x_kv)
        V = self.w_v(x_kv)

        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=1)
        context = torch.matmul(attn, V)

        out = self._combine_heads(context)
        out = self.w_o(out)
        return out, attn

batch_size, seq_len, d_model = 2, 5, 64
num_heads = 8

x = torch.rand(batch_size, seq_len, d_model)
mask = torch.ones(batch_size, 1, seq_len, seq_len)

mha_scratch = MultiHeadAttentionCustom(d_model, num_heads)
out, attn = mha_scratch(x, x, mask)
print("From scratch output:", out.shape)
print("Attention map:", attn.shape)

mha_torch = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)

out2, attn2 = mha_torch(x, x, x, attn_mask=None)
print("Torch output:", out2.shape)