from typing import Tuple
import torch.nn.functional as F
import torch
from torch import nn


class MultiplicativeAttention(nn.Module):
    """
    Multiplicative (Luong) Attention.

    Attributes:
        d_model (int): Dimension of hidden states.
        w_a (nn.Linear): Linear layer для согласования размерностей Q и K.
        dropout (nn.Dropout): Dropout для весов внимания.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        """
        Constructor

        :param d_model: Dimension of input embeddings / hidden states
        :param dropout: Dropout probability applied to attention weights
        """

        super().__init__()
        self.d_model = d_model
        self.w_a = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for multiplicative attention.

        :param query: Tensor of shape (batch_size, seq_len_q, d_model)
        :param key: Tensor of shape (batch_size, seq_len_k, d_model)
        :param value: Tensor of shape (batch_size, seq_len_k, d_model)
        :param mask: Optional mask tensor of shape (batch_size, 1, seq_len_q, seq_len_k)
        :return: Tuple containing:
            - context: Weighted sum of values, shape (batch_size, seq_len_q, d_model)
            - attn_weights: Attention weights, shape (batch_size, seq_len_q, seq_len_k)
        """

        query_proj = self.w_a(query)
        scores = torch.matmul(query_proj, key.transpose(-2, -1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, value)

        return context, attn_weights

batch_size, seq_len_q, seq_len_kv, d_model = 2, 5, 5, 64

query = torch.rand(batch_size, seq_len_q, d_model)
key = torch.rand(batch_size, seq_len_kv, d_model)
value = torch.rand(batch_size, seq_len_kv, d_model)
mask = torch.ones(batch_size, 1, seq_len_q, seq_len_kv)

mul_attn = MultiplicativeAttention(d_model)
context, attn_weights = mul_attn(query, key, value, mask)

print("Context shape:", context.shape)
print("Attention weights shape:", attn_weights.shape)