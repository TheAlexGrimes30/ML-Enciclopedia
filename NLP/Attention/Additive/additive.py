import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple

class AdditiveAttention(nn.Module):
    """
    Additive (Bahdanau) Attention.

    Attributes:
        d_model (int): Dimension of hidden states.
        w_q (nn.Linear): Linear layer to project query.
        w_k (nn.Linear): Linear layer to project key.
        v (nn.Linear): Linear layer to compute attention scores.
        dropout (nn.Dropout): Dropout applied to attention weights.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        """
        Constructor

        :param d_model: Dimension of hidden states (query/key/value)
        :param dropout: Dropout probability applied to attention weights
        """

        super().__init__()
        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for additive attention.

        :param query: Tensor of shape (batch_size, seq_len_q, d_model)
        :param key: Tensor of shape (batch_size, seq_len_k, d_model)
        :param value: Tensor of shape (batch_size, seq_len_k, d_model)
        :param mask: Optional mask tensor of shape (batch_size, 1, seq_len_q, seq_len_k)
        :return: Tuple containing:
            - context: Weighted sum of values, shape (batch_size, seq_len_q, d_model)
            - attn_weights: Attention weights, shape (batch_size, seq_len_q, seq_len_k)
        """

        Q_proj = self.w_q(query)
        K_proj = self.w_k(key)

        scores = self.v(torch.tanh(Q_proj.unsqueeze(2) + K_proj.unsqueeze(1))).squeeze(-1)

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

add_attn = AdditiveAttention(d_model)
context, attn_weights = add_attn(query, key, value, mask)

print("Context shape:", context.shape)
print("Attention weights shape:", attn_weights.shape)
