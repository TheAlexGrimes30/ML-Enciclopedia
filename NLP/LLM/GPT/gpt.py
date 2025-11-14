import math
from typing import Optional

import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


def make_causal_mask(sz: int, device: torch.device) -> torch.Tensor:
    """
    Create a causal (autoregressive) attention mask for decoder self-attention.

    In GPT-like (autoregressive) architectures, each token is only allowed
    to attend to itself and all previous tokens — never future ones.
    This ensures that the model generates text from left to right
    without "cheating" by seeing future words.

    Args:
        sz (int): Sequence length.
        device (torch.device): Device on which to create the mask.

    Returns:
        torch.Tensor: A mask of shape (1, 1, sz, sz) where True means
        a token can attend, and False means it cannot (future positions).
    """

    m = torch.tril(torch.ones(sz, sz, dtype=torch.bool, device=device))
    return m.unsqueeze(0).unsqueeze(0)

class MultiHeadAttention(nn.Module):
    """
    Implements the Multi-Head Self-Attention mechanism used in Transformer-based architectures (e.g., GPT, BERT, T5).

    Each attention head learns to focus on different parts of the input sequence,
    allowing the model to capture multiple types of dependencies in parallel.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model (int): Dimensionality of the input embeddings.
            n_heads (int): Number of parallel attention heads.
            dropout (float): Dropout rate applied after the attention softmax.

        Each head operates on a subspace of size d_k = d_model / n_heads.
        """

        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        for p in [self.q.weight, self.k.weight, self.v.weight, self.o.weight]:
            nn.init.xavier_normal_(p)

    def _shape(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Reshapes the input tensor for multi-head processing.

        Converts a tensor of shape (batch, seq_len, d_model) into
        (batch, n_heads, seq_len, d_k) to enable parallel attention computation.

        Args:
            x (torch.Tensor): Input tensor (batch, seq_len, d_model)
            seq_len (int): Sequence length
        Returns:
            torch.Tensor: Reshaped tensor (batch, n_heads, seq_len, d_k)
        """

        b = x.size(0)
        return x.view(b, seq_len, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs the forward pass of Multi-Head Attention.

        Steps:
        1. Project input into query (Q), key (K), and value (V) spaces.
        2. Compute scaled dot-product attention for each head.
        3. Apply optional causal or padding mask.
        4. Concatenate all attention heads and apply the final linear projection.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, d_model)
            mask (Optional[torch.Tensor]): Attention mask of shape (1, 1, seq_len, seq_len),
                                                   used to prevent attending to padding or future tokens.

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, d_model)
        """

        b, seq_len, _ = x.size()
        Q = self._shape(self.q(x), seq_len)
        K = self._shape(self.k(x), seq_len)
        V = self._shape(self.v(x), seq_len)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(b, seq_len, self.d_model)
        out = self.o(context)
        return out

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN) used in Transformer/GPT blocks.

    Each token embedding is processed independently, applying two linear transformations
    with a non-linear activation in between. This allows the model to learn richer
    representations beyond attention.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize the FeedForward layer.

        Parameters:
        - d_model (int): Dimensionality of input and output embeddings.
        - d_ff (int): Dimensionality of the hidden layer (usually larger than d_model).
        - dropout (float): Dropout probability applied after activation and output.
        """

        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_normal_(self.w1.weight)
        nn.init.xavier_normal_(self.w2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FeedForward layer.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
        - torch.Tensor: Output tensor of same shape (batch_size, seq_len, d_model)
        """

        x = self.w1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.w2(x)
        x = self.dropout(x)
        return x

class GPTBlock(nn.Module):
    """
    Single GPT Transformer Block.

    Consists of:
    1. LayerNorm -> Multi-Head Self-Attention -> Residual connection
    2. LayerNorm -> FeedForward -> Residual connection

    This structure is standard in GPT models.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize the GPT block.

        Parameters:
        - d_model (int): Dimensionality of input and output embeddings.
        - n_heads (int): Number of attention heads.
        - d_ff (int): Hidden dimension of the feed-forward network.
        - dropout (float): Dropout probability applied after attention and feed-forward.
        """

        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the GPT block.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
        - mask (torch.Tensor, optional): Causal mask for self-attention (True=keep, False=mask)

        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """

        y = self.norm1(x)
        y_attn = self.attn(y, mask)
        x = x + self.dropout(y_attn)
        z = self.norm2(x)
        z_ff = self.ff(z)
        x = x + self.dropout(z_ff)
        return x

class GPT(nn.Module):
    """
    Simplified GPT-like Transformer model for autoregressive text generation.

    Components:
    - Token embeddings
    - Positional embeddings
    - Stack of GPTBlocks (attention + feedforward)
    - Final layer normalization
    - Output head projecting to vocabulary logits
    """

    def __init__(self, vocab_size: int, d_model: int  = 128, n_heads: int = 4,
                 d_ff: int = 512, num_layers: int = 4, dropout: float = 0.1,
                 pad_token_id: int = 0, max_sequence: int = 512):
        """
        Initialize GPT model.

        Args:
            vocab_size: Size of the vocabulary.
            d_model: Hidden embedding dimension.
            n_heads: Number of attention heads.
            d_ff: Feed-forward hidden dimension.
            num_layers: Number of transformer blocks.
            dropout: Dropout probability.
            pad_token_id: Padding token ID.
            max_sequence: Maximum sequence length for positional embeddings.
        """

        super().__init__()
        self.pad_token_id = pad_token_id
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_sequence, d_model)
        self.layers = nn.ModuleList([GPTBlock(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model, eps=1e-6)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.xavier_normal_(self.embed.weight)
        nn.init.xavier_normal_(self.head.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GPT model.

        Args:
            x: Input tensor of token IDs, shape (batch_size, seq_len).

        Returns:
            logits: Tensor of shape (batch_size, seq_len, vocab_size) representing
                unnormalized log probabilities for each token in the vocabulary.
        """

        b, seq_len = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(b, seq_len)
        x = self.embed(x) + self.pos_embed(positions)
        mask = make_causal_mask(seq_len, x.device)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(self, x: torch.Tensor, max_len: int = 50, temperature: float = 1.0,
                 top_k: int = 50, top_p: float = 0.9, end_token_id: int = 2) -> torch.Tensor:
        """
        Autoregressive text generation using top-k and/or nucleus (top-p) sampling.

        Args:
            x: Input tensor of token IDs, shape (batch_size, seq_len), typically a start token.
            max_len: Maximum number of tokens to generate.
            temperature: Temperature for controlling randomness (higher = more random).
            top_k: Keep only the top-k logits before sampling.
            top_p: Keep only the smallest set of logits whose cumulative probability >= top_p.
            end_token_id: Token ID indicating the end of sequence.

        Returns:
            Tensor of generated token IDs of shape (batch_size, generated_seq_len),
            where generated_seq_len <= max_len.
        """

        cur = x.clone()
        for _ in range(max_len):
            logits = self.forward(cur)[:, -1, :]
            logits = logits / temperature

            if top_k is not None and top_k > 0:
                top_k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, top_k)
                min_values = values[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < min_values,
                    torch.full_like(logits, -float("inf")),
                    logits
                )

            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cumulative = torch.cumsum(probs, dim=-1)

                mask = cumulative > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                sorted_logits[mask] = -float("inf")
                logits = torch.zeros_like(logits).scatter(-1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            cur = torch.cat([cur, next_token], dim=1)

            if (next_token == end_token_id).all():
                break

        return cur[:, x.size(1):]

if __name__ == "__main__":
    vocab_size = 100
    pad = 0
    start = 1
    eos = 2
    seq_len = 12
    samples = 200
    batch_size = 8

    src = torch.randint(3, vocab_size, (samples, seq_len))
    tgt = src.clone()

    dataset = TensorDataset(src, tgt)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GPT(vocab_size=vocab_size, d_model=128, n_heads=4, d_ff=512, num_layers=3, dropout=0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=pad)

    model.train()
    for epoch in range(3):
        total_loss = 0.0
        for s, t in loader:
            s = s.to(device)
            t = t.to(device)
            decoder_input = torch.cat([torch.full((t.size(0), 1), start, dtype=torch.long, device=device), t[:, :-1]],
                                      dim=1)
            logits = model(decoder_input)
            loss = criterion(logits.view(-1, vocab_size), t.view(-1))
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1} loss: {total_loss / len(loader):.4f}")

    model.eval()
    smooth_fn = SmoothingFunction().method1
    bleu_scores = []
    with torch.no_grad():
        for s, t in loader:
            s = s.to(device)
            t = t.to(device)
            gen = model.generate(torch.full((t.size(0), 1), start, dtype=torch.long, device=device), max_len=seq_len)
            gen = gen[:, :seq_len].cpu().tolist()
            for i in range(t.size(0)):
                ref = [str(tok.item()) for tok in t[i]]
                cand = [str(tok) for tok in gen[i]]
                bleu = sentence_bleu([ref], cand, smoothing_function=smooth_fn)
                bleu_scores.append(bleu)
    print("Avg BLEU (toy):", sum(bleu_scores) / len(bleu_scores))
