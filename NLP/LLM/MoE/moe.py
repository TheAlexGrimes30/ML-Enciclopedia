import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

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
        return x.view(b, self.n_heads, seq_len, self.d_k)

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

        b, s, _ = x.size()

        Q = self._shape(self.q(x), s)
        K = self._shape(self.k(x), s)
        V = self._shape(self.v(x), s)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(b, s, self.d_model)
        return self.o(out)

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network used in Transformer blocks.

    This module applies two linear transformations with a non-linear
    activation function in between. The same feed-forward network
    is applied independently to each position (token) in the sequence.

    Architecture:
        Linear(d_model → d_ff)
        GELU
        Dropout
        Linear(d_ff → d_model)
        Dropout
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Initializes the feed-forward network.

        Args:
            d_model (int):
                Dimensionality of the model embeddings.
                This is the input and output dimension of the FFN.

            d_ff (int):
                Hidden layer size of the feed-forward network.
                Typically larger than d_model (e.g., 4x).

            dropout (float):
                Dropout probability applied after each linear layer.
        """

        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feed-forward network.

        Args:
            x (torch.Tensor):
                Input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
            torch.Tensor:
                Output tensor of the same shape (batch_size, sequence_length, d_model).
        """

        x = F.gelu(self.w1(x))
        x = self.dropout(x)
        return self.dropout(self.w2(x))

class MoEFeedForward(nn.Module):
    """
    Mixture-of-Experts (MoE) Feed-Forward layer.

    This module replaces a standard Transformer feed-forward block
    with a Mixture-of-Experts mechanism. Each token is routed to exactly
    one expert (Top-1 routing) using a learned gating network.

    Key features:
    - Token-level routing (each token chooses one expert)
    - Capacity constraint per expert (token dropping)
    - Routing weights (soft mixture scaling)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        dropout: float = 0.1,
        capacity_factor: float = 1.25
    ):
        """
        Initialize the MoE feed-forward layer.

        Args:
            d_model (int): Model hidden size (embedding dimension).
            d_ff (int): Hidden size of each expert's feed-forward network.
            num_experts (int): Number of parallel experts.
            dropout (float): Dropout probability inside experts.
            capacity_factor (float): Multiplier controlling how many tokens
                each expert is allowed to process:
                capacity = capacity_factor * (tokens / num_experts).
        """

        super().__init__()
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor

        self.gate = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([
            FeedForward(d_model, d_ff, dropout)
            for _ in range(num_experts)
        ])

        self.register_buffer("expert_counts", torch.zeros(num_experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MoE layer.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, seq_len, d_model].

        Returns:
            torch.Tensor: Output tensor of shape [batch, seq_len, d_model].
        """

        b, s, d = x.shape
        x_flat = x.view(-1, d)
        T = x_flat.size(0)

        logits = self.gate(x_flat)
        probs = F.softmax(logits, dim=-1)

        expert_ids = torch.argmax(probs, dim=-1)

        expert_weights = probs.gather(
            1, expert_ids.unsqueeze(1)
        ).squeeze(1)

        capacity = int(self.capacity_factor * T / self.num_experts)

        out = torch.zeros_like(x_flat)

        for i, expert in enumerate(self.experts):
            mask = expert_ids == i
            idx = mask.nonzero(as_tuple=False).squeeze(-1)

            if idx.numel() == 0:
                continue

            if idx.numel() > capacity:
                idx = idx[:capacity]

            expert_out = expert(x_flat[idx])
            out[idx] = expert_weights[idx].unsqueeze(1) * expert_out

            with torch.no_grad():
                self.expert_counts[i] += idx.numel()

        return out.view(b, s, d)

class GPTMoEBlock(nn.Module):
    """
    A single GPT-style Transformer block with Mixture-of-Experts (MoE)
    feed-forward network.

    This block follows the Pre-LayerNorm GPT architecture:
        1) LayerNorm
        2) Causal Multi-Head Self-Attention
        3) Residual connection
        4) LayerNorm
        5) MoE Feed-Forward Network
        6) Residual connection

    The MoE layer replaces the standard dense FFN and routes each token
    to one of multiple expert networks.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, num_experts: int):
        """
        Initialize the GPT-MoE block.

        Args:
            d_model (int):
                Model embedding dimension.
            n_heads (int):
                Number of attention heads.
            d_ff (int):
                Hidden dimension of each expert's feed-forward network.
            num_experts (int):
                Number of experts in the MoE layer.
        """

        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.moe = MoEFeedForward(d_model, d_ff, num_experts)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the GPT-MoE block.

        Args:
            x (torch.Tensor):
                Input tensor of shape (batch_size, sequence_length, d_model).
            mask (Optional[torch.Tensor]):
                Causal attention mask to prevent attending to future tokens.

        Returns:
            torch.Tensor:
                Output tensor of shape (batch_size, sequence_length, d_model).
        """

        x = x + self.attn(self.ln1(x), mask)
        x = x + self.moe(self.ln2(x))
        return x

class GPTMoE(nn.Module):
    """
    GPT-style autoregressive Transformer model with Mixture-of-Experts (MoE)
    feed-forward layers.

    The model consists of:
        - Token embeddings
        - Learned positional embeddings
        - A stack of GPT-MoE blocks (self-attention + MoE FFN)
        - Final layer normalization
        - Linear language modeling head

    This model predicts the next token in a sequence using causal
    self-attention.
    """

    def __init__(self, vocab: int,
                 d_model: int = 128, n_heads: int = 4,
                 d_ff: int = 256, layers: int = 3,
                 experts: int = 4, pos_vocab: int = 512):
        """
        Initialize the GPT-MoE model.

        Args:
            vocab (int):
                Size of the vocabulary.
            d_model (int):
                Embedding dimension of the model.
            n_heads (int):
                Number of attention heads per Transformer block.
            d_ff (int):
                Hidden dimension of the feed-forward networks inside experts.
            layers (int):
                Number of GPT-MoE blocks.
            experts (int):
                Number of experts per MoE feed-forward layer.
            pos_vocab (int):
                Maximum sequence length supported by positional embeddings.
        """

        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(pos_vocab, d_model)

        self.blocks = nn.ModuleList([
            GPTMoEBlock(d_model, n_heads, d_ff, experts)
            for _ in range(layers)
        ])

        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GPT-MoE model.

        Args:
            x (torch.Tensor):
                Input tensor of token IDs with shape
                (batch_size, sequence_length).

        Returns:
            torch.Tensor:
                 Logits over the vocabulary with shape
                (batch_size, sequence_length, vocab_size).
        """

        b, s = x.shape
        pos = torch.arange(s, device=x.device).unsqueeze(0)
        x = self.emb(x) + self.pos_emb(pos)
        mask = make_causal_mask(s, x.device)

        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.ln(x))

def evaluate(model: nn.Module,
             loader: DataLoader,
             vocab: int,
             device: torch.device) -> Tuple[float, float, float]:
    """
    Evaluate a language model on a dataset.

    Computes:
        - Average token-level cross-entropy loss
        - Perplexity
        - Token-level accuracy

    Args:
        model (nn.Module):
            The language model to evaluate.
        loader (DataLoader):
            DataLoader providing evaluation batches.
        vocab (int):
            Vocabulary size.
        device (torch.device):
            Device on which to run evaluation.

    Returns:
        Tuple[float, float, float]:
            - Average loss per token
            - Perplexity
            - Token-level accuracy
    """

    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0
    total_tokens = 0
    correct = 0

    pbar = tqdm(loader, desc="Evaluation", leave=False)

    with torch.no_grad():
        for _, t in pbar:
            t = t.to(device)

            inp = torch.cat(
                [torch.ones(t.size(0), 1, device=device, dtype=torch.long), t[:, :-1]],
                dim=1
            )

            logits = model(inp)
            loss = loss_fn(logits.view(-1, vocab), t.view(-1))

            total_loss += loss.item()
            total_tokens += t.numel()
            correct += (logits.argmax(-1) == t).sum().item()

            pbar.set_postfix(
                ppl=math.exp(total_loss / max(total_tokens, 1)),
                acc=correct / max(total_tokens, 1),
            )

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    acc = correct / total_tokens
    return avg_loss, ppl, acc

if __name__ == "__main__":
    vocab = 100
    seq = 12
    samples = 300
    batch = 8
    epochs = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = torch.randint(3, vocab, (samples, seq))
    loader = DataLoader(TensorDataset(data, data), batch_size=batch, shuffle=True)

    model = GPTMoE(vocab).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch + 1}/{epochs} [train]",
            leave=False
        )

        for _, t in pbar:
            t = t.to(device)

            inp = torch.cat(
                [torch.ones(t.size(0), 1, device=device, dtype=torch.long), t[:, :-1]],
                dim=1
            )

            logits = model(inp)
            loss = loss_fn(logits.view(-1, vocab), t.view(-1))

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch + 1} avg loss: {epoch_loss / len(loader):.4f}")

    loss, ppl, acc = evaluate(model, loader, vocab, device)

    print("\n=== Custom GPT-MoE ===")
    print(f"Loss: {loss:.4f}")
    print(f"PPL: {ppl:.2f}")
    print(f"Token Acc: {acc:.4f}")

