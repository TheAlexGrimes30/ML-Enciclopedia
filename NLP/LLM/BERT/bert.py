import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


def make_padding_mask(x: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """
    Create a mask to ignore padding tokens in attention.

    Args:
        x (torch.Tensor): Input tensor (batch_size, seq_len)
        pad_token_id (int): ID of padding token

    Returns:
        torch.Tensor: Mask of shape (batch_size, 1, 1, seq_len)
    """

    return (x != pad_token_id).unsqueeze(1).unsqueeze(2)

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention module.

    Splits the input into multiple heads, computes scaled dot-product attention
    for each head independently, and then concatenates the results.

    Args:
        d_model (int): Total hidden size of the model.
        n_heads (int): Number of attention heads.
        dropout (float): Dropout applied to attention weights.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Initializes the Multi-Head Attention module.

        Args:
            d_model (int): Total dimensionality of the model (embedding size).
            n_heads (int): Number of attention heads to split d_model into.
            dropout (float): Dropout probability applied to attention weights.

        Notes:
            - d_k is computed as d_model // n_heads, representing the dimension
                of each individual attention head.
            - Four linear layers are created:
                * q: projects input into queries
                * k: projects input into keys
                * v: projects input into values
                * o: final output projection that recombines all heads
            - Xavier initialization is applied to all weight matrices
                to improve training stability.
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

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshapes (B, seq_len, d_model) into (B, n_heads, seq_len, d_k).

        Then swaps dimensions so we get:
            - heads dimension before seq_len (as expected in attention)
        """

        b, seq_len, _ = x.size()
        return x.view(b, seq_len, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of multi-head attention.

        Steps:
            1. Compute Q, K, V matrices via linear layers
            2. Reshape and split into heads
            3. Compute scaled dot-product attention
            4. Apply mask (if provided)
            5. Softmax over scores → attention weights
            6. Multiply weights by V to get context vectors
            7. Concatenate all heads back together
            8. Final linear projection through self.o

        Args:
            x (Tensor): Input tensor of shape (B, seq_len, d_model)
            mask (Tensor, optional): Attention mask broadcastable to
                                             (B, n_heads, seq_len, seq_len)

        Returns:
            Tensor of shape (B, seq_len, d_model)
        """

        b, seq_len, _ = x.size()
        Q = self._shape(self.q(x))
        K = self._shape(self.k(x))
        V = self._shape(self.v(x))

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(b, seq_len, self.d_model)
        return self.o(context)

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN) used inside Transformer blocks.

    Applies:
        Linear -> GELU -> Dropout -> Linear -> Dropout

    This module transforms each token embedding independently,
    expanding dimensionality to d_ff and projecting back to d_model.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Initializes the feed-forward network.

        Args:
            d_model (int): Dimensionality of the input and output embeddings.
            d_ff (int): Dimensionality of the intermediate hidden layer.
            dropout (float): Dropout probability applied after activations.

        Components:
            w1: Expands dimension from d_model → d_ff.
            w2: Projects back from d_ff → d_model.
            dropout: Helps regularize the network and prevent overfitting.
        """

        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_normal_(self.w1.weight)
        nn.init.xavier_normal_(self.w2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FFN.

        Steps:
            1. Linear projection to higher dimension.
            2. GELU non-linearity.
            3. Dropout to reduce overfitting.
            4. Linear projection back to d_model.
            5. Another dropout for regularization.

        Args:
            x (Tensor): Input tensor of shape (B, seq_len, d_model)

        Returns:
            Tensor: Output tensor of shape (B, seq_len, d_model)
        """

        x = self.w1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.w2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    """
    A single Transformer block consisting of:
        - LayerNorm + Multi-Head Self-Attention
        - Residual connection
        - LayerNorm + Feed-Forward Network (FFN)
        - Another residual connection

    This is the core building block used in Transformer encoder/decoder stacks.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initializes all components of the Transformer block.

        Args:
            d_model (int): Dimensionality of the model (embedding size).
            n_heads (int): Number of attention heads.
            d_ff (int): Dimensionality of the intermediate FFN layer.
            dropout (float): Dropout probability used throughout the block.
        """

        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the Transformer block.

        Args:
            x (Tensor): Input tensor of shape (B, seq_len, d_model).
            mask (Tensor, optional): Attention mask (e.g., causal or padding mask).

        Returns:
            Tensor: Output tensor with the same shape as input.
        """

        y = self.norm1(x)
        y_attn = self.attn(y, mask)
        x = x + self.dropout(y_attn)
        z = self.norm2(x)
        z_ff = self.ff(z)
        x = x + self.dropout(z_ff)
        return x

class BERTModel(nn.Module):
    """
    Simplified BERT-style Transformer Encoder.

    Components:
    - Token embeddings
    - Segment embeddings (for sentence pairs)
    - Positional embeddings
    - Stack of Transformer blocks (attention + feed-forward)
    - Final LayerNorm
    """

    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4, d_ff: int = 512,
                 num_layers: int = 4, max_seq_len: int = 64, dropout: float = 0.1, pad_token_id: int = 0):
        """
        Initialize BERT model.

        Args:
            vocab_size: Size of the vocabulary.
            d_model: Hidden embedding dimension.
            n_heads: Number of attention heads.
            d_ff: Feed-forward hidden dimension.
            num_layers: Number of Transformer blocks.
            max_seq_len: Maximum sequence length for positional embeddings.
            dropout: Dropout probability.
            pad_token_id: ID of the padding token.
        """

        super().__init__()
        self.pad_token_id = pad_token_id
        self.embed  = nn.Embedding(vocab_size, d_model)
        self.seg_embed = nn.Embedding(2, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, dropout)
                                     for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model, eps=1e-6)
        nn.init.xavier_normal_(self.embed.weight)

        self.mlm_head = nn.Linear(d_model, vocab_size)
        nn.init.xavier_normal_(self.mlm_head.weight)

    def forward(self, x: torch.Tensor, seg: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the BERT encoder.

        Args:
            x: Input tensor of token IDs, shape (batch_size, seq_len)
            seg: Optional segment IDs (0 or 1) for sentence pairs, shape (batch_size, seq_len)
            mask: Optional attention mask to ignore padding tokens, shape (batch_size, 1, 1, seq_len)

        Returns:
            Tensor of shape (batch_size, seq_len, d_model), the final hidden states
        """

        b, seq_len = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(b, seq_len)
        x = self.embed(x) + self.pos_embed(positions)

        if seg is not None:
            x = x + self.seg_embed(seg)

        if mask is not None:
            mask = make_padding_mask(x, self.pad_token_id)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.ln_f(x)
        return x

class BERTClassifier(nn.Module):
    """
    BERT-based model for text classification with optional Masked Language Modeling (MLM) head.

    Components:
    - BERT encoder (bidirectional Transformer)
    - Classification head ([CLS] token used)
    - Optional MLM head for pretraining tasks
    """

    def __init__(self, vocab_size: int, num_classes: int, max_seq_len: int = 64, d_model: int = 128,
                 n_heads: int = 4, d_ff: int = 512,
                 num_layers: int = 4, dropout: float = 0.1, pad_token_id: int = 0):
        """
        Initialize the BERT classifier.

        Args:
            vocab_size: Size of the vocabulary.
            num_classes: Number of output classes for classification.
            max_seq_len: Maximum sequence length.
            d_model: Hidden embedding dimension.
            n_heads: Number of attention heads.
            d_ff: Hidden dimension for feed-forward networks.
            num_layers: Number of Transformer blocks in the encoder.
            dropout: Dropout probability.
            pad_token_id: ID of the padding token.
        """

        super().__init__()
        self.bert = BERTModel(vocab_size, d_model, n_heads, d_ff, num_layers, max_seq_len, dropout, pad_token_id)
        self.cls_head = nn.Linear(d_model, num_classes)
        nn.init.xavier_normal_(self.cls_head.weight)
        self.mlm_head = self.bert.mlm_head

    def forward(self, x: torch.Tensor, seg: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the classifier.

        Args:
            x: Input token IDs, shape (batch_size, seq_len)
            seg: Optional segment IDs (0 or 1), shape (batch_size, seq_len)

        Returns:
            Tuple:
                - logits_cls: Classification logits for each sample, shape (batch_size, num_classes)
                - logits_mlm: Optional MLM logits for each token, shape (batch_size, seq_len, vocab_size)
        """

        enc = self.bert(x, seg)
        cls_token = enc[:, 0, :]
        logits_cls = self.cls_head(cls_token)
        logits_mlm = self.mlm_head(enc)
        return logits_cls, logits_mlm

if __name__ == "__main__":
    vocab_size = 100
    num_classes = 3
    pad = 0
    seq_len = 12
    samples = 500
    batch_size = 16
    mask_prob = 0.15

    X = torch.randint(1, vocab_size, (samples, seq_len))
    y = torch.randint(0, num_classes, (samples,))

    X_mlm = X.clone()
    rand_mask = torch.rand_like(X, dtype=torch.float) < mask_prob
    X_mlm[rand_mask] = 0

    dataset = TensorDataset(X, y, X_mlm, rand_mask)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BERTClassifier(vocab_size=vocab_size, num_classes=num_classes, max_seq_len=seq_len)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion_cls = nn.CrossEntropyLoss(ignore_index=pad)
    criterion_mlm = nn.CrossEntropyLoss(ignore_index=pad)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    model.train()
    for epoch in range(5):
        total_loss = 0.0
        for batch_x, batch_y, batch_mlm_x, batch_mask in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_mlm_x = batch_mlm_x.to(device)
            batch_mask = batch_mask.to(device)

            logits_cls, logits_mlm = model(batch_x)
            loss_cls = criterion_cls(logits_cls, batch_y)

            logits_mlm_flat = logits_mlm.view(-1, vocab_size)
            target_mlm_flat = batch_x.view(-1)
            mask_flat = batch_mask.view(-1).bool()

            loss_mlm = criterion_mlm(logits_mlm_flat[mask_flat], target_mlm_flat[mask_flat])

            loss = loss_cls + loss_mlm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1} Loss: {total_loss / len(loader):.4f}")

    # Evaluation
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_x, batch_y, _, _ in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits_cls, _ = model(batch_x)
            preds = logits_cls.argmax(dim=-1)
            correct += (preds == batch_y).sum().item()

    print("Train Accuracy:", correct / samples)