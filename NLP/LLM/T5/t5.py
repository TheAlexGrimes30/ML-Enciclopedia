import math
from typing import Optional

import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def make_causal_mask(sz: int, device: torch.device) -> torch.Tensor:
    """
    Causal mask (upper-triangular) for decoder self-attention.
    Returns mask of shape (1, 1, sz, sz) with True for allowed positions.
    In T5 implementation masks are often additive biases; here we return boolean mask
    where True = allowed, False = masked.
    """

    m = torch.tril(torch.ones(sz, sz, dtype=torch.bool, device=device))
    return m.unsqueeze(0).unsqueeze(0)

def make_padding_mask(pad_tokens: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """
    Given input token ids shape (batch, seq_len), return mask (batch, 1, 1, seq_len)
    True for non-padded positions (allowed), False for padding.
    """

    mask = (pad_tokens != pad_token_id).unsqueeze(1).unsqueeze(1)
    return mask

class RelativePositionBias(nn.Module):
    """
    Implements a simplified version of the relative position bias used in T5.
    Each relative distance (difference between query and key positions)
    has a learned bias value for every attention head.

    This bias is added to the raw attention scores to help the model
    encode information about token ordering (e.g., how far apart tokens are).
    """

    def __init__(self, n_heads: int, max_distance: int = 128):
        """
        Initializes the relative position bias module.

        Parameters
            n_heads : int
                Number of attention heads.
            max_distance : int, optional (default = 128)
                Maximum absolute distance between positions that will be assigned
                a unique bias value. Larger distances are clipped to this range.
        """

        super().__init__()
        self.n_heads = n_heads
        self.max_distance = max_distance
        self.bucket_size = 2 * max_distance - 1
        self.relative_attention_bias = nn.Embedding(self.bucket_size, n_heads)
        nn.init.normal_(self.relative_attention_bias.weight, std=0.02)

    def forward(self, qlen: int, klen: int, device: torch.device) -> torch.Tensor:
        """
        Computes the relative position bias for all (query, key) pairs.
        Parameters
            qlen : int
                Length of the query sequence.
            klen : int
                Length of the key sequence.
            device : torch.device
                Device to create tensors on.

        Returns
            torch.Tensor
                Bias tensor of shape (1, n_heads, qlen, klen)
                — ready to be added to attention scores.
        """

        q_pos = torch.arange(qlen, device=device)[:, None]
        k_pos = torch.arange(klen, device=device)[None, :]
        rel_pos = q_pos - k_pos
        clipped = torch.clamp(rel_pos, -self.max_distance + 1, self.max_distance - 1)
        indices = clipped + (self.max_distance - 1)
        bias = self.relative_attention_bias(indices.view(-1)).view(qlen, klen, self.n_heads)
        bias = bias.permute(2, 0, 1).unsqueeze(0)
        return bias

class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Attention module similar to that in Transformer models.
    Projects inputs into multiple attention heads, computes scaled dot-product attention
    for each head, and then concatenates the results back into the original embedding space.

    Supports optional masking and relative position biases.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Initialize the multi-head attention module.

        Parameters
            d_model : int
                Dimensionality of input embeddings (and output embeddings).
            n_heads : int
                Number of attention heads.
            dropout : float, optional (default=0.1)
                Dropout probability applied to attention weights.
        """

        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        for p in [self.q.weight, self.k.weight, self.v.weight, self.o.weight]:
            nn.init.xavier_normal_(p)

    def _shape(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Reshape input to separate attention heads.

        Parameters
            x : torch.Tensor
                Input tensor of shape (batch_size, seq_len, d_model)
            seq_len : int
                Sequence length of the input

        Returns
            torch.Tensor
        Reshaped tensor of shape (batch_size, n_heads, seq_len, d_k)
        """

        b = x.size(0)
        return x.view(b, seq_len, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None,
                rel_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for multi-head attention.

        Parameters
            query : torch.Tensor
                Query embeddings (batch_size, q_len, d_model)
            key : torch.Tensor
                Key embeddings (batch_size, k_len, d_model)
            value : torch.Tensor
                Value embeddings (batch_size, k_len, d_model)
            mask : torch.Tensor, optional
                Boolean mask for attention (batch_size, 1, q_len, k_len) or broadcastable shape
                True = keep, False = mask
            rel_bias : torch.Tensor, optional
                Relative position bias tensor (1, n_heads, q_len, k_len) added to attention scores

        Returns
            torch.Tensor
                Output tensor of shape (batch_size, q_len, d_model)
        """

        b, q_len, _ = query.size()
        _, k_len, _ = key.size()

        Q = self._shape(self.q(query), q_len)
        K = self._shape(self.k(key), k_len)
        V = self._shape(self.v(value), k_len)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if rel_bias is not None:
            scores = scores + rel_bias

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(b, q_len, self.d_model)
        out = self.o(context)
        return out

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network used in Transformer layers.
    Applies two linear transformations with a non-linear activation in between.
    Each token embedding is processed independently, allowing richer representations.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Constructor for the FeedForward layer.

        Parameters
            d_model : int
                Dimensionality of the input and output embeddings.
            d_ff : int
                Dimensionality of the hidden layer (usually larger than d_model).
            dropout : float, optional (default=0.1)
                Dropout probability applied after activation and output.
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

        Parameters
            x : torch.Tensor
                Input tensor of shape (batch_size, seq_len, d_model)

        Returns
            torch.Tensor
                Output tensor of same shape (batch_size, seq_len, d_model)
        """

        x = self.w1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.w2(x)
        x = self.dropout(x)
        return x

class EncoderBlock(nn.Module):
    """
    Single Transformer Encoder Block with:
    - LayerNorm -> Multi-Head Self-Attention (with relative positional bias) -> Residual
    - LayerNorm -> FeedForward -> Residual
    This is similar to T5 encoder block structure.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1, max_res_pos: int = 128):
        """
        Constructor for EncoderBlock.

        Parameters
            d_model : int
                Hidden size of the model / token embeddings.
            n_heads : int
                Number of attention heads.
            d_ff : int
                Hidden dimension of the feed-forward network.
            dropout : float, optional (default=0.1)
                Dropout probability applied after attention and feed-forward.
            max_res_pos : int, optional (default=128)
                Maximum distance for relative positional embeddings.
        """

        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.rel_pos_bias = RelativePositionBias(n_heads, max_distance=max_res_pos)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the EncoderBlock.

        Parameters
            x : torch.Tensor
                Input tensor of shape (batch_size, seq_len, d_model)
            padding_mask : torch.Tensor, optional
                Mask for padded tokens (True=keep, False=mask), shape (batch, 1, 1, seq_len)

        Returns
            torch.Tensor
        Output tensor of shape (batch_size, seq_len, d_model)
        """

        y = self.norm1(x)
        rel_bias = self.rel_pos_bias(y.size(1), y.size(1), device=y.device)
        y_attn = self.self_attn(y, y, y, mask=padding_mask, rel_bias=rel_bias)
        x = x + self.dropout(y_attn)

        z = self.norm2(x)
        z_ff = self.ff(z)
        x = x + self.dropout(z_ff)
        return x

class DecoderBlock(nn.Module):
    """
    Single Transformer Decoder Block with:
    - LayerNorm -> Masked Self-Attention (with relative positional bias) -> Residual
    - LayerNorm -> Cross-Attention (attending to encoder outputs) -> Residual
    - LayerNorm -> FeedForward -> Residual
    This structure follows the T5 decoder block design.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, max_rel_pos: int = 128):
        """
        Constructor for DecoderBlock.

        Parameters
            d_model : int
                Hidden size of the model / token embeddings.
            n_heads : int
                Number of attention heads.
            d_ff : int
                Hidden dimension of the feed-forward network.
            dropout : float, optional (default=0.1)
                Dropout probability applied after attention and feed-forward.
            max_rel_pos : int, optional (default=128)
                Maximum relative distance for self-attention positional embeddings.
        """

        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.self_relpos = RelativePositionBias(n_heads, max_distance=max_rel_pos)

        self.norm_cross = nn.LayerNorm(d_model, eps=1e-6)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)

        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor, self_mask: Optional[torch.Tensor] = None,
                enc_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the DecoderBlock.

        Parameters
            x : torch.Tensor
                Input tensor (batch_size, tgt_seq_len, d_model)
            enc_out : torch.Tensor
                Encoder output tensor (batch_size, src_seq_len, d_model)
            self_mask : torch.Tensor, optional
                Causal mask for decoder self-attention (True=keep, False=mask)
            enc_padding_mask : torch.Tensor, optional
                Mask for padded positions in encoder output (True=keep, False=mask)

        Returns
            torch.Tensor
                Output tensor of shape (batch_size, tgt_seq_len, d_model)
        """

        y = self.norm1(x)
        rel_bias = self.self_relpos(y.size(1), y.size(1), device=y.device)
        y_attn = self.self_attn(y, y, y, mask=self_mask, rel_bias=rel_bias)
        x = x + self.dropout(y_attn)

        q = self.norm_cross(x)
        cross_attn = self.cross_attn(q, enc_out, enc_out, mask=enc_padding_mask, rel_bias=None)
        x = x + self.dropout(cross_attn)

        z = self.norm2(x)
        z_ff = self.ff(z)
        x = x + self.dropout(z_ff)
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, d_ff: int,
                 num_layers: int, dropout: float = 0.1, max_rel_pos: int = 128):
        """
        Transformer Encoder module consisting of:
        - Token embedding layer
        - Dropout
        - A stack of EncoderBlocks
        """

        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout, max_rel_pos) for _ in range(num_layers)
        ])
        nn.init.xavier_normal_(self.embed.weight)

    def forward(self, src_ids: torch.Tensor, src_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the encoder.

        Parameters:
            src_ids: (batch_size, seq_len) Input token ids
            src_padding_mask: (batch_size, 1, 1, seq_len) Mask for padded tokens (True = keep)

        Returns:
            x: (batch_size, seq_len, d_model) Encoder output representations
        """

        x = self.embed(src_ids)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, padding_mask=src_padding_mask)
        return x

class Decoder(nn.Module):
    """
    Transformer Decoder module consisting of:
    - Token embedding layer
    - Dropout
    - Stack of DecoderBlocks
    - Final LayerNorm
    - LM head projecting hidden states to vocabulary logits
    """

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, d_ff: int,
                 num_layers: int, dropout: float = 0.1, max_rel_pos: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout, max_rel_pos) for _ in range(num_layers)
        ])

        self.final_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.xavier_normal_(self.embed.weight)
        nn.init.xavier_normal_(self.lm_head.weight)

    def forward(self, tgt_ids: torch.Tensor, enc_out: torch.Tensor,
                self_mask: Optional[torch.Tensor] = None,
                enc_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the decoder.

        Parameters:
            tgt_ids: (batch_size, tgt_seq_len) Target token ids
            enc_out: (batch_size, src_seq_len, d_model) Encoder outputs
            self_mask: (1, 1, tgt_seq_len, tgt_seq_len) Causal mask for decoder self-attention
            enc_padding_mask: (batch_size, 1, 1, src_seq_len) Encoder padding mask

        Returns:
            logits: (batch_size, tgt_seq_len, vocab_size) Vocabulary logits
        """

        x = self.embed(tgt_ids)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, enc_out, self_mask=self_mask, enc_padding_mask=enc_padding_mask)
        x = self.final_layer_norm(x)
        logits = self.lm_head(x)
        return logits

class T5(nn.Module):
    """
    Simplified T5 model with shared embeddings:
    - Encoder
    - Decoder
    - Weight tying between shared embeddings and LM head
    """

    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8,
                 d_ff: int = 2048, num_layers: int = 6, dropout: float = 0.1,
                 max_rel_pos: int = 128, pad_token_id: int = 0):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.shared_embed = nn.Embedding(vocab_size, d_model)
        nn.init.xavier_normal_(self.shared_embed.weight)

        self.encoder = Encoder(vocab_size, d_model, n_heads, d_ff, num_layers, dropout, max_rel_pos)
        self.decoder = Decoder(vocab_size, d_model, n_heads, d_ff, num_layers, dropout, max_rel_pos)

        self.decoder.lm_head.weight = self.shared_embed.weight

    def get_padding_mask(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Compute padding mask for input token ids.

        Returns:
            mask: (batch_size, 1, 1, seq_len) True = non-padded tokens
        """

        return (ids != self.pad_token_id).unsqueeze(1).unsqueeze(1)

    def forward(self, src_ids: torch.Tensor, tgt_ids_in: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of T5.

        Parameters:
            src_ids: (batch_size, src_seq_len) Source token ids
            tgt_ids_in: (batch_size, tgt_seq_len) Input target ids for teacher forcing

        Returns:
            logits: (batch_size, tgt_seq_len, vocab_size) Decoder output logits
        """

        enc_mask = self.get_padding_mask(src_ids)
        dec_padding_mask = self.get_padding_mask(src_ids)
        enc_out = self.encoder(src_ids, src_padding_mask=enc_mask)
        device = src_ids.device
        causal = make_causal_mask(tgt_ids_in.size(1), device=device)
        logits = self.decoder(tgt_ids_in, enc_out, self_mask=causal, enc_padding_mask=dec_padding_mask)
        return logits

    @torch.no_grad()
    def generate_greedy(self, src_ids: torch.Tensor, max_len: int = 50, start_token_id: int = 1,
                        end_token_id: int = 2) -> torch.Tensor:
        """
        Greedy decoding for sequence generation.

        Parameters:
            src_ids: (batch_size, src_seq_len) Source token ids
            max_len: maximum target sequence length
            start_token_id: start-of-sequence token id
            end_token_id: end-of-sequence token id

        Returns:
            gen: (batch_size, gen_len) Generated token ids
        """
        enc_mask = self.get_padding_mask(src_ids)
        enc_out = self.encoder(src_ids, src_padding_mask=enc_mask)
        batch_size = src_ids.size(0)
        device = src_ids.device

        cur = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        outputs = []

        for step in range(max_len):
            causal = make_causal_mask(cur.size(1), device=device)
            logits = self.decoder(cur, enc_out, self_mask=causal, enc_padding_mask=enc_mask)  # (b, cur_len, vocab)
            next_logits = logits[:, -1, :]
            next_token = next_logits.argmax(dim=-1, keepdim=True)  # greedy
            outputs.append(next_token)
            cur = torch.cat([cur, next_token], dim=1)
            finished = finished | (next_token.squeeze(1) == end_token_id)
            if finished.all():
                break
        gen = torch.cat(outputs, dim=1)
        return gen

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

    model = T5(vocab_size=vocab_size, d_model=128, n_heads=4, d_ff=512, num_layers=3, dropout=0.1)
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
            decoder_input = torch.cat([torch.full((t.size(0), 1), start, dtype=torch.long, device=device), t[:, :-1]], dim=1)
            logits = model(s, decoder_input)
            loss = criterion(logits.view(-1, vocab_size), t.view(-1))
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {total_loss/len(loader):.4f}")

    model.eval()
    smooth_fn = SmoothingFunction().method1
    bleu_scores = []
    with torch.no_grad():
        for s, t in loader:
            s = s.to(device)
            t = t.to(device)
            gen = model.generate_greedy(s, max_len=seq_len, start_token_id=start, end_token_id=eos)
            gen = gen[:, :seq_len].cpu().tolist()
            for i in range(t.size(0)):
                ref = [str(tok.item()) for tok in t[i]]
                cand = [str(tok) for tok in gen[i]]
                bleu = sentence_bleu([ref], cand, smoothing_function=smooth_fn)
                bleu_scores.append(bleu)
    print("Avg BLEU (toy):", sum(bleu_scores) / len(bleu_scores))
    