import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """
    Generate a causal (subsequent) mask for decoder self-attention.

    In Transformer decoders, each position can only attend to
    previous positions and itself. This function returns an upper-triangular
    boolean matrix where True values indicate masked positions.

    :param sz: sequence length (int)
    :return: boolean mask tensor of shape (sz, sz) with True above the diagonal (torch.Tensor)
    """

    return torch.triu(torch.ones(sz, sz), diagonal=1).bool()

def apply_span_masking(x: torch.Tensor, mask_ratio: float = 0.15,
                       mask_token_id: int = 0, poisson_lambda: float = 3.0) -> torch.Tensor:
    """
    Applies span-based masking to the input token sequences.

    This function masks contiguous spans of tokens rather than masking individual tokens independently.
    The span lengths are sampled from a Poisson distribution, which encourages short but variable-length spans.
    The total number of masked tokens per sequence is determined by `mask_ratio`.

    Args:
        x (torch.Tensor): Input token tensor of shape (batch_size, seq_len).
        mask_ratio (float): Fraction of tokens to be masked within each sequence.
        mask_token_id (int): Token ID used for masking (e.g., <mask> token).
        poisson_lambda (float): Lambda parameter for the Poisson distribution that controls average span length.

    Returns:
        torch.Tensor: A masked copy of `x` with certain spans replaced by the mask token.
    """

    x_masked = x.clone()
    batch_size, seq_len = x.size()
    num_mask = max(1, int(seq_len * mask_ratio))

    for i in range(batch_size):
        masked_count = 0
        while masked_count < num_mask:
            span_len = min(torch.poisson(torch.tensor(poisson_lambda)).item() + 1, seq_len - 1)
            start = torch.randint(0, seq_len, (1,)).item()
            end = min(start + int(span_len), seq_len)
            if (x_masked[i, start:end] != mask_token_id).any():
                x_masked[i, start:end] = mask_token_id
                masked_count += (end - start)
    return x_masked

class MultiHeadAttentionCustom(nn.Module):
    """
    Custom implementation of Multi-Head Attention.

    This module projects the input into multiple attention heads, performs scaled dot-product attention
    independently for each head, and then concatenates the results back into the original representation space.

    Args:
        d_model (int): Dimensionality of the input embedding.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability applied to attention weights.

    Notes:
        - The input dimension must be divisible by the number of heads.
        - Each head operates on a subspace of dimension d_model / num_heads.
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Constructor

        :param d_model: Dimension of input embeddings.
        :param num_heads: Number of attention heads.
        :param dropout: Dropout probability applied to attention weights.
        """

        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

        for lin in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_normal_(lin.weight)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshapes the input tensor so that multiple attention heads can be processed in parallel.

        Args:
            x (Tensor): Input of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor: Reshaped tensor of shape (batch_size, num_heads, seq_len, d_k)
        """

        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reverses the head split operation by concatenating all heads back.

        Args:
            x (Tensor): Tensor of shape (batch_size, num_heads, seq_len, d_k)

        Returns:
            Tensor: Combined tensor of shape (batch_size, seq_len, d_model)
        """

        x = x.transpose(1,2).contiguous()
        batch_size, seq_len, _, _ = x.size()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(self, x_q: torch.Tensor, x_k: torch.Tensor,
                x_v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Computes multi-head scaled dot-product attention.

        Args:
            x_q (Tensor): Query tensor of shape (batch_size, seq_len, d_model)
            x_k (Tensor): Key tensor of shape (batch_size, seq_len, d_model)
            x_v (Tensor): Value tensor of shape (batch_size, seq_len, d_model)
            mask (Tensor, optional): Attention mask of shape (batch_size, 1, seq_len, seq_len),
                                        where masked positions contain 0.

        Returns:
            Tensor: Output of shape (batch_size, seq_len, d_model)
        """

        Q = self._split_heads(self.w_q(x_q))
        K = self._split_heads(self.w_k(x_k))
        V = self._split_heads(self.w_v(x_v))

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        context = torch.matmul(attn, V)
        return self._combine_heads(context)

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network used inside Transformer layers.

    This block processes each token embedding independently using two linear
    layers with a non-linear activation in between. It expands the embedding
    dimensionality and then projects it back, helping the model learn richer
    representations.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Constructor for the FeedForward network.

        Parameters
            d_model : int
                Dimensionality of the input and output embeddings.
            d_ff : int
                Dimensionality of the hidden layer, typically larger than d_model.
            dropout : float, optional (default = 0.1)
                Dropout probability applied after activation and output.
        """

        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Feed-Forward layer.

        Parameters
            x : torch.Tensor
                Input tensor of shape (batch_size, seq_len, d_model)

        Returns
            torch.Tensor
        Output tensor of shape (batch_size, seq_len, d_model)
        """

        x = F.gelu(self.fc1(x))
        x = self.act_dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder layer.

    Consists of two main sublayers:
    1) Multi-head self-attention
    2) Position-wise feed-forward network

    Each sublayer is followed by:
    - Residual connection (x + sublayer_output)
    - Layer normalization

    This design allows the model to learn contextual relationships
    between tokens and then transform token representations individually.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Constructor for EncoderLayer.

        Parameters
            d_model : int
                Dimensionality of token embeddings.
            n_heads : int
                Number of attention heads in self-attention.
            d_ff : int
                Dimensionality of the hidden layer in the feed-forward block.
            dropout : float, optional (default = 0.1)
                Dropout probability used in attention and feed-forward.
        """

        super().__init__()
        self.attn = MultiHeadAttentionCustom(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the encoder layer.

        Parameters
            x : torch.Tensor
                Input embeddings, shape (batch_size, seq_len, d_model).
                mask : torch.Tensor, optional
        Attention mask that prevents attending to certain positions.

        Returns
            torch.Tensor
        Output tensor of shape (batch_size, seq_len, d_model).
        """

        x = self.norm1(x + self.attn(x, x, x, mask))
        x = self.norm2(x + self.ff(x))
        return x

class DecoderLayer(nn.Module):
    """
    Single layer of the Transformer decoder.
    Combines self-attention, cross-attention, and feed-forward sublayers with residual connections and layer normalization.

    Methods:
        __init__: constructor, initializes sublayers
        forward: computes forward pass for one decoder layer
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Constructor
        :param d_model: dimension of input embeddings
        :param n_heads: number of attention heads
        :param d_ff: hidden size of feed-forward network
        :param dropout: dropout probability
        """

        super().__init__()
        self.self_attn = MultiHeadAttentionCustom(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttentionCustom(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor,
                src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of one decoder layer
        :param x: input tensor (batch_size, seq_len, d_model)
        :param enc_out: encoder outputs (batch_size, src_seq_len, d_model)
        :param src_mask: optional mask for encoder output (padding mask)
        :param tgt_mask: optional mask for decoder self-attention (causal mask)
        :return: output tensor of the same shape as input (batch_size, seq_len, d_model)
        """

        x = self.norm1(x + self.self_attn(x, x, x, tgt_mask))
        x = self.norm2(x + self.cross_attn(x, enc_out, enc_out, src_mask))
        x = self.norm3(x + self.ff(x))
        return x

class Encoder(nn.Module):
    """
    Transformer Encoder
    methods: constructor (__init__), forward
    """

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, d_ff: int,
                 num_layers: int, max_len: int = 512, dropout: float =0.1):

        """
        Constructor
        :param vocab_size: size of the vocabulary
        :param d_model: dimension of embeddings and model hidden size
        :param n_heads: number of attention heads
        :param d_ff: dimension of feed-forward network
        :param num_layers: number of encoder layers
        :param max_len: maximum sequence length
        :param dropout: dropout probability
        :return: None
        """

        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])
        nn.init.xavier_normal_(self.embed.weight)

    def forward(self, x: torch.Tensor, lang_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the encoder
        :param x: input token IDs, shape (batch_size, seq_len)
        :param lang_ids: optional language IDs for multilingual models, shape (batch_size, seq_len)
        :return: encoder outputs, shape (batch_size, seq_len, d_model)
        """

        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x_embed = self.embed(x) + self.pos_embed(pos)
        if lang_ids is not None:
            x_embed = x_embed + self.embed(lang_ids)
        x = self.dropout(x_embed)
        for layer in self.layers:
            x = layer(x)
        return x

class Decoder(nn.Module):
    """
    Transformer Decoder class
    methods: constructor, forward
    """

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, d_ff: int,
                 num_layers: int, max_len: int = 512, dropout: float = 0.1):
        """
        Constructor
        :param vocab_size: size of the vocabulary
        :param d_model: hidden size of the model
        :param n_heads: number of attention heads
        :param d_ff: size of feed-forward layer
        :param num_layers: number of decoder layers
        :param max_len: maximum sequence length
        :param dropout: dropout probability
        """

        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.xavier_normal_(self.embed.weight)
        nn.init.xavier_normal_(self.fc_out.weight)

    def forward(self, y: torch.Tensor, enc_out: torch.Tensor,
                tgt_lang_ids: torch.Tensor = None, teacher_forcing: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the decoder
        :param y: target tokens, shape (batch_size, tgt_seq_len)
        :param enc_out: encoder output, shape (batch_size, src_seq_len, d_model)
        :param tgt_lang_ids: optional target language embeddings, shape (batch_size, tgt_seq_len)
        :param teacher_forcing: optional tensor for teacher-forcing input, shape (batch_size, tgt_seq_len, d_model)
        :return: logits over vocabulary, shape (batch_size, tgt_seq_len, vocab_size)
        """

        pos = torch.arange(y.size(1), device=y.device).unsqueeze(0)
        y_embed = self.embed(y) + self.pos_embed(pos)
        if tgt_lang_ids is not None:
            y_embed = y_embed + self.embed(tgt_lang_ids)
        x = self.dropout(y_embed)

        tgt_mask = generate_square_subsequent_mask(x.size(1)).to(x.device)
        if teacher_forcing is not None:
            x = teacher_forcing

        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask=tgt_mask)
        return self.fc_out(x)

class BART(nn.Module):
    """
    BART (Bidirectional and Auto-Regressive Transformers) model.
    Combines an encoder and a decoder for sequence-to-sequence tasks like translation.

    methods:
        __init__          : constructor
        forward           : forward pass for training with optional teacher forcing
        generate          : autoregressive generation with beam search
    """

    def __init__(self, vocab_size: int, d_model: int = 256, n_heads: int = 8, d_ff: int = 512,
                 num_layers: int = 4, max_len: int = 512):
        """
        Constructor
        :param vocab_size: size of the vocabulary
        :param d_model: embedding dimension
        :param n_heads: number of attention heads
        :param d_ff: feed-forward network dimension
        :param num_layers: number of encoder/decoder layers
        :param max_len: maximum sequence length
        """

        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, n_heads, d_ff, num_layers, max_len)
        self.decoder = Decoder(vocab_size, d_model, n_heads, d_ff, num_layers, max_len)
        self.decoder.fc_out.weight = self.decoder.embed.weight

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_lang_ids: torch.Tensor = None,
                tgt_lang_ids: torch.Tensor = None, teacher_forcing: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for training
        :param src: source sequence tensor (batch_size, src_seq_len)
        :param tgt: target sequence tensor (batch_size, tgt_seq_len)
        :param src_lang_ids: optional language ids for source (batch_size, src_seq_len)
        :param tgt_lang_ids: optional language ids for target (batch_size, tgt_seq_len)
        :param teacher_forcing: optional tensor to force decoder inputs (batch_size, tgt_seq_len, d_model)
        :return: logits tensor (batch_size, tgt_seq_len, vocab_size)
        """

        src_masked = apply_span_masking(src)
        enc_out = self.encoder(src_masked, lang_ids=src_lang_ids)
        return self.decoder(tgt, enc_out, tgt_lang_ids=tgt_lang_ids, teacher_forcing=teacher_forcing)

    def generate(self, src: torch.Tensor, src_lang_ids: torch.Tensor = None,
                 max_len: int = 20, start_token: int = 1, beam_size: int = 3) -> torch.Tensor:
        """
        Autoregressive sequence generation using beam search
        :param src: source sequence tensor (batch_size, src_seq_len)
        :param src_lang_ids: optional language ids for source
        :param max_len: maximum length of generated sequence
        :param start_token: token id to start decoding
        :param beam_size: number of beams in beam search
        :return: tensor of generated sequences (batch_size, max_len)
        """

        enc_out = self.encoder(src, lang_ids=src_lang_ids)
        batch_size = src.size(0)
        sequences = [[(torch.tensor([start_token], device=src.device), 0.0)] for _ in range(batch_size)]

        for _ in range(max_len):
            all_candidates = [[] for _ in range(batch_size)]
            for i in range(batch_size):
                for seq, score in sequences[i]:
                    out = self.decoder(seq.unsqueeze(0), enc_out[i:i+1])
                    log_probs = F.log_softmax(out[:, -1, :], dim=-1).squeeze(0)
                    topk_probs, topk_idx = log_probs.topk(beam_size)
                    for k in range(beam_size):
                        candidate_seq = torch.cat([seq, topk_idx[k].unsqueeze(0)])
                        candidate_score = score + topk_probs[k].item()
                        all_candidates[i].append((candidate_seq, candidate_score))
                all_candidates[i] = sorted(all_candidates[i], key=lambda x: x[1], reverse=True)[:beam_size]
            sequences = all_candidates

        best_sequences = torch.stack([seqs[0][0] for seqs in sequences])
        return best_sequences

vocab_size = 50
seq_len = 10
num_samples = 100
batch_size = 8

src_data = torch.randint(1, vocab_size, (num_samples, seq_len))
tgt_data = src_data.clone()

dataset = TensorDataset(src_data, tgt_data)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = BART(vocab_size=vocab_size, d_model=128, n_heads=4, d_ff=256, num_layers=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=0)

model.train()
for epoch in range(3):
    total_loss = 0
    for src, tgt in loader:
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])
        loss = criterion(output.view(-1, vocab_size), tgt[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

model.eval()
smooth_fn = SmoothingFunction().method1

bleu_scores = []
with torch.no_grad():
    for src, tgt in loader:
        gen = model.generate(src, max_len=seq_len, start_token=1, beam_size=3)
        for i in range(src.size(0)):
            reference = tgt[i].tolist()
            candidate = gen[i].tolist()
            bleu = sentence_bleu([reference], candidate, smoothing_function=smooth_fn)
            bleu_scores.append(bleu)

print(f"Average BLEU score: {sum(bleu_scores)/len(bleu_scores):.4f}")