import math
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from NLP.LLM.BERT.bert import BERTClassifier


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

class Retriever:
    """
    Retriever for RAG systems.

    Uses precomputed document embeddings to find the most relevant context
    for each query based on cosine similarity.
    """

    def __init__(self, docs: List[torch.Tensor], doc_embeddings: torch.Tensor):
        """
        Initializes the retriever.

        Args:
            docs (List[torch.Tensor]): List of documents as token tensors.
            doc_embeddings (torch.Tensor): Precomputed embeddings for each document.
                Shape: (num_docs, embedding_dim)
        """

        self.docs = docs
        self.doc_embeddings = F.normalize(doc_embeddings, dim=-1)

    def retrieve(self, query_embeddings: torch.Tensor, top_k: int = 1) -> List[List[torch.Tensor]]:
        """
        Retrieve top-k most relevant documents for each query.

        Args:
            query_embeddings (torch.Tensor): Query embeddings, shape (batch_size, embedding_dim)
            top_k (int): Number of top documents to retrieve per query

        Returns:
            List[List[torch.Tensor]]: List of lists of retrieved document tensors
                                        for each query in the batch
        """

        query_embeddings = F.normalize(query_embeddings, dim=-1)
        sims = torch.matmul(query_embeddings, self.doc_embeddings.T)
        _, topk_idx = torch.topk(sims, top_k, dim=-1)

        retrieved_docs = []
        for indices in topk_idx:
            retrieved_docs.append([self.docs[i.item()] for i in indices])
        return retrieved_docs

class RAGGenerator(nn.Module):
    """
    Retrieval-Augmented Generation (RAG) model.

    Combines a Retriever with a generative model (GPT-like) to produce text
    conditioned on retrieved documents.
    """

    def __init__(self, gpt_model: nn.Module, retriever: Retriever, embedding_model: nn.Module):
        """
        Args:
            gpt_model: Generative model (GPT) that outputs token logits.
            retriever: Retriever instance that returns relevant document tensors.
            embedding_model: Model to embed queries for retrieval (e.g., small BERT encoder).
        """

        super().__init__()
        self.gpt = gpt_model
        self.retriever = retriever
        self.embed_model = embedding_model

    def forward(self, queries: torch.Tensor, max_len: int = 50, top_k_docs: int = 1) -> torch.Tensor:
        """
        Generate text for a batch of queries using retrieved documents.

        Args:
            queries: Input queries as token IDs, shape (batch_size, seq_len)
            max_len: Maximum length of generated sequences
            top_k_docs: Number of documents to retrieve per query

        Returns:
            generated_sequences: Tensor of generated token IDs, shape (batch_size, generated_len)
        """

        batch_size = queries.size(0)
        device = queries.device

        with torch.no_grad():
            hidden_states = self.embed_model.bert(queries)
            query_embeddings = hidden_states[:, 0, :]

        retrieved_docs = self.retriever.retrieve(query_embeddings, top_k=top_k_docs)

        input_sequences = []
        for i in range(batch_size):
            context_tokens = torch.cat(retrieved_docs[i], dim=0)
            input_seq = torch.cat([queries[i], context_tokens], dim=0)
            input_sequences.append(input_seq)

        max_input_len = max(seq.size(0) for seq in input_sequences)
        padded_inputs = torch.zeros(batch_size, max_input_len, dtype=torch.long, device=device)
        for i, seq in enumerate(input_sequences):
            padded_inputs[i, :seq.size(0)] = seq

        generated = self.gpt.generate(padded_inputs, max_len=max_len)
        return generated

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_size = 100
    num_classes = 3
    seq_len = 12
    num_docs = 50
    embedding_dim = 128

    docs = [torch.randint(1, vocab_size, (seq_len,), device=device) for _ in range(num_docs)]
    doc_tensors = torch.stack(docs)

    bert_model = BERTClassifier(vocab_size=vocab_size, num_classes=num_classes, max_seq_len=seq_len).to(device)
    bert_model.eval()

    with torch.no_grad():
        doc_hidden_states = bert_model.bert(doc_tensors)
        doc_embeddings = doc_hidden_states[:, 0, :]

    retriever = Retriever(docs, doc_embeddings)

    gpt_model = GPT(
        vocab_size=vocab_size,
        d_model=embedding_dim,
        n_heads=4,
        d_ff=256,
        num_layers=2
    ).to(device)

    rag = RAGGenerator(gpt_model, retriever, bert_model).to(device)

    queries = torch.randint(1, vocab_size, (3, seq_len), device=device)

    generated_tokens = rag(queries, max_len=20, top_k_docs=2)
    
    print("Generated tokens:")
    print(generated_tokens)