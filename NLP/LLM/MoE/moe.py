import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import SwitchTransformersForConditionalGeneration


def make_causal_mask(sz: int, device: torch.device) -> torch.Tensor:
    m = torch.tril(torch.ones(sz, sz, dtype=torch.bool, device=device))
    return m.unsqueeze(0).unsqueeze(0)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
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
        b = x.size(0)
        return x.view(b, seq_len, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = F.gelu(self.w1(x))
        x = self.dropout(x)
        return self.dropout(self.w2(x))

class MoEFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int, dropout: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([
            FeedForward(d_model, d_ff, dropout)
            for _ in range(num_experts)
        ])

        self.register_buffer("expert_counts", torch.zeros(num_experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d =  x.shape
        x_flat = x.view(-1, d)

        gate_probs = F.softmax(self.gate(x_flat), dim=-1)
        expert_ids = torch.argmax(gate_probs, dim=-1)

        with torch.no_grad():
            for i in range(self.num_experts):
                self.expert_counts[i] += (expert_ids == i).sum()

        out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = expert_ids == i
            if mask.any():
                out[mask] = expert(x_flat[mask])

        return out.view(b, s, d)

class GPTMoEBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, num_experts: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.moe = MoEFeedForward(d_model, d_ff, num_experts)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.moe(self.ln2(x))
        return x

class GPTMoE(nn.Module):
    def __init__(self, vocab: int,
                 d_model: int = 128, n_heads: int = 4,
                 d_ff: int = 256, layers: int = 3,
                 experts: int = 4, pos_vocab: int = 512):
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
        b, s = x.shape
        pos = torch.arange(s, device=x.device).unsqueeze(0)
        x = self.emb(x) + self.pos_emb(x)
        mask = make_causal_mask(s, x.device)

        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.ln(x))

def evaluate(model: nn.Module, loader: DataLoader, vocab: int, device: torch.device) -> Tuple[float, float, float]:
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

