import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def generate_square_subsequent_mask(sz):
    return torch.triu(torch.ones(sz, sz), diagonal=1).bool()

def apply_span_masking(x, mask_ratio=0.15, mask_token_id=0, span_length=3):
    x_masked = x.clone()
    batch_size, seq_len = x.size()
    for i in range(batch_size):
        num_mask = max(1, int(seq_len * mask_ratio))
        mask_positions = torch.randperm(seq_len)[:num_mask]
        for pos in mask_positions:
            end = min(seq_len, pos + span_length)
            x_masked[i, pos:end] = mask_token_id
    return x_masked

class MultiHeadAttentionCustom(nn.Module):
    def __init__(self, d_model, num_heads=8, dropout=0.1):
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

    def _split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)

    def _combine_heads(self, x):
        x = x.transpose(1,2).contiguous()
        batch_size, seq_len, _, _ = x.size()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(self, x_q, x_k, x_v, mask=None):
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
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.act_dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttentionCustom(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.attn(x, x, x, mask))
        x = self.norm2(x + self.ff(x))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttentionCustom(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttentionCustom(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.norm1(x + self.self_attn(x, x, x, tgt_mask))
        x = self.norm2(x + self.cross_attn(x, enc_out, enc_out, src_mask))
        x = self.norm3(x + self.ff(x))
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, num_layers, max_len=512, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])
        nn.init.xavier_normal_(self.embed.weight)

    def forward(self, x, lang_ids=None):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x_embed = self.embed(x) + self.pos_embed(pos)
        if lang_ids is not None:
            x_embed = x_embed + self.embed(lang_ids)  # add language embedding
        x = self.dropout(x_embed)
        for layer in self.layers:
            x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, num_layers, max_len=512, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.xavier_normal_(self.embed.weight)
        nn.init.xavier_normal_(self.fc_out.weight)

    def forward(self, y, enc_out, tgt_lang_ids=None, teacher_forcing=None):
        pos = torch.arange(y.size(1), device=y.device).unsqueeze(0)
        y_embed = self.embed(y) + self.pos_embed(pos)
        if tgt_lang_ids is not None:
            y_embed = y_embed + self.embed(tgt_lang_ids)
        x = self.dropout(y_embed)

        tgt_mask = generate_square_subsequent_mask(x.size(1)).to(x.device)
        if teacher_forcing is not None:
            x = teacher_forcing  # replace decoder input with teacher-forced targets

        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask=tgt_mask)
        return self.fc_out(x)

class mBART(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, d_ff=512, num_layers=4, max_len=512):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, n_heads, d_ff, num_layers, max_len)
        self.decoder = Decoder(vocab_size, d_model, n_heads, d_ff, num_layers, max_len)
        self.decoder.fc_out.weight = self.decoder.embed.weight

    def forward(self, src, tgt, src_lang_ids=None, tgt_lang_ids=None, teacher_forcing=None):
        src_masked = apply_span_masking(src)
        enc_out = self.encoder(src_masked, lang_ids=src_lang_ids)
        return self.decoder(tgt, enc_out, tgt_lang_ids=tgt_lang_ids, teacher_forcing=teacher_forcing)

    def generate(self, src, src_lang_ids=None, max_len=20, start_token=1, beam_size=3):
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


model = mBART(vocab_size=vocab_size, d_model=128, n_heads=4, d_ff=256, num_layers=2)
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