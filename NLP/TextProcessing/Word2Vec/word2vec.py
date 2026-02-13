from typing import List, Tuple

import numpy as np
import torch
from gensim.models import Word2Vec
from torch import nn, optim
from tqdm import tqdm

docs = [
    "I love NLP",
    "I love ML"
]

def tokenize(text: str) -> List[str]:
    return text.lower().split()

sentences = [tokenize(d) for d in docs]
vocab = sorted(set(w for s in sentences for w in s))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(vocab)
window_size = 2

def generate_skipgram_data(sentences: List[List[str]], window: int) -> List[Tuple[int, int]]:
    pairs = []
    for sent in sentences:
        for i, center in enumerate(sent):
            for j in range(max(0, i - window), min(len(sent), i + window + 1)):
                if i != j:
                    pairs.append((word2idx[center], word2idx[sent[j]]))
    return pairs

def generate_cbow_data(sentences: List[List[str]], window: int) -> List[Tuple[List[int], int]]:
    data = []
    for sent in sentences:
        for i, target in enumerate(sent):
            context = []
            for j in range(max(0, i - window), min(len(sent), i + window + 1)):
                if i != j:
                    context.append(word2idx[sent[j]])
            if context:
                data.append((context, word2idx[target]))
    return data

class SkipGram(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int):
        super().__init__()
        self.in_emb = nn.Embedding(vocab_size, emb_dim)
        self.out_emb = nn.Embedding(vocab_size, emb_dim)

    def forward(self, center: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        center_vec = self.in_emb(center)
        context_vec = self.out_emb(context)

        score = torch.sum(center_vec * context_vec, dim=1)
        loss = -torch.log(torch.sigmoid(score))
        return loss.mean()

class CBOW(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int):
        super().__init__()
        self.in_emb = nn.Embedding(vocab_size, emb_dim)
        self.linear = nn.Linear(emb_dim, vocab_size)

    def forward(self, context_idxs: torch.Tensor) -> torch.Tensor:
        emb = self.in_emb(context_idxs)
        mean_emb = emb.mean(dim=0)
        out = self.linear(mean_emb)
        return out


skipgram_pairs = generate_skipgram_data(sentences, window_size)
cbow_data = generate_cbow_data(sentences, window_size)

embedding_dim = 20
sg_model = SkipGram(vocab_size, embedding_dim)
optimizer = optim.Adam(sg_model.parameters(), lr=0.01)

for epoch in tqdm(range(300), desc="SkipGram Epochs"):
    total_loss = 0
    for center, context in tqdm(skipgram_pairs, leave=False, desc="Training"):
        center = torch.tensor([center])
        context = torch.tensor([context])

        loss = sg_model(center, context)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

cbow_model = CBOW(vocab_size, embedding_dim)
optimizer = optim.Adam(cbow_model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in tqdm(range(300), desc="CBOW Epochs"):
    total_loss = 0

    for context, target in tqdm(cbow_data, leave=False, desc="Training"):
        context = torch.tensor(context)
        target = torch.tensor([target])

        output = cbow_model(context)
        loss = criterion(output.unsqueeze(0), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

def get_vector_skipgram(word: str) -> np.ndarray:
    idx = torch.tensor([word2idx[word]])
    return sg_model.in_emb(idx).detach().numpy()[0]

def get_vector_cbow(word: str) -> np.ndarray:
    idx = torch.tensor([word2idx[word]])
    return cbow_model.in_emb(idx).detach().numpy()[0]

def doc_vector(model_func, doc):
    tokens = tokenize(doc)
    vectors = [model_func(w) for w in tokens]
    return np.mean(vectors, axis=0)

def get_vector_gensim_sg(word: str) -> np.ndarray:
    return w2v_model.wv[word]

def get_vector_gensim_cbow(word: str) -> np.ndarray:
    return w2v_cbow_model.wv[word]

for word in vocab:
    print(f"\nWord: {word}")
    print(" SkipGram:", get_vector_skipgram(word)[:5])
    print(" CBOW:    ", get_vector_cbow(word)[:5])

for d in docs:
    print(f"\nDocument: {d}")
    print(" SkipGram:", doc_vector(get_vector_skipgram, d)[:5])
    print(" CBOW:    ", doc_vector(get_vector_cbow, d)[:5])

w2v_model = Word2Vec(
    sentences=sentences,
    vector_size=embedding_dim,
    window=window_size,
    min_count=1,
    sg=1,
    epochs=300
)

w2v_cbow_model = Word2Vec(
    sentences=sentences,
    vector_size=embedding_dim,
    window=window_size,
    min_count=1,
    sg=0,
    epochs=300
)

for word in vocab:
    print(f"\nWord: {word}")
    print(" SkipGram (PyTorch):", get_vector_skipgram(word)[:5])
    print(" CBOW (PyTorch):    ", get_vector_cbow(word)[:5])
    print(" Gensim SG:         ", get_vector_gensim_sg(word)[:5])
    print(" Gensim CBOW:       ", get_vector_gensim_cbow(word)[:5])

for d in docs:
    print(f"\nDocument: {d}")
    print(" SkipGram (PyTorch):", doc_vector(get_vector_skipgram, d)[:5])
    print(" CBOW (PyTorch):    ", doc_vector(get_vector_cbow, d)[:5])
    print(" Gensim SG:         ", doc_vector(get_vector_gensim_sg, d)[:5])
    print(" Gensim CBOW:       ", doc_vector(get_vector_gensim_cbow, d)[:5])