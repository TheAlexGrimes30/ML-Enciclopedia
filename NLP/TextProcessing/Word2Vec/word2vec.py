from typing import List, Tuple, Callable

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
    """
    Split a sentence into lowercase tokens using whitespace.

    Args:
        text (str): Input sentence.

    Returns:
        List[str]: List of normalized tokens.
    """

    return text.lower().split()

sentences = [tokenize(d) for d in docs]
vocab = sorted(set(w for s in sentences for w in s))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(vocab)
window_size = 2

def generate_skipgram_data(sentences: List[List[str]], window: int) -> List[Tuple[int, int]]:
    """
    Generate (center, context) index pairs for Skip-Gram training.

    For each word in a sentence, this function collects surrounding
    words within the specified window as context words.

    Args:
        sentences (List[List[str]]): Tokenized corpus.
        window (int): Context window size.

    Returns:
        List[Tuple[int, int]]: List of (center_word_idx, context_word_idx) pairs.
    """

    pairs = []
    for sent in sentences:
        for i, center in enumerate(sent):
            for j in range(max(0, i - window), min(len(sent), i + window + 1)):
                if i != j:
                    pairs.append((word2idx[center], word2idx[sent[j]]))
    return pairs

def generate_cbow_data(sentences: List[List[str]], window: int) -> List[Tuple[List[int], int]]:
    """
    Generate (context, target) samples for CBOW training.

    The context consists of surrounding words within the given window,
    and the target is the current center word.

    Args:
        sentences (List[List[str]]): Tokenized corpus.
        window (int): Context window size.

    Returns:
        List[Tuple[List[int], int]]: List of (context_indices, target_index).
    """

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
    """
    Minimal Skip-Gram implementation using two embedding matrices.
    Predicts context words given a center word via dot-product similarity.
    """

    def __init__(self, vocab_size: int, emb_dim: int):
        super().__init__()
        self.in_emb = nn.Embedding(vocab_size, emb_dim)
        self.out_emb = nn.Embedding(vocab_size, emb_dim)

    def forward(self, center: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Compute Skip-Gram loss for a (center, context) pair.

        Args:
            center (Tensor): Index of the center word.
            context (Tensor): Index of the context word.

        Returns:
            Tensor: Scalar loss value.
        """

        center_vec = self.in_emb(center)
        context_vec = self.out_emb(context)

        score = torch.sum(center_vec * context_vec, dim=1)
        loss = -torch.log(torch.sigmoid(score))
        return loss.mean()

class CBOW(nn.Module):
    """
    Continuous Bag-of-Words model.
    Predicts the target word from the mean of context embeddings.
    """

    def __init__(self, vocab_size: int, emb_dim: int):
        super().__init__()
        self.in_emb = nn.Embedding(vocab_size, emb_dim)
        self.linear = nn.Linear(emb_dim, vocab_size)

    def forward(self, context_idxs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of CBOW model.

        Args:
            context_idxs (Tensor): Indices of context words.

        Returns:
            Tensor: Logits over the vocabulary.
        """

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
    """
    Retrieve a word embedding from the trained Skip-Gram model.
    """

    idx = torch.tensor([word2idx[word]])
    return sg_model.in_emb(idx).detach().numpy()[0]

def get_vector_cbow(word: str) -> np.ndarray:
    """
    Retrieve a word embedding from the trained CBOW model.
    """

    idx = torch.tensor([word2idx[word]])
    return cbow_model.in_emb(idx).detach().numpy()[0]

def doc_vector(model_func: Callable[[str], np.ndarray], doc: str) -> np.ndarray:
    """
    Compute a document embedding by averaging word embeddings.

    Args:
        model_func (Callable[[str], np.ndarray]):
            Function that takes a word and returns its embedding vector.
        doc (str): Input document.

    Returns:
        np.ndarray: Mean vector representation of the document.
    """

    tokens = tokenize(doc)
    vectors = [model_func(w) for w in tokens]
    return np.mean(vectors, axis=0)

def get_vector_gensim_sg(word: str) -> np.ndarray:
    """
    Retrieve a word embedding from gensim Skip-Gram Word2Vec.
    """

    return w2v_model.wv[word]

def get_vector_gensim_cbow(word: str) -> np.ndarray:
    """
    Retrieve a word embedding from gensim CBOW Word2Vec.
    """

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
