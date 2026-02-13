import math
from collections import Counter
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer

docs = [
    "I love NLP",
    "I love ML"
]

def tokenize(text: str) -> List[str]:
    return [w.lower() for w in text.split() if len(w) > 1]

def build_vocab(docs: List[str]) -> list[str]:
    vocab = sorted(set(w for d in docs for w in tokenize(d)))
    return vocab

def tfidf_vector(doc: str, docs: List[str], vocab: List[str]) -> List[float]:
    tokens = tokenize(doc)
    tf = Counter(tokens)
    vec = []
    N = len(docs)

    for word in vocab:
        tf_val = tf[word] / len(tokens) if tokens else 0
        df = sum(1 for d in docs if word in tokenize(d))
        idf = math.log((1 + N) / (1 + df)) + 1

        vec.append(tf_val * idf)

    norm = math.sqrt(sum(v*v for v in vec))
    if norm:
        vec = [v / norm for v in vec]

    return vec


vectorizer = TfidfVectorizer()
X_sklearn = vectorizer.fit_transform(docs)

for doc, vec in zip(docs, X_sklearn.toarray()):
    print(f"Document: '{doc}'")
    print("Vector:  ", vec)
    print()

manual_vocab = build_vocab(docs)

for doc in docs:
    vec = tfidf_vector(doc, docs, manual_vocab)
    print(f"Document: '{doc}'")
    print("Vector:  ", vec)
    print()
