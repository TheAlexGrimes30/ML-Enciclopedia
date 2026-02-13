import math
from collections import Counter
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer

docs = [
    "I love NLP",
    "I love ML"
]

def tokenize(text: str) -> List[str]:
    """
    Tokenize input text into normalized words.

    This function:
    - Splits the text by whitespace.
    - Converts all tokens to lowercase.
    - Removes tokens of length <= 1 (to mimic sklearn's default token pattern).

    Parameters
        text : str
            Input document as a string.

    Returns
        List[str]
            A list of cleaned tokens.
    """

    return [w.lower() for w in text.split() if len(w) > 1]

def build_vocab(docs: List[str]) -> list[str]:
    """
    Build a sorted vocabulary from a collection of documents.

    The vocabulary consists of all unique tokens extracted using `tokenize`.
    The result is sorted to ensure deterministic vector ordering.

    Parameters
        docs : List[str]
            A list of input documents.

    Returns
        List[str]
            Sorted list of unique words across the corpus.
    """

    vocab = sorted(set(w for d in docs for w in tokenize(d)))
    return vocab

def tfidf_vector(doc: str, docs: List[str], vocab: List[str]) -> List[float]:
    """
    Compute the TF-IDF vector for a single document.

    This implementation mirrors sklearn's behavior:
    - Uses term frequency normalized by document length.
    - Applies *smoothed IDF*:
        idf = log((1 + N) / (1 + df)) + 1
    - Performs L2 normalization so the resulting vector has unit length.

    Parameters
        doc : str
            The document to transform into a TF-IDF vector.
        docs : List[str]
            The full corpus (used to compute document frequencies).
        vocab : List[str]
            The global vocabulary defining vector dimensions.

    Returns
        List[float]
            L2-normalized TF-IDF vector aligned with `vocab`.
    """

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
