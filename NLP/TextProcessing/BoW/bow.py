from collections import Counter
from typing import List

from sklearn.feature_extraction.text import CountVectorizer

docs = [
    "I love NLP",
    "I love ML"
]

vectorizer = CountVectorizer()
X_sklearn = vectorizer.fit_transform(docs)
sk_vocab = vectorizer.get_feature_names_out()

for doc, vec in zip(docs, X_sklearn.toarray()):
    print(f"Document: '{doc}'")
    print("Vector:  ", vec)
    print()

def build_vocab(docs: List[str]) -> set:
    """
    Build a vocabulary set from a collection of documents for Bag-of-Words.

    This function:
    - Splits each document into words using whitespace.
    - Converts words to lowercase to ensure case-insensitive matching.
    - Removes tokens of length <= 1 (to ignore very short/non-informative tokens).
    - Collects unique words across all documents.
    - Sorts them to ensure a stable and reproducible feature order.

    Parameters
        docs : List[str]
            A list of input documents.

    Returns
        set
            A sorted collection of unique vocabulary terms.
            (Used as the feature space for BoW vectors.)
    """

    vocab = sorted(
        set(word.lower()
            for d in docs
            for word in d.split()
            if len(word) > 1)
    )
    return vocab


def bow_vector(doc: str, vocab: set) -> List[int]:
    """
    Convert a document into a Bag-of-Words (BoW) vector.

    The BoW representation counts how many times each vocabulary word
    appears in the document. No weighting or normalization is applied —
    this is a pure frequency-based representation.

    Steps:
    - Tokenize the document (lowercase + remove short tokens).
    - Count word occurrences using `Counter`.
    - Produce a vector aligned with `vocab`, where each position
        corresponds to the frequency of that vocabulary word.

    Parameters
        doc : str
            The input document to vectorize.
        vocab : set
            The global vocabulary defining vector dimensions and order.

    Returns
        List[int]
            A list of word counts corresponding to each vocabulary term.
    """

    words = [w.lower() for w in doc.split() if len(w) > 1]
    cnt = Counter(words)
    return [cnt[w] for w in vocab]

manual_vocab = build_vocab(docs)

for doc in docs:
    vec = bow_vector(doc, manual_vocab)
    print(f"Document: '{doc}'")
    print("Vector:  ", vec)
    print()
