from typing import List, Counter, Any

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
    vocab = sorted(
        set(word.lower()
            for d in docs
            for word in d.split()
            if len(word) > 1)
    )
    return vocab


def bow_vector(doc: str, vocab: set) -> List[int]:
    words = [w.lower() for w in doc.split() if len(w) > 1]
    cnt = Counter(words)
    return [cnt[w] for w in vocab]

manual_vocab = build_vocab(docs)

for doc in docs:
    vec = bow_vector(doc, manual_vocab)
    print(f"Document: '{doc}'")
    print("Vector:  ", vec)
    print()
