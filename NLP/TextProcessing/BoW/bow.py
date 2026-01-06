from typing import List

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "I love machine learning",
    "Machine learning is amazing",
    "I hate spam emails",
    "Spam is annoying",
    "I love deep learning",
    "Deep learning is fun",
    "Spam emails are terrible",
]

labels = [1, 1, 0, 0, 1, 1, 0]

class CustomBoW:
    def __init__(self):
        self.vocab = {}

    def fit(self, corpus: List[str]) -> None:
        all_words = set()
        for doc in corpus:
            words = doc.lower().split()
            all_words.update(words)
        self.vocab = {word: idx for idx, word in enumerate(sorted(all_words))}

    def transform(self, corpus: List[str]) -> np.ndarray:
        X = np.zeros((len(corpus), len(self.vocab)), dtype=int)
        for i, doc in enumerate(corpus):
            for word in doc.lower().split():
                if word in self.vocab:
                    X[i, self.vocab[word]] += 1

        return X

    def fit_transform(self, corpus: List[str]) -> np.ndarray:
        self.fit(corpus)
        return self.transform(corpus)

bow = CustomBoW()
X_custom = bow.fit_transform(corpus)

vectorizer = CountVectorizer()
X_sklearn = vectorizer.fit_transform(corpus).toarray()

print("Custom BoW Vocabulary:")
print(bow.vocab)
print("\nCustom BoW Feature Matrix:")
print(X_custom)

print("\nSklearn CountVectorizer Vocabulary:")
print(vectorizer.vocabulary_)
print("\nSklearn Feature Matrix:")
print(X_sklearn)