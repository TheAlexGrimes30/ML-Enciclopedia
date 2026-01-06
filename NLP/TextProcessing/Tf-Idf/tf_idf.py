from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "I love machine learning",
    "Machine learning is amazing",
    "I hate spam emails",
    "Spam is annoying",
    "I love deep learning",
    "Deep learning is fun",
    "Spam emails are terrible",
]

class CustomTfIdf:
    def __init__(self):
        self.vocab = {}
        self.idf = None

    def fit(self, corpus: List[str]) -> None:
        all_words = set(word.lower() for doc in corpus for word in doc.split())
        self.vocab = {word: idx for idx, word in enumerate(sorted(all_words))}
        n_docs = len(corpus)
        df = np.zeros(len(self.vocab))

        for doc in corpus:
            words_in_doc = set(word.lower() for word in doc.split())
            for word in words_in_doc:
                df[self.vocab[word]] += 1

        self.idf = np.log((n_docs + 1) / (df + 1)) + 1

    def transform(self, corpus: List[str]) -> np.ndarray:
        X = np.zeros((len(corpus), len(self.vocab)))
        for i, doc in enumerate(corpus):
            tf = np.zeros(len(self.vocab))
            words = doc.lower().split()
            for word in words:
                if word in self.vocab:
                    tf[self.vocab[word]] += 1

            if len(words) > 0:
                tf /= len(words)
            X[i] = tf * self.idf

        return X

    def fit_transform(self, corpus: List[str]) -> np.ndarray:
        self.fit(corpus)
        return self.transform(corpus)

custom_tfidf = CustomTfIdf()
X_custom = custom_tfidf.fit_transform(corpus)

print("Vocabulary:", custom_tfidf.vocab)
print("Custom TF-IDF Matrix:\n", np.round(X_custom, 3))

vectorizer = TfidfVectorizer()
X_sklearn = vectorizer.fit_transform(corpus).toarray()

print("\nSklearn Vocabulary:", vectorizer.vocabulary_)
print("Sklearn TF-IDF Matrix:\n", np.round(X_sklearn, 3))
