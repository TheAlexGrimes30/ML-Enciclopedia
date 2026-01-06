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
    """
    Custom implementation of TF-IDF (Term Frequency - Inverse Document Frequency).

    Attributes
        vocab : dict
            Dictionary mapping each unique word to a column index in the feature matrix.
        idf : np.ndarray
            Array of IDF (inverse document frequency) values for each word in the vocabulary.
    """

    def __init__(self):
        """
        Constructor of the TF-IDF vectorizer with empty vocabulary and IDF values.
        """

        self.vocab = {}
        self.idf = None

    def fit(self, corpus: List[str]) -> None:
        """
        Learn the vocabulary and compute IDF values from the given corpus.

        Parameters
            corpus : List[str]
                A list of documents (strings) to fit the TF-IDF model on.

        Steps
            1. Build a vocabulary of all unique words across the corpus.
            2. Count document frequency (DF) for each word.
            3. Compute IDF using smoothed formula: log((N + 1) / (DF + 1)) + 1
        """

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
        """
        Transform documents into TF-IDF feature matrix.

        Parameters
            corpus : List[str]
                A list of documents (strings) to transform.

        Returns
            X : np.ndarray
                2D array of shape (n_documents, n_features) where each row represents
                a document and each column represents the TF-IDF weight of a word.

        Steps
            1. Compute term frequency (TF) for each word in each document.
            2. Normalize TF by dividing by the total number of words in the document.
            3. Multiply TF by precomputed IDF values to get TF-IDF.
        """

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
        """
        Fit the TF-IDF model on the corpus and transform the documents.

        Parameters
            corpus : List[str]
                A list of documents (strings) to fit and transform.

        Returns
            X : np.ndarray
                TF-IDF feature matrix.
        """

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
