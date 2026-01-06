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
    """
    Bag of Words (BoW) for text vectorization.

    This class converts a list of text documents into a numerical feature matrix
    where each column corresponds to a unique word in the corpus and each row
    corresponds to a document. The value in each cell represents the frequency
    of the word in the document.
    """

    def __init__(self):
        """
        Constructor of the CustomBoW instance.

        Attributes:
            vocab (dict): A dictionary mapping each unique word to a column index.
        """

        self.vocab = {}

    def fit(self, corpus: List[str]) -> None:
        """
        Build the vocabulary from the given corpus.

        Each unique word in the corpus is assigned a unique index in the vocabulary.

        Args:
            corpus (List[str]): List of text documents to learn the vocabulary from.

        Returns:
            None
        """

        all_words = set()
        for doc in corpus:
            words = doc.lower().split()
            all_words.update(words)
        self.vocab = {word: idx for idx, word in enumerate(sorted(all_words))}

    def transform(self, corpus: List[str]) -> np.ndarray:
        """
        Transform the given corpus into a numerical feature matrix using the learned vocabulary.

        Args:
            corpus (List[str]): List of text documents to convert.

        Returns:
            np.ndarray: Feature matrix of shape (n_documents, n_unique_words),
                        where each cell contains the frequency of the corresponding word in the document.
        """

        X = np.zeros((len(corpus), len(self.vocab)), dtype=int)
        for i, doc in enumerate(corpus):
            for word in doc.lower().split():
                if word in self.vocab:
                    X[i, self.vocab[word]] += 1

        return X

    def fit_transform(self, corpus: List[str]) -> np.ndarray:
        """
        Fit the vocabulary and transform the corpus into a feature matrix in one step.

        Args:
            corpus (List[str]): List of text documents to fit and transform.

        Returns:
            np.ndarray: Feature matrix of shape (n_documents, n_unique_words).
        """

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