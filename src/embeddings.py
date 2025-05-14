import unicodedata
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

class TFIDFEmbedding:
    def __init__(self, corpus):
        """
        corpus: list of sentences (list of strings)
        """
        self.corpus = [self._normalize(text) for text in corpus]
        self.vectorizer = TfidfVectorizer(lowercase=True)
        self.vectorizer.fit(self.corpus)

    def _normalize(self, text):
        # Normalize Unicode and convert to lowercase
        text = unicodedata.normalize('NFKC', text)
        return text.lower()

    def embed(self, text) -> np.ndarray:
        """
        Return the TF-IDF vector of a sentence (as numpy.ndarray)
        """
        text = self._normalize(text)
        return self.vectorizer.transform([text]).toarray()

    def embed_corpus(self) -> np.ndarray:
        """
        Return the TF-IDF matrix for the entire corpus (as numpy.ndarray)
        """
        return self.vectorizer.transform(self.corpus).toarray()

    def export(self, path):
        """
        Save the vectorizer and corpus to file
        """
        joblib.dump({'vectorizer': self.vectorizer, 'corpus': self.corpus}, path)

    @classmethod
    def load(cls, path):
        """
        Load the vectorizer and corpus from file, return TFIDFEmbedding instance
        """
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj.vectorizer = data['vectorizer']
        obj.corpus = data['corpus']
        return obj
