import unicodedata
import os
import joblib
import math
import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
import re

class BaseEmbedding(ABC):
    @abstractmethod
    def embed(self, text) -> np.ndarray:
        pass

    @abstractmethod
    def embed_corpus(self) -> np.ndarray:
        pass

    @abstractmethod
    def query(self, text: str, sentence_set: list, k: int) -> list:
        pass

    @abstractmethod
    def export(self, path: str):
        pass


class TFIDFEmbedding(BaseEmbedding):
    def __init__(self, corpus_or_path):
        if isinstance(corpus_or_path, str) and os.path.exists(corpus_or_path):
            data = joblib.load(corpus_or_path)
            self.corpus = data['corpus']
            self.vocab = data['vocab']
            self.idf = data['idf']
        elif isinstance(corpus_or_path, list):
            self.corpus = [self._normalize(text) for text in corpus_or_path]
            self._build_vocab()
        else:
            raise ValueError("Input must be a list of strings (corpus) or a valid file path.")

    def _normalize(self, text):
        text = str(text)
        text = unicodedata.normalize('NFKC', text)
        return text.lower()

    def _tokenize(self, text):
        return text.split()

    def _build_vocab(self):
        df = defaultdict(int)
        vocab_set = set()
        tokenized_corpus = []

        for doc in self.corpus:
            tokens = set(self._tokenize(doc))
            tokenized_corpus.append(tokens)
            for token in tokens:
                df[token] += 1
                vocab_set.add(token)

        self.vocab = {word: idx for idx, word in enumerate(sorted(vocab_set))}
        N = len(self.corpus)
        self.idf = {word: math.log(N / (df[word])) for word in self.vocab}

    def _tfidf_vector(self, text):
        tokens = self._tokenize(self._normalize(text))
        tf = Counter(tokens)
        vec = np.zeros(len(self.vocab), dtype=np.float32)

        for token, count in tf.items():
            if token in self.vocab:
                tf_val = count / len(tokens)
                idf_val = self.idf.get(token, 0)
                vec[self.vocab[token]] = tf_val * idf_val
        return vec

    def embed(self, text):
        return self._tfidf_vector(text).reshape(1, -1)

    def embed_corpus(self):
        return np.vstack([self._tfidf_vector(text) for text in self.corpus])

    def query(self, text: str, sentence_set, k: int) -> list:
        sentences = [self._normalize(s) for s in sentence_set]
        sentence_matrix = np.vstack([self._tfidf_vector(s) for s in sentences])
        query_vec = self._tfidf_vector(text)
        similarities = sentence_matrix @ query_vec
        similarities = similarities.reshape(-1)

        exp_sim = np.exp(similarities - np.max(similarities))
        probs = exp_sim / np.sum(exp_sim)
        top_k_idx = np.argsort(probs)[-k:][::-1]

        return [sentence_set[i] for i in top_k_idx]

    def export(self, path):
        joblib.dump({'corpus': self.corpus, 'vocab': self.vocab, 'idf': self.idf}, path)


class CharNgramEmbedding(BaseEmbedding):
    def __init__(self, corpus_or_path, n=3):
        self.n = n
        if isinstance(corpus_or_path, str) and os.path.exists(corpus_or_path):
            self._load(corpus_or_path)
        elif isinstance(corpus_or_path, list):
            self.raw_corpus = corpus_or_path  # 🔹 Lưu bản gốc
            self.corpus = [self._normalize(s) for s in corpus_or_path]
            self.vocab = self._build_vocab(self.corpus)
            self.vocab_index = {ng: i for i, ng in enumerate(self.vocab)}
            self.corpus_embeddings = np.stack([self.embed(s) for s in self.corpus])
        else:
            raise ValueError("Input must be a list of strings or a valid path.")

    def _normalize(self, text):
        text = str(text).strip()
        text = re.sub(r'\s+', ' ', text) 
        return text

    def _char_ngrams(self, text):
        return [text[i:i+self.n] for i in range(len(text)-self.n+1)]

    def _build_vocab(self, corpus):
        vocab = set()
        for text in corpus:
            vocab.update(self._char_ngrams(text))
        return sorted(vocab)

    def embed(self, text) -> np.ndarray:
        text = self._normalize(text)
        ngrams = self._char_ngrams(text)
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        for ng in ngrams:
            if ng in self.vocab_index:
                vec[self.vocab_index[ng]] += 1
        if vec.sum() > 0:
            vec = vec / vec.sum()
        return vec

    def embed_corpus(self) -> np.ndarray:
        return self.corpus_embeddings

    def query(self, text: str, sentence_set: list, k: int) -> list:
        sentences = [self._normalize(s) for s in sentence_set]
        matrix = np.stack([self.embed(s) for s in sentences])
        query_vec = self.embed(text)
        sims = matrix @ query_vec
        exp_sim = np.exp(sims - np.max(sims))
        probs = exp_sim / exp_sim.sum()
        topk = np.argsort(probs)[-k:][::-1]
        return [sentence_set[i] for i in topk]

    def query_in_initial_set(self, text: str, k: int = 5) -> list:
        query_vec = self.embed(text)
        sims = self.corpus_embeddings @ query_vec
        exp_sim = np.exp(sims - np.max(sims))
        probs = exp_sim / exp_sim.sum()
        topk = np.argsort(probs)[-k:][::-1]
        return [self.raw_corpus[i] for i in topk]

    def export(self, path):
        joblib.dump({
            'n': self.n,
            'raw_corpus': self.raw_corpus,  # 🔹 Thêm vào
            'corpus': self.corpus,
            'vocab': self.vocab,
            'corpus_embeddings': self.corpus_embeddings
        }, path)

    def _load(self, path):
        data = joblib.load(path)
        self.n = data['n']
        self.raw_corpus = data['raw_corpus']  # 🔹 Lấy lại
        self.corpus = data['corpus']
        self.vocab = data['vocab']
        self.vocab_index = {ng: i for i, ng in enumerate(self.vocab)}
        self.corpus_embeddings = data['corpus_embeddings']

