import numpy as np
import joblib
import os

class KNNVectorSearch:
    def __init__(self, X_train=None, Y_train=None, path=None):
        if path is not None and os.path.exists(path):
            self._load(path)
        elif X_train is not None and Y_train is not None:
            self.X_train = self._normalize(X_train)  # shape: (n_samples, dim)
            self.Y_train = np.array(Y_train)
        else:
            raise ValueError("Must provide either X_train and Y_train or a valid path to load from.")

    def _normalize(self, X):
        # Normalize vectors to unit length for cosine similarity
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.clip(norms, 1e-10, None)

    def query(self, x, k=5):
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        x = self._normalize(x)

        # Cosine similarity is dot product since vectors are normalized
        sims = self.X_train @ x.T  # shape: (n_samples, 1)
        sims = sims.squeeze()
        topk = np.argsort(sims)[-k:][::-1]
        return [(self.Y_train[i], sims[i]) for i in topk]

    def batch_query(self, X_batch, k=5):
        X_batch = self._normalize(X_batch)
        sims = self.X_train @ X_batch.T  # shape: (n_samples, batch_size)
        topk_indices = np.argsort(sims, axis=0)[-k:][::-1]  # shape: (k, batch_size)

        results = []
        for i in range(X_batch.shape[0]):
            indices = topk_indices[:, i]
            results.append([(self.Y_train[idx], sims[idx, i]) for idx in indices])
        return results

    def export(self, path):
        joblib.dump({
            'X_train': self.X_train,
            'Y_train': self.Y_train
        }, path)

    def _load(self, path):
        data = joblib.load(path)
        self.X_train = data['X_train']
        self.Y_train = data['Y_train']
