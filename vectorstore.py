import os
import pickle

import faiss
import numpy as np

import config


class VectorStore:
    def __init__(self):
        self.index = None
        self.metadata = []

    def build(self, embeddings, metadata):
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        self.metadata = metadata

    def save(self):
        os.makedirs(config.DATA_DIR, exist_ok=True)
        faiss.write_index(self.index, config.INDEX_PATH)
        with open(config.METADATA_PATH, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        if not os.path.exists(config.INDEX_PATH) or not os.path.exists(config.METADATA_PATH):
            return False
        self.index = faiss.read_index(config.INDEX_PATH)
        with open(config.METADATA_PATH, "rb") as f:
            self.metadata = pickle.load(f)
        return True

    def search(self, query_embedding, top_k):
        if self.index is None or self.index.ntotal == 0:
            return []
        query_vector = np.array([query_embedding], dtype=np.float32)
        scores, indices = self.index.search(query_vector, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((float(score), self.metadata[idx]))
        return results


_store = VectorStore()


def get_store():
    return _store
