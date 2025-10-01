# src/vector_db.py
import numpy as np
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

class VectorDB:
    def __init__(self, dim, use_faiss=False):
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.dim = dim
        self.vectors = []
        self.metadatas = []
        if self.use_faiss:
            self.index = faiss.IndexFlatL2(dim)

    def add(self, vectors, metadata_list):
        if self.use_faiss:
            self.index.add(vectors.astype('float32'))
        else:
            self.vectors.append(vectors)
        self.metadatas.extend(metadata_list)

    def search(self, qvec, top_k=5):
        if self.use_faiss:
            D, I = self.index.search(qvec.astype('float32'), top_k)
            return I, D
        else:
            # brute force
            stacked = np.vstack(self.vectors) if len(self.vectors) else np.zeros((0, self.dim))
            if stacked.shape[0] == 0:
                return [], []
            sims = np.dot(stacked, qvec.T).squeeze()
            idx = np.argsort(-sims)[:top_k]
            return idx, sims[idx]
