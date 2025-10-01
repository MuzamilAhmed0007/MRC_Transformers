# src/embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingProvider:
    def __init__(self, model_name="paraphrase-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
