"""
Pretrained Dense Evidence Retrieval Module for FinGuard.
Uses 'sentence-transformers/all-MiniLM-L6-v2' to encode evidence sentences and claims
into normalized dense embeddings.
Features:
1. Pretrained Transformer encoder (AutoTokenizer + AutoModel + Mean Pooling).
2. Local disk caching (outputs/dense_evidence_embeddings.npz) to avoid re-encoding.
3. Fast inner-product cosine similarity retrieval.
4. Preserves evidence IDs, source URLs, and parent claim IDs.
5. CPU/CUDA compatible with graceful offline fallback.
"""

import os
import sys
import json
import numpy as np
import torch
from typing import List, Dict, Tuple, Any

DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CACHE_PATH = "outputs/dense_evidence_embeddings.npz"

def mean_pooling(model_output, attention_mask):
    """Mean Pooling - Take attention mask into account for correct averaging."""
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class PretrainedDenseRetriever:
    def __init__(self, model_name: str = DENSE_MODEL_NAME, cache_path: str = DEFAULT_CACHE_PATH, device: str = None):
        self.model_name = model_name
        self.cache_path = cache_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.is_available = False
        
        # In-memory corpus & embeddings
        self.corpus: List[Dict[str, Any]] = []
        self.doc_ids: List[int] = []
        self.embeddings: np.ndarray = None # normalized shape (N, 384)
        
        self._init_model()

    def _init_model(self):
        """Initializes tokenizer and model with offline resilience."""
        try:
            from transformers import AutoTokenizer, AutoModel
            print(f"Loading pretrained dense retriever: {self.model_name} on {self.device}...", flush=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self.is_available = True
            print("Pretrained dense retriever successfully initialized!", flush=True)
        except Exception as e:
            print(f"Notice: Pretrained dense model could not be loaded ({e}). Dense retrieval unavailable.", flush=True)
            self.is_available = False

    def encode_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Encodes a list of texts into normalized dense numpy vectors."""
        if not self.is_available or self.model is None:
            raise RuntimeError("Dense retriever model is not available.")
            
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                encoded_input = self.tokenizer(
                    batch_texts, padding=True, truncation=True, max_length=128, return_tensors='pt'
                ).to(self.device)
                model_output = self.model(**encoded_input)
                sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
                # L2 normalize
                sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
                all_embeddings.append(sentence_embeddings.cpu().numpy())
                
        return np.vstack(all_embeddings)

    def build_index(self, corpus: List[Dict[str, Any]], force_recompute: bool = False):
        """
        Encodes corpus or loads from local cache if unchanged.
        """
        self.corpus = corpus
        self.doc_ids = [doc["doc_id"] for doc in corpus]
        
        # Check if cached embeddings exist on disk
        if not force_recompute and os.path.exists(self.cache_path):
            try:
                data = np.load(self.cache_path, allow_pickle=True)
                cached_doc_ids = data["doc_ids"]
                if len(cached_doc_ids) == len(self.doc_ids) and np.array_equal(cached_doc_ids, self.doc_ids):
                    self.embeddings = data["embeddings"]
                    print(f"Loaded {len(self.embeddings):,} cached dense embeddings from {self.cache_path}", flush=True)
                    return
            except Exception as e:
                print(f"Notice: Could not load cache ({e}), recomputing embeddings...", flush=True)

        if not self.is_available:
            print("Cannot build dense index: dense model is unavailable.", flush=True)
            return

        print(f"Computing dense embeddings for {len(corpus):,} evidence sentences (batch size 64)...", flush=True)
        texts = [doc["text"] for doc in corpus]
        self.embeddings = self.encode_texts(texts, batch_size=64)
        
        # Cache to disk
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        np.savez_compressed(self.cache_path, doc_ids=np.array(self.doc_ids), embeddings=self.embeddings)
        print(f"Saved dense embeddings to {self.cache_path}", flush=True)

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Encodes query and retrieves top_k documents by cosine similarity.
        Returns list of (doc_id, cosine_similarity).
        """
        if not self.is_available or self.embeddings is None:
            return []
            
        q_vec = self.encode_texts([query])[0] # shape (384,)
        # Inner product of normalized vectors equals cosine similarity
        sims = np.dot(self.embeddings, q_vec)
        top_indices = np.argsort(sims)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            doc_id = self.doc_ids[idx]
            results.append((doc_id, float(sims[idx])))
        return results

if __name__ == "__main__":
    retriever = PretrainedDenseRetriever()
    sample_corpus = [
        {"doc_id": 1, "text": "Joe Biden called for an end to shareholder capitalism."},
        {"doc_id": 2, "text": "Amazon paid zero dollars in federal corporate income taxes in 2018."},
        {"doc_id": 3, "text": "The national debt reached twenty trillion dollars."}
    ]
    retriever.build_index(sample_corpus, force_recompute=True)
    res = retriever.retrieve("corporate taxes Amazon", top_k=2)
    print("Test Dense Retrieval:", res)
