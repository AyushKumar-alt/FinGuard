"""
Pretrained Cross-Encoder Reranker Module for FinGuard.
Uses 'cross-encoder/ms-marco-MiniLM-L-6-v2' to jointly score (claim, evidence) pairs for relevance.

Crucial Distinctions:
1. A cross-encoder relevance score indicates semantic relevance and query alignment, NOT veracity.
2. It does not replace the NLI verifier; it filters candidate retrieval before verification.
3. Preserves all document provenance: doc_id, text, source_urls, parent_claim_ids.
4. Supports CPU execution with offline resilience.
"""

import os
import torch
import numpy as np
from typing import List, Dict, Any, Tuple

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class PretrainedCrossEncoderReranker:
    def __init__(self, model_name: str = RERANKER_MODEL_NAME, device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.is_available = False
        
        self._init_model()

    def _init_model(self):
        """Initializes cross-encoder tokenizer and classification head."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            print(f"Loading pretrained cross-encoder reranker: {self.model_name} on {self.device}...", flush=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self.is_available = True
            print("Pretrained cross-encoder reranker successfully initialized!", flush=True)
        except Exception as e:
            print(f"Notice: Pretrained cross-encoder unavailable ({e}). Reranker will operate in passthrough mode.", flush=True)
            self.is_available = False

    def rerank(self, claim: str, candidate_docs: List[Dict[str, Any]], top_k: int = 5, batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Scores candidate claim-evidence pairs jointly using cross-encoder.
        Returns top_k candidate documents sorted by relevance score.
        """
        if not candidate_docs:
            return []
            
        if not self.is_available or self.model is None:
            # Fallback: return top_k candidates in their existing order
            return candidate_docs[:top_k]
            
        pairs = [[claim, doc["text"]] for doc in candidate_docs]
        scores = []
        
        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                encoded = self.tokenizer(
                    batch_pairs, padding=True, truncation=True, max_length=160, return_tensors='pt'
                ).to(self.device)
                logits = self.model(**encoded).logits.flatten()
                # Apply sigmoid to normalize to [0, 1] relevance score
                probs = torch.sigmoid(logits).cpu().numpy()
                scores.extend(probs.tolist())
                
        # Attach rerank score and sort
        reranked_docs = []
        for doc, score in zip(candidate_docs, scores):
            doc_copy = dict(doc)
            doc_copy["rerank_score"] = round(float(score), 4)
            reranked_docs.append(doc_copy)
            
        reranked_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked_docs[:top_k]

if __name__ == "__main__":
    reranker = PretrainedCrossEncoderReranker()
    claim = "Did Amazon pay zero federal corporate income tax in 2018?"
    candidates = [
        {"doc_id": 1, "text": "The Federal Reserve adjusted benchmark interest rates to stabilize inflation.", "source_urls": ["url1"]},
        {"doc_id": 2, "text": "Official SEC filings confirmed Amazon paid $0 in federal income taxes for 2018 despite record profits.", "source_urls": ["url2"]},
        {"doc_id": 3, "text": "Amazon launched a new retail distribution center in the midwest.", "source_urls": ["url3"]}
    ]
    ranked = reranker.rerank(claim, candidates, top_k=2)
    print("\nTest Reranking Output:")
    for r in ranked:
        print(f"  Doc {r['doc_id']} (Score: {r['rerank_score']}): {r['text']}")
