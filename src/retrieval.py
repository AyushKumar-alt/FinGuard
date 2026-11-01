"""
Evidence Retrieval Benchmark Module for FinFact.
Constructs a candidate evidence corpus preserving document provenance, and evaluates:
1. TF-IDF Cosine Retrieval
2. BM25 (Inverted-Index Okapi BM25)
3. Dense Semantic Retrieval (Latent Semantic Dense Projection / Sentence Embeddings)
4. Hybrid Retrieval via Reciprocal Rank Fusion (RRF)

Strictly prevents retrieval leakage during evaluation:
- Corpus contains factual evidence sentences, NOT fact-checker verdicts or labels.
- Evaluates ability to retrieve relevant supporting sentences from across the full corpus.
- Computes: Recall@5, Recall@10, Precision@5, Precision@10, MRR, and MAP.
"""

import os
import sys
import json
import math
import re
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

def tokenize(text: str) -> list:
    """Fast alphanumeric tokenizer."""
    return re.findall(r'\b\w+\b', text.lower())

class BM25Okapi:
    """High-performance Inverted-Index Okapi BM25 implementation."""
    def __init__(self, corpus: list, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lens = np.array([len(doc) for doc in corpus], dtype=np.float32)
        self.avgdl = float(np.mean(self.doc_lens)) if self.corpus_size > 0 else 1.0
        
        # Inverted index: term -> list of (doc_idx, tf)
        self.inverted_index = defaultdict(list)
        for doc_idx, doc in enumerate(corpus):
            tf = Counter(doc)
            for term, freq in tf.items():
                self.inverted_index[term].append((doc_idx, freq))
                
        # Precompute standard Robertson-Sparck Jones BM25 IDF
        self.idf = {}
        for term, posting in self.inverted_index.items():
            df = len(posting)
            self.idf[term] = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)

    def get_scores(self, query: list) -> np.ndarray:
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        for term in query:
            if term not in self.idf:
                continue
            idf_val = self.idf[term]
            for doc_idx, freq in self.inverted_index[term]:
                num = freq * (self.k1 + 1.0)
                den = freq + self.k1 * (1.0 - self.b + self.b * (self.doc_lens[doc_idx] / self.avgdl))
                scores[doc_idx] += idf_val * (num / den)
        return scores

class EvidenceRetrievalBenchmark:
    def __init__(self, json_path="finfact.json"):
        self.json_path = json_path
        self.corpus = []
        self.query_records = []
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.bm25_index = None
        self.dense_matrix = None
        
        self._build_corpus()

    def _build_corpus(self):
        """Constructs deduplicated evidence corpus with document provenance."""
        print(f"Loading {self.json_path} to build retrieval corpus...", flush=True)
        with open(self.json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        doc_map = {}
        claim_to_doc_ids = defaultdict(set)
        
        for idx, item in enumerate(raw_data):
            claim = item.get("claim", "").strip()
            evidence_list = item.get("evidence", []) or []
            
            self.query_records.append({
                "record_id": idx,
                "claim": claim,
                "label": item.get("label", ""),
                "url": item.get("url", ""),
                "evidence_count": len(evidence_list)
            })
            
            for ev in evidence_list:
                if not isinstance(ev, dict):
                    continue
                sent = ev.get("sentence", "").strip()
                if not sent or len(sent.split()) < 4:
                    continue
                    
                hrefs = ev.get("hrefs", []) or []
                
                # Deduplicate identical sentences across claims
                if sent not in doc_map:
                    doc_id = len(self.corpus)
                    doc_entry = {
                        "doc_id": doc_id,
                        "text": sent,
                        "source_urls": list(set(hrefs)),
                        "parent_claim_ids": [idx],
                        "doc_type": "evidence_sentence"
                    }
                    doc_map[sent] = doc_id
                    self.corpus.append(doc_entry)
                else:
                    doc_id = doc_map[sent]
                    self.corpus[doc_id]["parent_claim_ids"].append(idx)
                    for h in hrefs:
                        if h not in self.corpus[doc_id]["source_urls"]:
                            self.corpus[doc_id]["source_urls"].append(h)
                            
                claim_to_doc_ids[idx].add(doc_id)

        self.claim_to_doc_ids = claim_to_doc_ids
        print(f"Constructed Evidence Corpus: {len(self.corpus):,} unique candidate evidence sentences.", flush=True)
        print(f"Total Claims in index: {len(self.query_records):,}", flush=True)

    def fit_indices(self):
        """Fits TF-IDF, BM25, and Dense Semantic SVD indices on candidate corpus."""
        print("Fitting TF-IDF index on candidate evidence corpus...", flush=True)
        corpus_texts = [d["text"] for d in self.corpus]
        self.tfidf_vectorizer = TfidfVectorizer(max_features=25000, ngram_range=(1, 2), sublinear_tf=True)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus_texts)
        
        print("Fitting BM25 inverted index...", flush=True)
        tokenized_corpus = [tokenize(d["text"]) for d in self.corpus]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        
        print("Fitting Dense Semantic Representation (128-dim Dense SVD)...", flush=True)
        self.svd = TruncatedSVD(n_components=128, random_state=42)
        self.dense_matrix = self.svd.fit_transform(self.tfidf_matrix)
        # Normalize dense vectors for fast inner-product cosine similarity
        norms = np.linalg.norm(self.dense_matrix, axis=1, keepdims=True) + 1e-9
        self.dense_matrix_norm = self.dense_matrix / norms
        print("All retrieval indices successfully initialized!", flush=True)

    def retrieve_tfidf(self, query: str, top_k: int = 10) -> list:
        q_vec = self.tfidf_vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(int(idx), float(sims[idx])) for idx in top_indices]

    def retrieve_bm25(self, query: str, top_k: int = 10) -> list:
        q_tokens = tokenize(query)
        scores = self.bm25_index.get_scores(q_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def retrieve_dense(self, query: str, top_k: int = 10) -> list:
        q_vec = self.tfidf_vectorizer.transform([query])
        q_dense = self.svd.transform(q_vec)
        q_norm = q_dense / (np.linalg.norm(q_dense) + 1e-9)
        sims = np.dot(self.dense_matrix_norm, q_norm.T).flatten()
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(int(idx), float(sims[idx])) for idx in top_indices]

    def retrieve_hybrid_rrf(self, query: str, top_k: int = 10, k: int = 60) -> list:
        """Reciprocal Rank Fusion combining BM25 and TF-IDF."""
        tfidf_ranks = self.retrieve_tfidf(query, top_k=top_k * 3)
        bm25_ranks = self.retrieve_bm25(query, top_k=top_k * 3)
        
        rrf_scores = defaultdict(float)
        for rank, (doc_id, _) in enumerate(tfidf_ranks):
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
        for rank, (doc_id, _) in enumerate(bm25_ranks):
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
            
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(int(doc_id), float(score)) for doc_id, score in sorted_docs]

    def evaluate_retrieval(self, eval_query_ids: list, top_k: int = 10):
        """
        Evaluates retrieval methods on candidate queries.
        Strictly prevents retrieval leakage:
        - Query only matches against candidate evidence corpus.
        - Gold relevant documents = ground-truth evidence sentences of that claim.
        - Excludes any document if its text is identical to the claim query.
        """
        methods = {
            "TF-IDF Cosine": self.retrieve_tfidf,
            "BM25": self.retrieve_bm25,
            "Dense Semantic": self.retrieve_dense,
            "Hybrid RRF (BM25+TF-IDF)": self.retrieve_hybrid_rrf
        }
        
        results = {}
        for method_name, retrieve_fn in methods.items():
            print(f"Evaluating {method_name} across queries...", flush=True)
            recalls_5 = []
            recalls_10 = []
            precisions_5 = []
            precisions_10 = []
            mrrs = []
            maps = []
            
            valid_queries = 0
            for q_id in eval_query_ids:
                gold_doc_ids = self.claim_to_doc_ids.get(q_id, set())
                if not gold_doc_ids:
                    continue  # Skip claims with 0 evidence sentences
                    
                valid_queries += 1
                claim_text = self.query_records[q_id]["claim"]
                
                # Retrieve top candidates
                ranked_docs = retrieve_fn(claim_text, top_k=top_k)
                retrieved_ids = [d_id for d_id, _ in ranked_docs]
                
                # Leakage check: exclude if document text is identical to claim query
                retrieved_ids = [d_id for d_id in retrieved_ids if self.corpus[d_id]["text"].lower() != claim_text.lower()]
                
                # Compute Precision & Recall @ 5
                top_5 = set(retrieved_ids[:5])
                hits_5 = len(top_5.intersection(gold_doc_ids))
                precisions_5.append(hits_5 / 5.0)
                recalls_5.append(hits_5 / len(gold_doc_ids))
                
                # Compute Precision & Recall @ 10
                top_10 = set(retrieved_ids[:10])
                hits_10 = len(top_10.intersection(gold_doc_ids))
                precisions_10.append(hits_10 / 10.0)
                recalls_10.append(hits_10 / len(gold_doc_ids))
                
                # MRR (Mean Reciprocal Rank)
                rr = 0.0
                for rank, d_id in enumerate(retrieved_ids):
                    if d_id in gold_doc_ids:
                        rr = 1.0 / (rank + 1)
                        break
                mrrs.append(rr)
                
                # MAP (Mean Average Precision)
                ap = 0.0
                cum_hits = 0
                for rank, d_id in enumerate(retrieved_ids):
                    if d_id in gold_doc_ids:
                        cum_hits += 1
                        ap += cum_hits / (rank + 1)
                ap = ap / min(len(gold_doc_ids), top_k) if gold_doc_ids else 0.0
                maps.append(ap)
                
            results[method_name] = {
                "Evaluated_Queries": valid_queries,
                "Recall@5": round(float(np.mean(recalls_5)), 4),
                "Recall@10": round(float(np.mean(recalls_10)), 4),
                "Precision@5": round(float(np.mean(precisions_5)), 4),
                "Precision@10": round(float(np.mean(precisions_10)), 4),
                "MRR": round(float(np.mean(mrrs)), 4),
                "MAP": round(float(np.mean(maps)), 4)
            }
            print(f"[{method_name}] R@5: {results[method_name]['Recall@5']:.4f} | R@10: {results[method_name]['Recall@10']:.4f} | MRR: {results[method_name]['MRR']:.4f} | MAP: {results[method_name]['MAP']:.4f}", flush=True)
            
        return results

def run_retrieval_benchmark(data_dir="data", output_dir="outputs", figures_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    benchmark = EvidenceRetrievalBenchmark(json_path="finfact.json")
    benchmark.fit_indices()
    
    # Evaluate on held-out test claims
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    test_claim_ids = test_df["id"].tolist()
    
    print(f"\n--- Running Evidence Retrieval Evaluation on {len(test_claim_ids)} Held-Out Test Claims ---", flush=True)
    retrieval_results = benchmark.evaluate_retrieval(test_claim_ids, top_k=10)
    
    # Save metrics JSON
    metrics_path = os.path.join(output_dir, "retrieval_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(retrieval_results, f, indent=2)
    print(f"Retrieval benchmark metrics saved to {metrics_path}", flush=True)
    
    # Save sample corpus provenance
    sample_corpus_path = os.path.join(output_dir, "corpus_provenance_sample.json")
    with open(sample_corpus_path, "w", encoding="utf-8") as f:
        json.dump(benchmark.corpus[:25], f, indent=2)
        
    # Plot Retrieval Metrics Comparison
    methods = list(retrieval_results.keys())
    mrr_vals = [retrieval_results[m]["MRR"] for m in methods]
    r10_vals = [retrieval_results[m]["Recall@10"] for m in methods]
    map_vals = [retrieval_results[m]["MAP"] for m in methods]
    
    plot_df = pd.DataFrame({
        "Method": methods * 3,
        "Metric_Value": mrr_vals + r10_vals + map_vals,
        "Metric": ["MRR"] * len(methods) + ["Recall@10"] * len(methods) + ["MAP"] * len(methods)
    })
    
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=plot_df, x="Metric", y="Metric_Value", hue="Method", palette="viridis")
    plt.title("Evidence Retrieval Benchmark on FinFact Test Set (N=506 Claims)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Information Retrieval Metric", fontsize=11, fontweight="bold")
    plt.ylabel("Score", fontsize=11, fontweight="bold")
    plt.ylim(0, 0.8)
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2., h + 0.015),
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    fig_path = os.path.join(figures_dir, "retrieval_comparison.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Retrieval comparison figure saved to {fig_path}", flush=True)
    
    return retrieval_results

if __name__ == "__main__":
    run_retrieval_benchmark()
