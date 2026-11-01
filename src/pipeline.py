"""
FinGuard End-to-End Financial Claim Verification Pipeline.
Orchestrates:
1. Financial Claim Understanding (Entities, metrics, amounts, periods, directions)
2. Query Construction & Expansion (Direction-preserving financial expansions)
3. Multi-Strategy Evidence Retrieval (BM25, Pretrained Dense, Hybrid RRF, Boolean/Phrase)
4. Cross-Encoder Joint Reranking (ms-marco-MiniLM-L-6-v2)
5. Evidence-Grounded Verification (Pretrained Cross-Encoder NLI / Calibrated Baseline)
6. Multi-Sentence Conflict & Uncertainty Aggregation

Provides clear fallbacks for offline environments and prevents source fabrication.
"""

import os
import sys
from typing import Dict, List, Any, Optional

# Ensure local imports work reliably
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.claim_understanding import extract_financial_entities
from src.query_construction import FinancialQueryBuilder
from src.retrieval import EvidenceRetrievalBenchmark
from src.dense_retrieval import PretrainedDenseRetriever
from src.reranker import PretrainedCrossEncoderReranker
from src.verifier import FinancialClaimVerifier

class FinGuardPipeline:
    def __init__(self, json_path: str = "finfact.json", load_heavy_models: bool = True,
                 nli_model_name: str = "cross-encoder/nli-distilroberta-base"):
        self.json_path = json_path
        print(f"Initializing FinGuard Pipeline with {json_path}...", flush=True)
        
        # 1. Query Builder
        self.query_builder = FinancialQueryBuilder(max_expansions=3)
        
        # 2. Evidence Retrieval Corpus & Indices (BM25, TF-IDF)
        self.retriever = EvidenceRetrievalBenchmark(json_path=json_path)
        self.retriever.fit_indices()
        self.corpus = self.retriever.corpus
        
        # 3. Dense Retriever (all-MiniLM-L6-v2 with local cache)
        self.dense_retriever = None
        if load_heavy_models:
            try:
                self.dense_retriever = PretrainedDenseRetriever()
                if self.dense_retriever.is_available:
                    self.dense_retriever.build_index(self.corpus, force_recompute=False)
            except Exception as e:
                print(f"Notice: Dense retriever failed to initialize ({e}). Falling back to lexical retrieval.", flush=True)
                self.dense_retriever = None
                
        # 4. Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)
        self.reranker = None
        if load_heavy_models:
            try:
                self.reranker = PretrainedCrossEncoderReranker()
            except Exception as e:
                print(f"Notice: Reranker failed to initialize ({e}). Reranking will be passthrough.", flush=True)
                self.reranker = None
                
        # 5. Financial Claim Verifier (NLI Transformer + Calibrated Baseline)
        self.verifier = FinancialClaimVerifier(
            baseline_model_path="outputs/best_baseline_model.pkl",
            nli_model_name=nli_model_name,
            use_transformer=load_heavy_models
        )
        print("FinGuard Pipeline successfully initialized!", flush=True)

    @staticmethod
    def _reciprocal_rank_fusion(rankings, top_k: int, k: int = 60):
        """Fuse ranked (doc_id, score) lists using reciprocal rank contributions."""
        fused_scores = {}
        for ranking in rankings:
            for rank, (doc_id, _) in enumerate(ranking, start=1):
                fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        return sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

    @staticmethod
    def _weighted_reciprocal_rank_fusion(weighted_rankings, top_k: int = 15, k: int = 60):
        """
        Fuse multiple ranked lists using Weighted Reciprocal Rank Fusion:
            RRF(d) = sum_{q in Q} w_q / (k + rank_q(d))
        Where weighted_rankings is a list of (ranked_list, weight) tuples.
        """
        fused_scores = {}
        for ranking, weight in weighted_rankings:
            w = float(weight)
            for rank, (doc_id, _) in enumerate(ranking, start=1):
                fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + w / (k + rank)
        return sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

    def run(self, claim: str,
            retrieval_mode: str = "bm25",
            use_query_expansion: bool = True,
            use_reranker: bool = True,
            use_nli: bool = True,
            top_k_retrieve: int = 15,
            top_k_rerank: int = 4) -> Dict[str, Any]:
        """
        Executes end-to-end verification of a financial claim.
        """
        if not claim or not claim.strip():
            return {
                "claim": "",
                "verdict": "NEI",
                "verdict_display": "NEI (Empty Claim)",
                "confidence": 0.0,
                "is_uncertain": True,
                "uncertainty_reason": "No claim text provided.",
                "explanation": "Please enter a non-empty financial claim.",
                "supporting_evidence": [],
                "contradicting_evidence": [],
                "neutral_evidence": []
            }

        # Step 1: Claim Understanding
        parsed_claim = extract_financial_entities(claim)
        
        # Step 2: Query Construction & Expansion
        query_info = self.query_builder.build_queries(claim, parsed_claim)
        multi_queries = query_info.get("multi_queries", [])
        search_query = query_info["combined_query"] if use_query_expansion else claim
        is_expansion_enabled = use_query_expansion and getattr(self.query_builder, "enabled", True)
        
        # Step 3: Retrieval Execution
        retrieved_pairs = [] # list of (doc_id, score)
        used_mode = retrieval_mode.lower()

        if used_mode in ("hybrid_bm25_dense", "hybrid_rrf_bm25_dense"):
            if self.dense_retriever and self.dense_retriever.is_available:
                bm25_ranks = self.retriever.retrieve_bm25(search_query, top_k=top_k_retrieve * 2)
                dense_ranks = self.dense_retriever.retrieve(search_query, top_k=top_k_retrieve * 2)
                retrieved_pairs = self._reciprocal_rank_fusion(
                    [bm25_ranks, dense_ranks], top_k=top_k_retrieve
                )
                retrieval_engine = "hybrid_bm25_dense_rrf"
            else:
                retrieved_pairs = self.retriever.retrieve_bm25(search_query, top_k=top_k_retrieve)
                retrieval_engine = "bm25_fallback_dense_unavailable"
        elif used_mode == "dense":
            if self.dense_retriever and self.dense_retriever.is_available:
                retrieved_pairs = self.dense_retriever.retrieve(search_query, top_k=top_k_retrieve)
                retrieval_engine = "pretrained_dense"
            else:
                retrieved_pairs = self.retriever.retrieve_bm25(search_query, top_k=top_k_retrieve)
                retrieval_engine = "bm25_fallback_dense_unavailable"
        elif used_mode in ("svd", "dense_svd", "lsa"):
            retrieved_pairs = self.retriever.retrieve_dense(search_query, top_k=top_k_retrieve)
            retrieval_engine = "tfidf_svd"
        elif used_mode in ("hybrid", "rrf", "hybrid_rrf"):
            retrieved_pairs = self.retriever.retrieve_hybrid_rrf(search_query, top_k=top_k_retrieve)
            retrieval_engine = "hybrid_bm25_tfidf_rrf"
        elif used_mode == "tfidf":
            retrieved_pairs = self.retriever.retrieve_tfidf(search_query, top_k=top_k_retrieve)
            retrieval_engine = "tfidf"
        elif used_mode in ("p1_lemma", "lemma_p1"):
            try:
                from src.p1_pipeline import Pipeline1
                if not hasattr(self, "_p1_pipeline") or self._p1_pipeline is None:
                    self._p1_pipeline = Pipeline1()
                retrieved_pairs = self._p1_pipeline.retrieve(search_query, top_k=top_k_retrieve)
                retrieval_engine = "pipeline1_lemmatized_tfidf"
            except Exception as e:
                print(f"Pipeline 1 retrieval failed ({e}), falling back to BM25.", flush=True)
                retrieved_pairs = self.retriever.retrieve_bm25(search_query, top_k=top_k_retrieve)
                retrieval_engine = "bm25_fallback_pipeline1_unavailable"
        elif used_mode == "bm25":
            if is_expansion_enabled and multi_queries and len(multi_queries) > 1:
                per_query_k = self.query_builder.config.get("per_query_top_k", top_k_retrieve)
                final_k = self.query_builder.config.get("final_candidate_top_k", top_k_retrieve)
                rrf_k = self.query_builder.config.get("rrf_k", 60)
                
                weighted_rankings = []
                for q_obj in multi_queries:
                    ranks = self.retriever.retrieve_bm25(q_obj["query"], top_k=per_query_k)
                    weighted_rankings.append((ranks, float(q_obj.get("weight", 1.0))))
                    
                retrieved_pairs = self._weighted_reciprocal_rank_fusion(
                    weighted_rankings, top_k=final_k, k=rrf_k
                )
                retrieval_engine = "bm25_weighted_rrf"
            else:
                retrieved_pairs = self.retriever.retrieve_bm25(search_query, top_k=top_k_retrieve)
                retrieval_engine = "bm25"
        else:
            raise ValueError(f"Unsupported retrieval_mode: {retrieval_mode}")

        # Assemble candidate docs with provenance
        candidate_docs = []
        for doc_id, score in retrieved_pairs:
            if hasattr(self, "_p1_pipeline") and self._p1_pipeline and ("p1_lemma" in used_mode or "lemma_p1" in used_mode):
                text = self._p1_pipeline.p1_index.get_original_text(doc_id)
                meta = self._p1_pipeline.p1_index.inner_index.doc_metadata.get(doc_id, {})
                candidate_docs.append({
                    "doc_id": doc_id,
                    "text": text,
                    "score": round(float(score), 4),
                    "source_urls": meta.get("source_urls", []),
                    "parent_claim_ids": meta.get("parent_claim_ids", []),
                    "retrieval_pipeline": retrieval_engine,
                })
            elif doc_id < len(self.corpus):
                doc = self.corpus[doc_id]
                candidate_docs.append({
                    "doc_id": doc_id,
                    "text": doc["text"],
                    "score": round(float(score), 4),
                    "source_urls": doc["source_urls"],
                    "parent_claim_ids": doc["parent_claim_ids"],
                    "retrieval_pipeline": retrieval_engine,
                })

        # Step 4: Cross-Encoder Joint Reranking
        if use_reranker and self.reranker and self.reranker.is_available and candidate_docs:
            selected_docs = self.reranker.rerank(claim, candidate_docs, top_k=top_k_rerank)
            reranker_active = True
        else:
            selected_docs = candidate_docs[:top_k_rerank]
            reranker_active = False

        # Step 5: Verification (NLI or Calibrated Baseline)
        engine_type = "transformer_nli" if (use_nli and self.verifier.has_transformer) else "baseline"
        verification_res = self.verifier.verify_claim(
            claim=claim,
            evidence_docs=selected_docs,
            engine_choice=engine_type
        )

        return {
            "claim": claim,
            "parsed_claim": parsed_claim,
            "query_info": query_info,
            "search_query": search_query,
            "retrieval_mode": retrieval_mode,
            "retrieval_engine": retrieval_engine,
            "total_candidates_retrieved": len(candidate_docs),
            "selected_evidence_docs": selected_docs,
            "reranker_active": reranker_active,
            "verdict": verification_res["verdict"],
            "verdict_display": verification_res["verdict_display"],
            "confidence": verification_res["confidence"],
            "class_probabilities": verification_res["class_probabilities"],
            "is_uncertain": verification_res["is_uncertain"],
            "is_conflicting": verification_res["is_conflicting"],
            "uncertainty_reason": verification_res["uncertainty_reason"],
            "explanation": verification_res["explanation"],
            "supporting_evidence": verification_res["supporting_evidence"],
            "contradicting_evidence": verification_res["contradicting_evidence"],
            "neutral_evidence": verification_res["neutral_evidence"],
            "verifier_engine": verification_res["active_engine"],
            "component_status": {
                "dense_retriever": self.dense_retriever is not None and self.dense_retriever.is_available,
                "cross_encoder_reranker": self.reranker is not None and self.reranker.is_available,
                "nli_verifier": self.verifier.has_transformer,
                "baseline_verifier": self.verifier.baseline_model is not None
            }
        }

if __name__ == "__main__":
    pipeline = FinGuardPipeline(json_path="finfact.json", load_heavy_models=True)
    test_claim = "Did Biden Call for an 'End to Shareholder Capitalism' in July 2020?"
    out = pipeline.run(test_claim, retrieval_mode="bm25", use_query_expansion=True, use_reranker=True, use_nli=True)
    print("\nPipeline Result:")
    print("Verdict:", out["verdict"])
    print("Confidence:", out["confidence"])
    print("Engine:", out["verifier_engine"])
    print("Selected Evidence count:", len(out["selected_evidence_docs"]))
    for i, ev in enumerate(out["selected_evidence_docs"]):
        print(f"  [{i+1}] (Score: {ev.get('rerank_score', ev.get('score'))}): {ev['text']}")
