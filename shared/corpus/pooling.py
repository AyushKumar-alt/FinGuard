"""
Candidate Evidence Pooler for FinGuard Gold Relevance Benchmark.

For a fixed, reproducible evaluation split of held-out claims:
1. Queries all 8 retrieval systems (BM25, TF-IDF, SVD, Dense, BM25+Dense RRF,
   BM25+TF-IDF RRF, Cross-Encoder Reranker, Query Expansion BM25).
2. Collects top-K candidate evidence units (default top-20 per system).
3. Deduplicates by stable evidence_id.
4. Records retrieving systems and ranks.
5. Injects all dataset-provided reference evidence sentences to ensure 100% reference recall targets.
6. Exports reviewable CSV and JSONL files for human and model-assisted annotation.
"""

import os
import sys
import json
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gold_benchmark_schema import (
    EvidenceAnnotationRecord,
    format_claim_id,
    format_evidence_id,
    parse_evidence_id,
    normalize_text_for_matching,
    BENCHMARK_VERSION
)
from src.reference_evidence import ReferenceEvidenceExtractor
from src.retrieval import EvidenceRetrievalBenchmark
from src.dense_retrieval import PretrainedDenseRetriever
from src.reranker import PretrainedCrossEncoderReranker
from src.query_construction import FinancialQueryBuilder

class CandidatePooler:
    def __init__(self,
                 benchmark_dir: str = "data/benchmark/v1",
                 pool_depth_per_system: int = 20):
        self.benchmark_dir = benchmark_dir
        self.pool_depth = pool_depth_per_system
        
        # Load canonical reference extractor
        self.extractor = ReferenceEvidenceExtractor(json_path="finfact.json")
        self.claims = self.extractor.claims
        self.corpus = self.extractor.corpus
        self.reference_mappings = self.extractor.reference_mappings
        
        # Initialize retrieval engines
        print("Initializing retrieval systems for candidate pooling...", flush=True)
        self.lexical_benchmark = EvidenceRetrievalBenchmark(json_path="finfact.json")
        self.lexical_benchmark.fit_indices()
        
        # Dense Retriever
        self.dense_retriever = PretrainedDenseRetriever()
        self.dense_retriever.build_index(self.lexical_benchmark.corpus, force_recompute=False)
        
        # Cross-Encoder Reranker
        self.reranker = PretrainedCrossEncoderReranker()
        
        # Query Builder
        self.query_builder = FinancialQueryBuilder(max_expansions=3)
        print("All retrieval engines ready for candidate pooling.", flush=True)

    def select_pilot_claim_ids(self, sample_size: int = 75, seed: int = 42) -> List[str]:
        """
        Extracts a reproducible stratified pilot sample of held-out test claims.
        Guarantees exact class distribution and preserves zero-evidence claims.
        """
        test_path = "data/test.csv"
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"{test_path} missing.")
            
        test_df = pd.read_csv(test_path)
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=sample_size, random_state=seed)
        _, pilot_idx = next(sss.split(test_df, test_df["label_id"]))
        pilot_df = test_df.iloc[pilot_idx].copy()
        
        pilot_cids = [format_claim_id(int(r["id"])) for _, r in pilot_df.iterrows()]
        print(f"Selected {len(pilot_cids)} pilot claims (seed={seed}). Class counts: {dict(pilot_df['label'].value_counts())}")
        return pilot_cids

    def pool_candidates_for_claim(self, claim_id: str) -> Dict[str, EvidenceAnnotationRecord]:
        """
        Executes all 8 retrieval systems and unions candidates for a single claim.
        """
        claim_data = self.claims[claim_id]
        claim_text = claim_data["claim_text"]
        label = claim_data["label"]
        
        # 1. Query Expansion
        q_info = self.query_builder.build_queries(claim_text)
        expanded_query = q_info["combined_query"]

        # 2. Query each system
        system_results: Dict[str, List[int]] = {}
        
        # System 1: BM25
        res_bm25 = [did for did, _ in self.lexical_benchmark.retrieve_bm25(claim_text, top_k=self.pool_depth)]
        system_results["BM25"] = res_bm25
        
        # System 2: TF-IDF
        res_tfidf = [did for did, _ in self.lexical_benchmark.retrieve_tfidf(claim_text, top_k=self.pool_depth)]
        system_results["TFIDF"] = res_tfidf
        
        # System 3: SVD
        res_svd = [did for did, _ in self.lexical_benchmark.retrieve_dense(claim_text, top_k=self.pool_depth)]
        system_results["SVD"] = res_svd
        
        # System 4: Dense Transformer
        if self.dense_retriever and self.dense_retriever.is_available:
            res_dense = [did for did, _ in self.dense_retriever.retrieve(claim_text, top_k=self.pool_depth)]
        else:
            res_dense = []
        system_results["Dense_Transformer"] = res_dense
        
        # System 5: Hybrid BM25 + Dense RRF
        rrf_dense = defaultdict(float)
        k_val = 60
        for rank, did in enumerate(res_bm25):
            rrf_dense[did] += 1.0 / (k_val + rank + 1)
        for rank, did in enumerate(res_dense):
            rrf_dense[did] += 1.0 / (k_val + rank + 1)
        res_hybrid_dense = [did for did, _ in sorted(rrf_dense.items(), key=lambda x: x[1], reverse=True)[:self.pool_depth]]
        system_results["Hybrid_BM25_Dense"] = res_hybrid_dense
        
        # System 6: Hybrid BM25 + TF-IDF RRF
        res_hybrid_tfidf = [did for did, _ in self.lexical_benchmark.retrieve_hybrid_rrf(claim_text, top_k=self.pool_depth)]
        system_results["Hybrid_BM25_TFIDF"] = res_hybrid_tfidf
        
        # System 7: Cross-Encoder Reranker
        if self.reranker and self.reranker.is_available and res_bm25:
            cand_docs = [self.lexical_benchmark.corpus[did] for did in res_bm25]
            reranked = self.reranker.rerank(claim_text, cand_docs, top_k=self.pool_depth)
            res_rerank = [d["doc_id"] for d in reranked]
        else:
            res_rerank = res_bm25[:self.pool_depth]
        system_results["Cross_Encoder_Rerank"] = res_rerank
        
        # System 8: Query Expansion BM25
        res_exp = [did for did, _ in self.lexical_benchmark.retrieve_bm25(expanded_query, top_k=self.pool_depth)]
        system_results["Query_Expansion_BM25"] = res_exp

        # Collect union of candidate document indices
        candidate_pool: Dict[str, EvidenceAnnotationRecord] = {}
        
        # Track retrieval origins
        candidate_systems: Dict[str, List[str]] = defaultdict(list)
        candidate_ranks: Dict[str, Dict[str, int]] = defaultdict(dict)
        
        for sys_name, doc_list in system_results.items():
            for rank_idx, doc_id in enumerate(doc_list):
                eid = format_evidence_id(doc_id)
                candidate_systems[eid].append(sys_name)
                candidate_ranks[eid][sys_name] = rank_idx + 1

        # Check reference evidence for this claim
        known_refs = {r["evidence_id"]: r for r in self.reference_mappings.get(claim_id, [])}
        
        # Build candidate records
        all_eids = set(candidate_systems.keys()).union(set(known_refs.keys()))
        
        for eid in all_eids:
            doc_idx = parse_evidence_id(eid)
            # Find evidence unit text
            if eid in self.corpus:
                ev_unit = self.corpus[eid]
                ev_text = ev_unit["text"]
                ev_urls = ev_unit["source_urls"]
            elif doc_idx < len(self.lexical_benchmark.corpus):
                doc_obj = self.lexical_benchmark.corpus[doc_idx]
                ev_text = doc_obj["text"]
                ev_urls = doc_obj["source_urls"]
            else:
                ev_text = known_refs.get(eid, {}).get("text", "")
                ev_urls = known_refs.get(eid, {}).get("source_urls", [])

            # Check if this candidate is a dataset-provided reference evidence
            if eid in known_refs:
                ref_info = known_refs[eid]
                relevance_grade = 3
                judgment_status = "silver_reference"
                support_relation = ref_info["initial_support_relation"]
                annotation_source = "finfact_reference"
                annotation_method = "exact_reference_match"
                annotator = "dataset_curator"
                adjudication = "pending"
                rationale = f"Dataset-provided reference evidence explicitly linked to claim in finfact.json (fact-checker verdict: {label})."
            else:
                relevance_grade = None
                judgment_status = "unjudged"
                support_relation = "unknown"
                annotation_source = "unjudged_pool"
                annotation_method = "unjudged"
                annotator = "unassigned"
                adjudication = "none"
                rationale = "Retrieved in multi-system pooling; awaiting human or model-assisted relevance judgment."

            record = EvidenceAnnotationRecord(
                claim_id=claim_id,
                evidence_id=eid,
                relevance_grade=relevance_grade,
                judgment_status=judgment_status,
                support_relation=support_relation,
                annotation_source=annotation_source,
                annotation_method=annotation_method,
                annotator_or_reviewer=annotator,
                adjudication_status=adjudication,
                rationale=rationale,
                source_url=ev_urls[0] if ev_urls else None,
                dataset_version=BENCHMARK_VERSION,
                claim_text=claim_text,
                evidence_text=ev_text,
                retrieved_by_systems=candidate_systems.get(eid, []),
                system_ranks=candidate_ranks.get(eid, {})
            )
            candidate_pool[eid] = record

        return candidate_pool

    def build_pooled_benchmark_file(self,
                                    claim_ids: List[str],
                                    output_csv: str = "data/benchmark/v1/pooled_candidates_annotation.csv",
                                    output_jsonl: str = "data/benchmark/v1/pooled_candidates_annotation.jsonl") -> Dict[str, Any]:
        """
        Pools candidates for all evaluation claims, exports reviewable files, and returns summary stats.
        """
        print(f"\nConstructing pooled evidence set across {len(claim_ids)} claims...", flush=True)
        all_records: List[EvidenceAnnotationRecord] = []
        
        claims_with_retrieved_refs = 0
        total_reference_evidence_items = 0
        total_reference_evidence_retrieved = 0
        
        for idx, cid in enumerate(claim_ids):
            pool = self.pool_candidates_for_claim(cid)
            claim_records = list(pool.values())
            all_records.extend(claim_records)
            
            # Check reference coverage
            ref_eids = {r["evidence_id"] for r in self.reference_mappings.get(cid, [])}
            total_reference_evidence_items += len(ref_eids)
            
            retrieved_eids = {r.evidence_id for r in claim_records if r.retrieved_by_systems}
            matched_refs = ref_eids.intersection(retrieved_eids)
            total_reference_evidence_retrieved += len(matched_refs)
            if matched_refs:
                claims_with_retrieved_refs += 1
                
            if (idx + 1) % 15 == 0 or (idx + 1) == len(claim_ids):
                print(f"  Pooled {idx + 1}/{len(claim_ids)} claims ({len(all_records)} total pairs accumulated)...", flush=True)

        # Export to CSV & JSONL
        records_dicts = [r.to_dict() for r in all_records]
        
        # Flatten system_ranks for clean tabular review in CSV
        df_rows = []
        for d in records_dicts:
            row = dict(d)
            row["retrieved_by_systems"] = ", ".join(d["retrieved_by_systems"])
            row["system_ranks"] = json.dumps(d["system_ranks"])
            df_rows.append(row)
            
        df = pd.DataFrame(df_rows)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False, encoding="utf-8")
        
        with open(output_jsonl, "w", encoding="utf-8") as f:
            for d in records_dicts:
                f.write(json.dumps(d) + "\n")
                
        # Calculate summary statistics
        unique_claims = len(claim_ids)
        total_pairs = len(all_records)
        unique_ev_units = len(set(r.evidence_id for r in all_records))
        judged_silver_count = sum(1 for r in all_records if r.judgment_status == "silver_reference")
        unjudged_count = sum(1 for r in all_records if r.judgment_status == "unjudged")
        
        stats = {
            "benchmark_version": BENCHMARK_VERSION,
            "evaluated_claims_count": unique_claims,
            "total_candidate_pairs": total_pairs,
            "unique_evidence_units_pooled": unique_ev_units,
            "silver_reference_pairs": judged_silver_count,
            "unjudged_pairs": unjudged_count,
            "total_reference_evidence_items": total_reference_evidence_items,
            "total_reference_evidence_retrieved": total_reference_evidence_retrieved,
            "reference_evidence_retrieval_rate": round(total_reference_evidence_retrieved / max(1, total_reference_evidence_items), 4),
            "claims_with_at_least_one_ref_retrieved": claims_with_retrieved_refs,
            "pool_depth_per_system": self.pool_depth,
            "contributing_systems": [
                "BM25", "TFIDF", "SVD", "Dense_Transformer",
                "Hybrid_BM25_Dense", "Hybrid_BM25_TFIDF",
                "Cross_Encoder_Rerank", "Query_Expansion_BM25"
            ],
            "exported_files": {
                "csv": output_csv,
                "jsonl": output_jsonl
            }
        }
        
        print(f"\nPooled candidate set constructed successfully!", flush=True)
        print(f"Total candidate pairs: {total_pairs:,} across {unique_claims} claims.")
        print(f"Silver reference pairs: {judged_silver_count:,} | Unjudged pairs: {unjudged_count:,}")
        print(f"Reference evidence retrieval rate: {stats['reference_evidence_retrieval_rate']*100:.2f}%")
        print(f"Files saved: {output_csv} and {output_jsonl}")
        
        return stats

if __name__ == "__main__":
    pooler = CandidatePooler(pool_depth_per_system=20)
    pilot_cids = pooler.select_pilot_claim_ids(sample_size=75, seed=42)
    stats = pooler.build_pooled_benchmark_file(pilot_cids)
    print("\nPooled Benchmark Stats:", json.dumps(stats, indent=2))
