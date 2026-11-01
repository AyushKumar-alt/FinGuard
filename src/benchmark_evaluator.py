"""
FinGuard Shared Retrieval Benchmark Evaluator.

Single, authoritative evaluation harness evaluating all 8 retrieval systems against
the exact same claims, candidate corpus, and frozen benchmark judgments.

Metrics Computed:
- Recall@1, @5, @10, @20
- Precision@5, @10
- MRR (Mean Reciprocal Rank)
- MAP (Mean Average Precision - standard TREC formula)
- nDCG@5, @10 (Graded relevance: 2^g - 1 / log2(rank+1))
- Judged Coverage at @5, @10, @20
- Reference Evidence Retrieval Rate @10, @20
- Latency (mean and median in ms)
- Paired bootstrap 95% confidence intervals over claims
"""

import os
import sys
import json
import math
import time
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gold_benchmark_schema import format_evidence_id, parse_evidence_id
from src.retrieval import EvidenceRetrievalBenchmark
from src.dense_retrieval import PretrainedDenseRetriever
from src.reranker import PretrainedCrossEncoderReranker
from src.query_construction import FinancialQueryBuilder

class BenchmarkEvaluator:
    def __init__(self,
                 config_path: str = "configs/gold_evidence_benchmark.json"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.binary_threshold = self.config.get("binary_relevance_threshold", 2)
        
        # Load frozen manifests
        with open(self.config["claims_manifest_path"], "r", encoding="utf-8") as f:
            self.claims = json.load(f)
            
        with open(self.config["reference_evidence_path"], "r", encoding="utf-8") as f:
            self.reference_evidence = json.load(f)
            
        with open(self.config["relevance_judgments_path"], "r", encoding="utf-8") as f:
            judgments_list = json.load(f)

        # Build judgment lookup: (claim_id, evidence_id) -> record
        self.judgments: Dict[Tuple[str, str], Dict[str, Any]] = {}
        # Also build per-claim relevant sets: claim_id -> {evidence_id: grade}
        self.claim_relevance: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.claim_judged_set: Dict[str, set] = defaultdict(set)
        all_pool_claims = set()

        for d in judgments_list:
            cid = d["claim_id"]
            eid = d["evidence_id"]
            self.judgments[(cid, eid)] = d
            all_pool_claims.add(cid)
            grade = d.get("relevance_grade")
            # AUDIT FIX: Only add to judged set if record has an explicit, non-null judgment
            if grade is not None:
                self.claim_judged_set[cid].add(eid)
                if int(grade) >= self.binary_threshold:
                    self.claim_relevance[cid][eid] = int(grade)

        # Evaluated claim IDs
        self.eval_claim_ids = self.config.get("evaluation_claim_ids", sorted(list(all_pool_claims)))
        print(f"Evaluator initialized with {len(self.eval_claim_ids)} evaluation claims and binary relevance threshold >= {self.binary_threshold}.", flush=True)

        # Initialize retrieval engines
        self._init_retrievers()

    def _init_retrievers(self):
        print("Initializing retrieval benchmark engines...", flush=True)
        self.lexical_benchmark = EvidenceRetrievalBenchmark(json_path="finfact.json")
        self.lexical_benchmark.fit_indices()
        self.corpus = self.lexical_benchmark.corpus

        self.dense_retriever = PretrainedDenseRetriever()
        self.dense_retriever.build_index(self.corpus, force_recompute=False)

        self.reranker = PretrainedCrossEncoderReranker()
        self.query_builder = FinancialQueryBuilder(max_expansions=3)

    def compute_single_query_metrics(self,
                                     retrieved_eids: List[str],
                                     claim_id: str,
                                     top_k: int = 20) -> Dict[str, float]:
        """
        Computes exact IR metrics for a single claim ranking.
        """
        rel_map = self.claim_relevance.get(claim_id, {})
        judged_set = self.claim_judged_set.get(claim_id, set())
        ref_eids = {r["evidence_id"] for r in self.reference_evidence.get(claim_id, [])}
        
        num_rel = len(rel_map)
        retrieved_k = retrieved_eids[:top_k]
        
        # 1. Binary relevance indicator list
        rel_binary = [1 if (eid in rel_map and rel_map[eid] >= self.binary_threshold) else 0 for eid in retrieved_k]
        # Graded relevance list
        rel_graded = [rel_map.get(eid, 0) for eid in retrieved_k]
        # Judged indicator list
        judged_flags = [1 if eid in judged_set else 0 for eid in retrieved_k]

        # Precision @ K
        p5 = sum(rel_binary[:5]) / 5.0
        p10 = sum(rel_binary[:10]) / 10.0

        # Recall @ K
        if num_rel > 0:
            r1 = sum(rel_binary[:1]) / float(num_rel)
            r5 = sum(rel_binary[:5]) / float(num_rel)
            r10 = sum(rel_binary[:10]) / float(num_rel)
            r20 = sum(rel_binary[:20]) / float(num_rel)
        else:
            r1 = r5 = r10 = r20 = 0.0

        # MRR (Mean Reciprocal Rank)
        rr = 0.0
        for rank, is_rel in enumerate(rel_binary):
            if is_rel:
                rr = 1.0 / (rank + 1)
                break

        # MAP (Mean Average Precision - Standard TREC formulation)
        ap = 0.0
        if num_rel > 0:
            cum_hits = 0
            for rank, is_rel in enumerate(rel_binary):
                if is_rel:
                    cum_hits += 1
                    prec_at_rank = cum_hits / (rank + 1)
                    ap += prec_at_rank
            ap = ap / float(num_rel)

        # nDCG @ 5 and @ 10
        ndcg5 = self._compute_ndcg(rel_graded[:5], list(rel_map.values()), k=5)
        ndcg10 = self._compute_ndcg(rel_graded[:10], list(rel_map.values()), k=10)

        # Judged Coverage @ 5, @ 10, @ 20
        # True fraction of retrieved items having an explicit, non-null judgment
        cov5 = sum(judged_flags[:5]) / 5.0
        cov10 = sum(judged_flags[:10]) / 10.0
        cov20 = sum(judged_flags[:20]) / 20.0

        # bpref (Buckley & Voorhees 2004) - Incomplete information measure
        bpref = self._compute_bpref(retrieved_k, rel_map, judged_set)

        # Reference evidence retrieval rate
        if ref_eids:
            hits_ref_10 = len(set(retrieved_k[:10]).intersection(ref_eids))
            hits_ref_20 = len(set(retrieved_k[:20]).intersection(ref_eids))
            ref_rate_10 = hits_ref_10 / float(len(ref_eids))
            ref_rate_20 = hits_ref_20 / float(len(ref_eids))
        else:
            ref_rate_10 = ref_rate_20 = 0.0

        return {
            "R@1": r1,
            "R@5": r5,
            "R@10": r10,
            "R@20": r20,
            "P@5": p5,
            "P@10": p10,
            "MRR": rr,
            "MAP": ap,
            "nDCG@5": ndcg5,
            "nDCG@10": ndcg10,
            "bpref": bpref,
            "Coverage@5": cov5,
            "Coverage@10": cov10,
            "Coverage@20": cov20,
            "RefRate@10": ref_rate_10,
            "RefRate@20": ref_rate_20
        }

    def _compute_ndcg(self, retrieved_grades: List[int], all_relevant_grades: List[int], k: int) -> float:
        """
        Computes Normalized Discounted Cumulative Gain at rank K.
        DCG = sum_{i=1}^k (2^{rel_i} - 1) / log2(i + 1)
        """
        dcg = 0.0
        for i, g in enumerate(retrieved_grades[:k]):
            if g > 0:
                dcg += (math.pow(2, g) - 1.0) / math.log2(i + 2)

        # Compute Ideal DCG (IDCG)
        sorted_ideal = sorted(all_relevant_grades, reverse=True)[:k]
        idcg = 0.0
        for i, g in enumerate(sorted_ideal):
            if g > 0:
                idcg += (math.pow(2, g) - 1.0) / math.log2(i + 2)

        if idcg == 0.0:
            return 0.0
        return dcg / idcg

    def _compute_bpref(self, retrieved_eids: List[str], rel_map: Dict[str, int], judged_set: set) -> float:
        """
        Computes binary preference (bpref) as defined by Buckley & Voorhees (SIGIR 2004):
        bpref = (1 / R) * sum_{r in R} (1 - (num non-relevant ranked ahead of r) / min(R, N))
        where R is total known relevant items for this query,
        N is total known non-relevant items judged for this query.
        Unjudged documents are completely excluded from the calculation.
        """
        num_rel = len(rel_map)
        if num_rel == 0:
            return 0.0

        num_nonrel_total = len(judged_set) - num_rel
        if num_nonrel_total == 0:
            ret_rel_count = sum(1 for eid in retrieved_eids if eid in rel_map)
            return ret_rel_count / float(num_rel)

        denom = min(num_rel, num_nonrel_total)
        bpref_sum = 0.0
        nonrel_seen = 0

        for eid in retrieved_eids:
            if eid in rel_map:
                bpref_sum += max(0.0, 1.0 - (nonrel_seen / float(denom)))
            elif eid in judged_set:
                nonrel_seen += 1
            # Unjudged items are intentionally skipped without penalty

        return bpref_sum / float(num_rel)

    def evaluate_system(self, system_name: str, top_k: int = 20) -> Dict[str, Any]:
        """
        Evaluates a specified retrieval system across all evaluation claims.
        """
        print(f"\n--- Evaluating System: {system_name} on {len(self.eval_claim_ids)} claims ---", flush=True)
        query_metrics: Dict[str, List[float]] = defaultdict(list)
        latencies_ms: List[float] = []

        claims_with_rel = 0
        claims_zero_rel = 0

        for cid in self.eval_claim_ids:
            claim_text = self.claims[cid]["claim_text"]
            num_rel = len(self.claim_relevance.get(cid, {}))
            if num_rel > 0:
                claims_with_rel += 1
            else:
                claims_zero_rel += 1

            start_t = time.perf_counter()
            retrieved_eids = self._retrieve_for_system(system_name, claim_text, top_k=top_k)
            latency = (time.perf_counter() - start_t) * 1000.0
            latencies_ms.append(latency)

            # Compute metrics
            m = self.compute_single_query_metrics(retrieved_eids, cid, top_k=top_k)
            for k, v in m.items():
                query_metrics[k].append(v)

        # Macro averages
        summary = {
            "System": system_name,
            "Evaluated_Claims": len(self.eval_claim_ids),
            "Claims_With_Relevant_Evidence": claims_with_rel,
            "Claims_Zero_Relevant": claims_zero_rel,
            "Latency_Mean_ms": round(float(np.mean(latencies_ms)), 2),
            "Latency_Median_ms": round(float(np.median(latencies_ms)), 2)
        }

        for metric_name, values in query_metrics.items():
            mean_val = float(np.mean(values))
            summary[metric_name] = round(mean_val, 4)
            # Compute 95% bootstrap confidence interval
            ci_low, ci_high = self._bootstrap_ci(values, n_boot=500, ci=0.95)
            summary[f"{metric_name}_CI95"] = [round(ci_low, 4), round(ci_high, 4)]

        print(f"[{system_name}] MRR: {summary['MRR']:.4f} (95% CI: {summary['MRR_CI95']}) | "
              f"R@10: {summary['R@10']:.4f} | nDCG@10: {summary['nDCG@10']:.4f} | "
              f"P@10: {summary['P@10']:.4f} | Latency: {summary['Latency_Mean_ms']}ms", flush=True)

        return summary

    def _bootstrap_ci(self, values: List[float], n_boot: int = 500, ci: float = 0.95) -> Tuple[float, float]:
        """Computes bootstrap confidence interval over claim scores."""
        arr = np.array(values)
        n = len(arr)
        if n == 0:
            return 0.0, 0.0
        boot_means = []
        rng = np.random.RandomState(42)
        for _ in range(n_boot):
            sample = rng.choice(arr, size=n, replace=True)
            boot_means.append(np.mean(sample))
        alpha = (1.0 - ci) / 2.0
        low = float(np.percentile(boot_means, alpha * 100))
        high = float(np.percentile(boot_means, (1.0 - alpha) * 100))
        return low, high

    def _retrieve_for_system(self, system_name: str, claim_text: str, top_k: int = 20) -> List[str]:
        """Routes query to the specified retrieval architecture and returns canonical evidence_ids."""
        q_clean = claim_text.strip()
        
        if system_name == "BM25":
            ranked = self.lexical_benchmark.retrieve_bm25(q_clean, top_k=top_k)
            return [format_evidence_id(did) for did, _ in ranked]
            
        elif system_name == "TFIDF":
            ranked = self.lexical_benchmark.retrieve_tfidf(q_clean, top_k=top_k)
            return [format_evidence_id(did) for did, _ in ranked]
            
        elif system_name == "SVD":
            ranked = self.lexical_benchmark.retrieve_dense(q_clean, top_k=top_k)
            return [format_evidence_id(did) for did, _ in ranked]
            
        elif system_name == "Dense_Transformer":
            if self.dense_retriever and self.dense_retriever.is_available:
                ranked = self.dense_retriever.retrieve(q_clean, top_k=top_k)
                return [format_evidence_id(did) for did, _ in ranked]
            return []
            
        elif system_name == "Hybrid_BM25_Dense":
            bm25_ranks = self.lexical_benchmark.retrieve_bm25(q_clean, top_k=top_k * 2)
            dense_ranks = self.dense_retriever.retrieve(q_clean, top_k=top_k * 2) if self.dense_retriever else []
            rrf = defaultdict(float)
            k_val = 60
            for rank, (did, _) in enumerate(bm25_ranks):
                rrf[did] += 1.0 / (k_val + rank + 1)
            for rank, (did, _) in enumerate(dense_ranks):
                rrf[did] += 1.0 / (k_val + rank + 1)
            ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
            return [format_evidence_id(did) for did, _ in ranked]
            
        elif system_name == "Hybrid_BM25_TFIDF":
            ranked = self.lexical_benchmark.retrieve_hybrid_rrf(q_clean, top_k=top_k)
            return [format_evidence_id(did) for did, _ in ranked]
            
        elif system_name == "Cross_Encoder_Rerank":
            # Retrieve initial candidate set via BM25, then rerank with ms-marco
            initial_bm25 = self.lexical_benchmark.retrieve_bm25(q_clean, top_k=top_k)
            cand_docs = [self.lexical_benchmark.corpus[did] for did, _ in initial_bm25]
            if self.reranker and self.reranker.is_available and cand_docs:
                reranked = self.reranker.rerank(q_clean, cand_docs, top_k=top_k)
                return [format_evidence_id(d["doc_id"]) for d in reranked]
            return [format_evidence_id(did) for did, _ in initial_bm25]
            
        elif system_name == "Query_Expansion_BM25":
            q_info = self.query_builder.build_queries(q_clean)
            exp_query = q_info["combined_query"]
            ranked = self.lexical_benchmark.retrieve_bm25(exp_query, top_k=top_k)
            return [format_evidence_id(did) for did, _ in ranked]
            
        else:
            raise ValueError(f"Unknown retrieval system: {system_name}")

    def evaluate_all_systems(self,
                             output_json: str = "outputs/gold_benchmark_results.json",
                             output_md: str = "outputs/gold_benchmark_results.md") -> List[Dict[str, Any]]:
        """
        Runs evaluation across all declared systems and saves results.
        """
        systems = self.config.get("retrieval_systems", [
            "BM25", "TFIDF", "SVD", "Dense_Transformer",
            "Hybrid_BM25_Dense", "Hybrid_BM25_TFIDF",
            "Cross_Encoder_Rerank", "Query_Expansion_BM25"
        ])

        all_results = []
        for sys_name in systems:
            res = self.evaluate_system(sys_name, top_k=20)
            all_results.append(res)

        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

        # Generate Research Report Table
        df = pd.DataFrame(all_results)
        cols_for_table = [
            "System", "MRR", "R@5", "R@10", "nDCG@5", "nDCG@10", "bpref", "P@5", "P@10", "Coverage@10", "RefRate@10", "Latency_Mean_ms"
        ]
        sub_df = df[[c for c in cols_for_table if c in df.columns]]

        md_content = f"# FinGuard Gold-Evidence Relevance Benchmark Results\n\n"
        md_content += f"**Benchmark Version:** {self.config.get('version', '1.0.0')}  \n"
        md_content += f"**Evaluation Claims:** {len(self.eval_claim_ids)} held-out claims  \n"
        md_content += f"**Relevance Threshold:** Grade >= {self.binary_threshold} (Grades 2 & 3 counted as relevant)  \n"
        md_content += f"**Candidate Corpus:** 12,941 candidate evidence sentences  \n\n"
        md_content += "### Retrieval Performance Matrix\n\n"
        md_content += sub_df.to_markdown(index=False)
        md_content += "\n\n### Metric Explanations\n"
        md_content += "- **MRR:** Mean Reciprocal Rank of first relevant item (Grade >= 2).\n"
        md_content += "- **R@K:** Recall at rank K relative to total relevant items for the claim.\n"
        md_content += "- **nDCG@K:** Normalized Discounted Cumulative Gain using graded gains (2^g - 1).\n"
        md_content += "- **bpref:** Binary Preference measure (Buckley & Voorhees 2004) under incomplete relevance judgments.\n"
        md_content += "- **P@K:** Precision at rank K.\n"
        md_content += "- **Coverage@K:** Fraction of retrieved top-K items having explicit, non-null judgments.\n"
        md_content += "- **RefRate@K:** Fraction of original dataset reference evidence retrieved in top-K.\n"
        md_content += "- **Latency:** End-to-end retrieval time in milliseconds.\n"

        with open(output_md, "w", encoding="utf-8") as f:
            f.write(md_content)

        print(f"\nEvaluation complete! Results saved to {output_json} and {output_md}", flush=True)
        return all_results

    def evaluate_sensitivity_analysis(self, top_k: int = 20) -> pd.DataFrame:
        """
        Conducts sensitivity analysis comparing all systems across three evaluation regimes:
        1. Conventional Cranfield: Unjudged candidates treated as non-relevant (Grade 0).
        2. Judged-Only: Unjudged candidates filtered out prior to ranking evaluation.
        3. Incomplete Information: bpref (Buckley & Voorhees 2004).
        """
        print(f"\n=== Incomplete-Information Sensitivity Analysis (Top-{top_k}) ===", flush=True)
        systems = self.config.get("retrieval_systems", [
            "BM25", "TFIDF", "SVD", "Dense_Transformer",
            "Hybrid_BM25_Dense", "Hybrid_BM25_TFIDF",
            "Cross_Encoder_Rerank", "Query_Expansion_BM25"
        ])

        rows = []
        for sys_name in systems:
            cranfield_mrrs = []
            judged_only_mrrs = []
            bprefs = []

            for cid in self.eval_claim_ids:
                claim_text = self.claims[cid]["claim_text"]
                retrieved_eids = self._retrieve_for_system(sys_name, claim_text, top_k=top_k)
                rel_map = self.claim_relevance.get(cid, {})
                judged_set = self.claim_judged_set.get(cid, set())

                # Regime 1: Conventional Cranfield MRR
                rr_cran = 0.0
                for rank, eid in enumerate(retrieved_eids):
                    if eid in rel_map and rel_map[eid] >= self.binary_threshold:
                        rr_cran = 1.0 / (rank + 1)
                        break
                cranfield_mrrs.append(rr_cran)

                # Regime 2: Judged-Only MRR (skip unjudged)
                judged_retrieved = [eid for eid in retrieved_eids if eid in judged_set]
                rr_judged = 0.0
                for rank, eid in enumerate(judged_retrieved):
                    if eid in rel_map and rel_map[eid] >= self.binary_threshold:
                        rr_judged = 1.0 / (rank + 1)
                        break
                judged_only_mrrs.append(rr_judged)

                # Regime 3: bpref
                bp = self._compute_bpref(retrieved_eids, rel_map, judged_set)
                bprefs.append(bp)

            rows.append({
                "System": sys_name,
                "MRR_Cranfield": round(float(np.mean(cranfield_mrrs)), 4),
                "MRR_JudgedOnly": round(float(np.mean(judged_only_mrrs)), 4),
                "bpref": round(float(np.mean(bprefs)), 4)
            })

        df_sens = pd.DataFrame(rows)
        print("\nSensitivity Analysis Matrix:")
        print(df_sens.to_string(index=False))
        return df_sens

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FinGuard Benchmark Evaluator")
    parser.add_argument("--config", type=str, default="configs/gold_evidence_benchmark.json")
    parser.add_argument("--sensitivity", action="store_true", help="Run incomplete information sensitivity analysis")
    parser.add_argument("--output-json", type=str, default=None, help="Path to save evaluation results JSON")
    parser.add_argument("--output-md", type=str, default=None, help="Path to save evaluation report Markdown")
    args = parser.parse_args()

    evaluator = BenchmarkEvaluator(config_path=args.config)
    ver = evaluator.config.get("version", "1.0.0")
    default_json = f"outputs/gold_benchmark_results_v{ver.split('.')[0]}.json" if ver != "1.0.0" else "outputs/gold_benchmark_results.json"
    default_md = f"outputs/gold_benchmark_results_v{ver.split('.')[0]}.md" if ver != "1.0.0" else "outputs/gold_benchmark_results.md"
    
    out_json = args.output_json or default_json
    out_md = args.output_md or default_md

    evaluator.evaluate_all_systems(output_json=out_json, output_md=out_md)
    if args.sensitivity:
        evaluator.evaluate_sensitivity_analysis()
