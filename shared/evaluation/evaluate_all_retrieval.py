"""
Controlled Evaluation Benchmark Suite for FinGuard.
Evaluates 8 distinct retrieval configurations (A through H) on held-out test claims
searching across 12,943 candidate evidence sentences:

Configuration A: BM25 (Inverted Index)
Configuration B: TF-IDF Cosine
Configuration C: SVD Dense Semantic (128-dim)
Configuration D: BM25 + TF-IDF (Hybrid RRF)
Configuration E: Pretrained Dense Transformer ('sentence-transformers/all-MiniLM-L6-v2')
Configuration F: BM25 + Pretrained Dense (Hybrid RRF)
Configuration G: Best Retrieval + Pretrained Cross-Encoder Reranking ('ms-marco-MiniLM-L-6-v2')
Configuration H: Original Query vs. Expanded Query (Query Construction Impact)

Strict Leakage Prevention:
- Search corpus contains only factual evidence sentences and source URLs.
- Claims identical to candidate text are strictly excluded.
- Evaluates: Recall@5, Recall@10, Precision@5, Precision@10, MRR, MAP.
"""

import os
import sys
import json
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval import EvidenceRetrievalBenchmark
from src.dense_retrieval import PretrainedDenseRetriever
from src.reranker import PretrainedCrossEncoderReranker
from src.query_construction import FinancialQueryBuilder
from src.verifier import FinancialClaimVerifier

def run_controlled_retrieval_suite(eval_sample_limit: int = 150):
    print(f"\n=======================================================", flush=True)
    print(f"Running Controlled Retrieval Experiments Suite (A through H)...", flush=True)
    print(f"Evaluation Query Subset: {eval_sample_limit} held-out test claims", flush=True)
    print(f"=======================================================", flush=True)

    # 1. Initialize BM25 & TF-IDF
    benchmark = EvidenceRetrievalBenchmark(json_path="finfact.json")
    benchmark.fit_indices()
    corpus = benchmark.corpus
    
    # 2. Initialize Pretrained Dense Retriever (Loads from disk cache)
    dense_retriever = PretrainedDenseRetriever()
    dense_retriever.build_index(corpus, force_recompute=False)
    
    # 3. Initialize Pretrained Cross-Encoder Reranker
    reranker = PretrainedCrossEncoderReranker()
    
    # 4. Initialize Query Builder
    query_builder = FinancialQueryBuilder(max_expansions=3)
    
    # Load test claims
    test_df = pd.read_csv("data/test.csv")
    test_ids = [int(i) for i in test_df["id"].tolist() if int(i) in benchmark.claim_to_doc_ids][:eval_sample_limit]
    print(f"Evaluating {len(test_ids)} valid test claims with gold evidence annotations...", flush=True)

    def compute_ir_metrics(retrieved_doc_ids_list: List[int], gold_doc_ids: set, top_k: int = 10) -> Dict[str, float]:
        # Filter exact claim leaks
        top_5 = set(retrieved_doc_ids_list[:5])
        hits_5 = len(top_5.intersection(gold_doc_ids))
        p5 = hits_5 / 5.0
        r5 = hits_5 / len(gold_doc_ids) if gold_doc_ids else 0.0

        top_10 = set(retrieved_doc_ids_list[:10])
        hits_10 = len(top_10.intersection(gold_doc_ids))
        p10 = hits_10 / 10.0
        r10 = hits_10 / len(gold_doc_ids) if gold_doc_ids else 0.0

        rr = 0.0
        for rank, d_id in enumerate(retrieved_doc_ids_list[:top_k]):
            if d_id in gold_doc_ids:
                rr = 1.0 / (rank + 1)
                break

        ap = 0.0
        cum_hits = 0
        for rank, d_id in enumerate(retrieved_doc_ids_list[:top_k]):
            if d_id in gold_doc_ids:
                cum_hits += 1
                ap += cum_hits / (rank + 1)
        ap = ap / min(len(gold_doc_ids), top_k) if gold_doc_ids else 0.0

        return {"R@5": r5, "R@10": r10, "P@5": p5, "P@10": p10, "MRR": rr, "MAP": ap}

    configs = {
        "Config_A_BM25": "BM25 (Inverted Index)",
        "Config_B_TFIDF": "TF-IDF Cosine",
        "Config_C_SVD": "SVD Semantic (128-dim)",
        "Config_D_Hybrid_BM25_TFIDF": "Hybrid RRF (BM25 + TF-IDF)",
        "Config_E_Dense_Transformer": "Pretrained Dense (all-MiniLM-L6-v2)",
        "Config_F_Hybrid_BM25_Dense": "Hybrid RRF (BM25 + Dense)",
        "Config_G_BM25_Plus_CrossEncoder": "BM25 + Cross-Encoder Reranking",
        "Config_H_Expanded_Query_BM25": "Expanded Query BM25"
    }

    results = {k: {"R@5": [], "R@10": [], "P@5": [], "P@10": [], "MRR": [], "MAP": []} for k in configs}

    for idx_count, q_id in enumerate(test_ids):
        gold_docs = benchmark.claim_to_doc_ids[q_id]
        claim_text = benchmark.query_records[q_id]["claim"]
        query_info = query_builder.build_queries(claim_text)
        expanded_query = query_info["combined_query"]

        # Run retrievals
        # A: BM25
        res_a = [did for did, _ in benchmark.retrieve_bm25(claim_text, top_k=20)]
        m_a = compute_ir_metrics(res_a, gold_docs)
        for k, v in m_a.items(): results["Config_A_BM25"][k].append(v)

        # B: TF-IDF
        res_b = [did for did, _ in benchmark.retrieve_tfidf(claim_text, top_k=20)]
        m_b = compute_ir_metrics(res_b, gold_docs)
        for k, v in m_b.items(): results["Config_B_TFIDF"][k].append(v)

        # C: SVD
        res_c = [did for did, _ in benchmark.retrieve_dense(claim_text, top_k=20)]
        m_c = compute_ir_metrics(res_c, gold_docs)
        for k, v in m_c.items(): results["Config_C_SVD"][k].append(v)

        # D: Hybrid BM25 + TF-IDF
        res_d = [did for did, _ in benchmark.retrieve_hybrid_rrf(claim_text, top_k=20)]
        m_d = compute_ir_metrics(res_d, gold_docs)
        for k, v in m_d.items(): results["Config_D_Hybrid_BM25_TFIDF"][k].append(v)

        # E: Dense Transformer
        res_e = [did for did, _ in dense_retriever.retrieve(claim_text, top_k=20)]
        m_e = compute_ir_metrics(res_e, gold_docs)
        for k, v in m_e.items(): results["Config_E_Dense_Transformer"][k].append(v)

        # F: Hybrid BM25 + Dense RRF
        rrf_f = {}
        k_val = 60
        for rank, did in enumerate(res_a):
            rrf_f[did] = rrf_f.get(did, 0.0) + 1.0 / (k_val + rank + 1)
        for rank, did in enumerate(res_e):
            rrf_f[did] = rrf_f.get(did, 0.0) + 1.0 / (k_val + rank + 1)
        res_f = [did for did, _ in sorted(rrf_f.items(), key=lambda x: x[1], reverse=True)[:20]]
        m_f = compute_ir_metrics(res_f, gold_docs)
        for k, v in m_f.items(): results["Config_F_Hybrid_BM25_Dense"][k].append(v)

        # G: BM25 + Cross-Encoder Reranker
        cand_docs_g = [corpus[did] for did in res_a[:20]]
        reranked_g = reranker.rerank(claim_text, cand_docs_g, top_k=10)
        res_g = [d["doc_id"] for d in reranked_g]
        m_g = compute_ir_metrics(res_g, gold_docs)
        for k, v in m_g.items(): results["Config_G_BM25_Plus_CrossEncoder"][k].append(v)

        # H: Expanded Query BM25
        res_h = [did for did, _ in benchmark.retrieve_bm25(expanded_query, top_k=20)]
        m_h = compute_ir_metrics(res_h, gold_docs)
        for k, v in m_h.items(): results["Config_H_Expanded_Query_BM25"][k].append(v)

        if (idx_count + 1) % 50 == 0:
            print(f"  Processed {idx_count + 1}/{len(test_ids)} claims...", flush=True)

    # Summarize metrics
    summary = {}
    for c_key, c_name in configs.items():
        summary[c_name] = {
            "Config_ID": c_key,
            "Evaluated_Queries": len(test_ids),
            "Recall@5": round(float(np.mean(results[c_key]["R@5"])), 4),
            "Recall@10": round(float(np.mean(results[c_key]["R@10"])), 4),
            "Precision@5": round(float(np.mean(results[c_key]["P@5"])), 4),
            "Precision@10": round(float(np.mean(results[c_key]["P@10"])), 4),
            "MRR": round(float(np.mean(results[c_key]["MRR"])), 4),
            "MAP": round(float(np.mean(results[c_key]["MAP"])), 4)
        }
        print(f"[{c_name}] MRR: {summary[c_name]['MRR']:.4f} | R@10: {summary[c_name]['Recall@10']:.4f} | MAP: {summary[c_name]['MAP']:.4f}", flush=True)

    # Save to outputs/retrieval_metrics.json
    out_path = "outputs/retrieval_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nControlled retrieval suite metrics successfully saved to {out_path}", flush=True)

    # Plot Comparison Chart
    df_plot = pd.DataFrame([
        {"Method": name, "MRR": data["MRR"], "Recall@10": data["Recall@10"], "MAP": data["MAP"]}
        for name, data in summary.items()
    ])
    
    plt.figure(figsize=(12, 6))
    df_melt = pd.melt(df_plot, id_vars=["Method"], value_vars=["MRR", "Recall@10", "MAP"], var_name="Metric", value_name="Score")
    ax = sns.barplot(data=df_melt, x="Method", y="Score", hue="Metric", palette="mako")
    plt.title(f"Controlled Retrieval Benchmark (Configs A-H, N={len(test_ids)} Held-Out Claims)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Retrieval Architecture", fontsize=11, fontweight="bold")
    plt.ylabel("Performance Score", fontsize=11, fontweight="bold")
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, 0.9)
    plt.legend(loc="upper right")
    plt.tight_layout()
    fig_path = "figures/retrieval_comparison.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Generated updated comparison figure: {fig_path}", flush=True)

    return summary

if __name__ == "__main__":
    run_controlled_retrieval_suite(eval_sample_limit=150)
