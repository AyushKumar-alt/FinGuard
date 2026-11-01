"""
Benchmark Freezer & Manifest Generator for FinGuard Gold Relevance Benchmark.

Computes cryptographic SHA-256 checksums across all versioned benchmark files,
records corpus fingerprints, evaluation claim IDs, and exports immutable manifests:
1. data/benchmark/v1/benchmark_manifest.json
2. configs/gold_evidence_benchmark.json
"""

import os
import sys
import json
import hashlib
from datetime import datetime
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gold_benchmark_schema import BENCHMARK_VERSION

def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def freeze_benchmark_artifacts(benchmark_dir: str = "data/benchmark/v1",
                               config_path: str = "configs/gold_evidence_benchmark.json"):
    files_to_hash = [
        "claims_manifest.json",
        "corpus_manifest.json",
        "reference_evidence.json",
        "pooled_candidates_annotation.csv",
        "pooled_candidates_annotation.jsonl",
        "relevance_judgments.json",
        "relevance_judgments.csv"
    ]

    checksums: Dict[str, str] = {}
    file_sizes: Dict[str, int] = {}

    for fname in files_to_hash:
        fpath = os.path.join(benchmark_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing required benchmark artifact: {fpath}")
        checksums[fname] = compute_file_sha256(fpath)
        file_sizes[fname] = os.path.getsize(fpath)

    # Load claims and judgments to get summary metrics
    with open(os.path.join(benchmark_dir, "claims_manifest.json"), "r", encoding="utf-8") as f:
        claims = json.load(f)

    with open(os.path.join(benchmark_dir, "relevance_judgments.json"), "r", encoding="utf-8") as f:
        judgments = json.load(f)

    # Pilot evaluation claim IDs
    eval_cids = sorted(list(set(d["claim_id"] for d in judgments)))
    silver_count = sum(1 for d in judgments if d["judgment_status"] == "silver_reference")
    unjudged_count = sum(1 for d in judgments if d["judgment_status"] == "unjudged")
    reviewed_count = sum(1 for d in judgments if d["judgment_status"] in ("reviewed", "adjudicated"))

    manifest: Dict[str, Any] = {
        "benchmark_name": "FinGuard Gold-Evidence Relevance Benchmark",
        "benchmark_version": BENCHMARK_VERSION,
        "frozen_timestamp": datetime.utcnow().isoformat() + "Z",
        "split_configuration": {
            "evaluation_split": "held_out_test_pilot",
            "evaluated_claims_count": len(eval_cids),
            "stratified_sample_seed": 42,
            "source_test_dataset": "data/test.csv (506 claims total)"
        },
        "corpus_statistics": {
            "total_claims_in_finfact": len(claims),
            "canonical_candidate_sentences": 12941,
            "total_candidate_pairs_in_pool": len(judgments),
            "silver_reference_evidence_pairs": silver_count,
            "unjudged_candidate_pairs": unjudged_count,
            "reviewed_or_adjudicated_pairs": reviewed_count
        },
        "relevance_grading_scale": {
            "3": "Directly decisive evidence (strongly supports or refutes assertion)",
            "2": "Substantially relevant evidence (meaningful factual support/contradiction)",
            "1": "Contextually relevant evidence (background context)",
            "0": "Judged not relevant (reviewed and confirmed unhelpful)",
            "None": "Unjudged (unknown relevance, preserved explicitly)"
        },
        "evaluation_claim_ids": eval_cids,
        "artifact_checksums": checksums,
        "artifact_file_sizes_bytes": file_sizes
    }

    manifest_path = os.path.join(benchmark_dir, "benchmark_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Export main configuration file
    config_dict = {
        "benchmark_name": "FinGuard Gold-Evidence Relevance Benchmark",
        "version": BENCHMARK_VERSION,
        "manifest_path": manifest_path,
        "claims_manifest_path": os.path.join(benchmark_dir, "claims_manifest.json"),
        "corpus_manifest_path": os.path.join(benchmark_dir, "corpus_manifest.json"),
        "reference_evidence_path": os.path.join(benchmark_dir, "reference_evidence.json"),
        "relevance_judgments_path": os.path.join(benchmark_dir, "relevance_judgments.json"),
        "pooled_candidates_path": os.path.join(benchmark_dir, "pooled_candidates_annotation.csv"),
        "pool_depth_per_system": 20,
        "evaluation_sample_size": len(eval_cids),
        "random_seed": 42,
        "binary_relevance_threshold": 2,  # Grade >= 2 is relevant (Grades 2 & 3)
        "graded_relevance_max": 3,
        "retrieval_systems": [
            "BM25",
            "TFIDF",
            "SVD",
            "Dense_Transformer",
            "Hybrid_BM25_Dense",
            "Hybrid_BM25_TFIDF",
            "Cross_Encoder_Rerank",
            "Query_Expansion_BM25"
        ],
        "checksums": checksums
    }

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    print(f"\nFrozen benchmark manifest saved to {manifest_path}!", flush=True)
    print(f"Benchmark configuration saved to {config_path}!", flush=True)
    print(f"Total evaluated claims: {len(eval_cids)} | Total candidate pairs: {len(judgments)}")
    for fname, csum in checksums.items():
        print(f"  - {fname}: {csum[:16]}... ({file_sizes[fname]:,} bytes)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Freeze Benchmark Artifacts")
    parser.add_argument("--dir", type=str, default="data/benchmark/v1", help="Benchmark directory")
    parser.add_argument("--config", type=str, default="configs/gold_evidence_benchmark.json", help="Config output path")
    parser.add_argument("--version", type=str, default=None, help="Benchmark version string")
    args = parser.parse_args()

    freeze_benchmark_artifacts(benchmark_dir=args.dir, config_path=args.config)
