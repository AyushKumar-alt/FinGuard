"""
Evaluates Claim Verification across three systems on the held-out test split:
1. Calibrated Logistic Regression Baseline (TF-IDF + Claim + Evidence)
2. Pretrained Cross-Encoder NLI Verifier (Zero-Shot on Claim + Gold Evidence)
3. Full End-to-End Pipeline (BM25 Retrieval + Cross-Encoder Reranking + NLI Verification)

Saves results to outputs/verification_metrics.json.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.verifier import FinancialClaimVerifier
from src.pipeline import FinGuardPipeline

def run_evaluation():
    test_path = "data/test.csv"
    if not os.path.exists(test_path):
        print(f"Error: {test_path} not found.")
        return

    test_df = pd.read_csv(test_path)
    total_test = len(test_df)
    print(f"Loaded held-out test split: {total_test} claims.")

    # Load existing metrics file if present
    metrics_file = "outputs/verification_metrics.json"
    data = {}
    if os.path.exists(metrics_file):
        with open(metrics_file, "r", encoding="utf-8") as f:
            data = json.load(f)

    if "verification_systems_comparison" not in data:
        data["verification_systems_comparison"] = {}

    # 1. System 1: Calibrated Logistic Regression Baseline (Full Test Set, n=506)
    print("\n--- Evaluating System 1: Calibrated Logistic Regression Baseline (n=506) ---")
    verifier = FinancialClaimVerifier(use_transformer=False)
    lr_preds = []
    for _, row in test_df.iterrows():
        claim = str(row["claim"])
        ev = str(row.get("evidence_text", ""))
        pred_label, conf, _ = verifier.predict_baseline(claim, ev)
        lr_preds.append(pred_label)

    lr_acc = accuracy_score(test_df["label"], lr_preds)
    lr_macro_f1 = f1_score(test_df["label"], lr_preds, average="macro")
    lr_weighted_f1 = f1_score(test_df["label"], lr_preds, average="weighted")
    lr_report = classification_report(test_df["label"], lr_preds, output_dict=True)

    print(f"Logistic Regression: Accuracy = {lr_acc*100:.2f}%, Macro F1 = {lr_macro_f1:.4f}")

    data["verification_systems_comparison"]["1_Logistic_Regression_Baseline"] = {
        "description": "TF-IDF + Calibrated Balanced Logistic Regression on Claim + Gold Evidence context",
        "sample_size": total_test,
        "accuracy": round(float(lr_acc), 4),
        "macro_f1": round(float(lr_macro_f1), 4),
        "weighted_f1": round(float(lr_weighted_f1), 4),
        "per_class": {
            "false": {"f1": round(float(lr_report["false"]["f1-score"]), 4), "support": int(lr_report["false"]["support"])},
            "true": {"f1": round(float(lr_report["true"]["f1-score"]), 4), "support": int(lr_report["true"]["support"])},
            "NEI": {"f1": round(float(lr_report["NEI"]["f1-score"]), 4), "support": int(lr_report["NEI"]["support"])}
        }
    }

    # 2. System 2: Pretrained NLI Verifier (Zero-shot cross-encoder on Gold Evidence)
    print("\n--- Evaluating System 2: Pretrained NLI Verifier on Gold Evidence (n=100 sample) ---")
    nli_verifier = FinancialClaimVerifier(use_transformer=True)
    sample_df = test_df.sample(n=100, random_state=42).copy()
    
    nli_preds = []
    for _, row in sample_df.iterrows():
        claim = str(row["claim"])
        ev = str(row.get("evidence_text", ""))
        res = nli_verifier.verify_claim(claim, raw_evidence_text=ev, engine_choice="transformer_nli")
        nli_preds.append(res["verdict"])

    nli_acc = accuracy_score(sample_df["label"], nli_preds)
    nli_macro_f1 = f1_score(sample_df["label"], nli_preds, average="macro")
    nli_weighted_f1 = f1_score(sample_df["label"], nli_preds, average="weighted")
    nli_report = classification_report(sample_df["label"], nli_preds, output_dict=True, zero_division=0)

    print(f"Pretrained NLI Verifier (Gold Evidence): Accuracy = {nli_acc*100:.2f}%, Macro F1 = {nli_macro_f1:.4f}")

    data["verification_systems_comparison"]["2_Pretrained_NLI_Verifier_Gold_Evidence"] = {
        "description": "Pretrained cross-encoder/nli-distilroberta-base with multi-sentence conflict aggregation on Gold Evidence",
        "sample_size": len(sample_df),
        "accuracy": round(float(nli_acc), 4),
        "macro_f1": round(float(nli_macro_f1), 4),
        "weighted_f1": round(float(nli_weighted_f1), 4),
        "per_class": {
            "false": {"f1": round(float(nli_report["false"]["f1-score"]), 4), "support": int(nli_report["false"]["support"])},
            "true": {"f1": round(float(nli_report["true"]["f1-score"]), 4), "support": int(nli_report["true"]["support"])},
            "NEI": {"f1": round(float(nli_report["NEI"]["f1-score"]), 4), "support": int(nli_report["NEI"]["support"])}
        }
    }

    # 3. System 3: Full End-to-End Pipeline (BM25 + Cross-Encoder Rerank + NLI Verifier)
    print("\n--- Evaluating System 3: Full End-to-End Pipeline (Retrieval + Rerank + NLI, n=100 sample) ---")
    pipeline = FinGuardPipeline(json_path="finfact.json", load_heavy_models=True)
    
    pipe_preds = []
    for _, row in sample_df.iterrows():
        claim = str(row["claim"])
        res = pipeline.run(
            claim=claim,
            retrieval_mode="bm25",
            use_query_expansion=True,
            use_reranker=True,
            use_nli=True,
            top_k_retrieve=15,
            top_k_rerank=4
        )
        pipe_preds.append(res["verdict"])

    pipe_acc = accuracy_score(sample_df["label"], pipe_preds)
    pipe_macro_f1 = f1_score(sample_df["label"], pipe_preds, average="macro")
    pipe_weighted_f1 = f1_score(sample_df["label"], pipe_preds, average="weighted")
    pipe_report = classification_report(sample_df["label"], pipe_preds, output_dict=True, zero_division=0)

    print(f"Full End-to-End Pipeline: Accuracy = {pipe_acc*100:.2f}%, Macro F1 = {pipe_macro_f1:.4f}")

    data["verification_systems_comparison"]["3_Full_End_to_End_Pipeline"] = {
        "description": "Financial Claim Parsing + BM25 Retrieval + ms-marco Reranking + nli-distilroberta Verification (Open-domain fact verification)",
        "sample_size": len(sample_df),
        "accuracy": round(float(pipe_acc), 4),
        "macro_f1": round(float(pipe_macro_f1), 4),
        "weighted_f1": round(float(pipe_weighted_f1), 4),
        "per_class": {
            "false": {"f1": round(float(pipe_report["false"]["f1-score"]), 4), "support": int(pipe_report["false"]["support"])},
            "true": {"f1": round(float(pipe_report["true"]["f1-score"]), 4), "support": int(pipe_report["true"]["support"])},
            "NEI": {"f1": round(float(pipe_report["NEI"]["f1-score"]), 4), "support": int(pipe_report["NEI"]["support"])}
        }
    }

    # Save to file
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"\nSuccessfully wrote updated verification comparison to {metrics_file}!")

if __name__ == "__main__":
    run_evaluation()
