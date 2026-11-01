"""
FinGuard Phase 2: End-to-End Claim Verification Evaluation.

Evaluates and compares on the exact same 675 source-grouped held-out test claims:
- System A: TF-IDF + Logistic Regression Baseline
- System B: BM25 + Cross-Encoder Reranker + Zero-Shot NLI (Default Aggregation)
- System C: BM25 + Cross-Encoder Reranker + Financially Fine-Tuned NLI (Validation-Selected Aggregation)

Uses identical retrieval and Cross-Encoder candidates between B and C to isolate
the causal impact of domain fine-tuning.
"""

import os
import sys
import json
import logging
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any

# Ensure offline execution
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("evaluate_end_to_end")

LABELS = ["false", "true", "NEI"]

def load_system_predictions(filepath: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(filepath)
    sys_a = df[df["system"] == "A"].copy()
    sys_b = df[df["system"] == "B"].copy()
    return sys_a, sys_b

def compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    per_p, per_r, per_f1, per_supp = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()
    counts = Counter(y_pred)

    return {
        "accuracy": round(float(acc) * 100.0, 2),
        "macro_f1": round(float(macro_f1) * 100.0, 2),
        "weighted_f1": round(float(weighted_f1) * 100.0, 2),
        "per_class": {
            "false": {
                "precision": round(float(per_p[0]) * 100.0, 2),
                "recall": round(float(per_r[0]) * 100.0, 2),
                "f1": round(float(per_f1[0]) * 100.0, 2),
                "support": int(per_supp[0])
            },
            "true": {
                "precision": round(float(per_p[1]) * 100.0, 2),
                "recall": round(float(per_r[1]) * 100.0, 2),
                "f1": round(float(per_f1[1]) * 100.0, 2),
                "support": int(per_supp[1])
            },
            "NEI": {
                "precision": round(float(per_p[2]) * 100.0, 2),
                "recall": round(float(per_r[2]) * 100.0, 2),
                "f1": round(float(per_f1[2]) * 100.0, 2),
                "support": int(per_supp[2])
            }
        },
        "predicted_distribution": {
            "false": counts.get("false", 0),
            "true": counts.get("true", 0),
            "NEI": counts.get("NEI", 0)
        },
        "confusion_matrix": {
            "labels": LABELS,
            "matrix": cm
        }
    }

def main():
    logger.info("Starting End-to-End Evaluation on 675 Held-Out Test Claims...")

    pred_csv_path = ROOT / "outputs" / "experiments" / "aligned_grouped_verification" / "per_claim_predictions.csv"
    sys_a_df, sys_b_df = load_system_predictions(pred_csv_path)

    logger.info("Loaded System A (%d claims) and System B (%d claims).", len(sys_a_df), len(sys_b_df))

    # Load fine-tuned NLI model
    finetuned_dir = ROOT / "models" / "finetuned_nli"
    logger.info("Loading fine-tuned NLI model from %s on cpu...", finetuned_dir)
    from src.verifier import FinancialClaimVerifier
    finetuned_verifier = FinancialClaimVerifier(
        nli_model_name=str(finetuned_dir),
        use_transformer=True,
        device="cpu"
    )

    # Load validation-tuned aggregation thresholds
    val_tuning_path = ROOT / "phase2" / "results" / "financial_nli" / "validation_threshold_tuning.json"
    if val_tuning_path.exists():
        with open(val_tuning_path, "r", encoding="utf-8") as f:
            val_tuning = json.load(f)
        t_entail = val_tuning["best_thresholds"]["entailment_threshold"]
        t_contra = val_tuning["best_thresholds"]["contradiction_threshold"]
    else:
        t_entail = 0.50
        t_contra = 0.65
    logger.info("Using validation-selected aggregation thresholds: T_entail=%.2f, T_contra=%.2f", t_entail, t_contra)

    sys_c_rows = []
    y_true = []
    y_pred_c = []

    for idx, row in sys_b_df.iterrows():
        claim_id = row["claim_id"]
        claim = row["claim"]
        gold_label = row["gold_label"]
        y_true.append(gold_label)

        # Parse exact reranked candidates from System B
        raw_reranked = row["reranked_candidates"]
        reranked_candidates = json.loads(raw_reranked) if pd.notnull(raw_reranked) else []
        top4_docs = reranked_candidates[:4]

        # Score with Fine-Tuned NLI
        supporting = []
        contradicting = []
        neutral = []
        all_records = []

        for doc in top4_docs:
            text = doc.get("text", "")
            if not text.strip():
                continue
            scores = finetuned_verifier.predict_nli_single_pair(claim, text)
            item_rec = {
                "doc_id": doc.get("doc_id"),
                "text": text,
                "entailment_score": scores["entailment"],
                "contradiction_score": scores["contradiction"],
                "neutral_score": scores["neutral"]
            }
            all_records.append(item_rec)

            if scores["contradiction"] >= 0.50:
                contradicting.append(item_rec)
            elif scores["entailment"] >= 0.50:
                supporting.append(item_rec)
            else:
                neutral.append(item_rec)

        # Apply Validation-Tuned Aggregation Logic
        max_contra = max([e["contradiction_score"] for e in all_records] or [0.0])
        max_entail = max([e["entailment_score"] for e in all_records] or [0.0])
        has_conflict = len(supporting) > 0 and len(contradicting) > 0

        if not all_records:
            pred_verdict = "NEI"
            confidence = 0.50
        elif has_conflict:
            pred_verdict = "NEI"
            confidence = 0.55
        elif contradicting and max_contra >= t_contra:
            pred_verdict = "false"
            confidence = max_contra
        elif supporting and max_entail >= t_entail:
            pred_verdict = "true"
            confidence = max_entail
        else:
            pred_verdict = "NEI"
            confidence = 0.50

        y_pred_c.append(pred_verdict)

        sys_c_rows.append({
            "claim_id": claim_id,
            "source_url": row["source_url"],
            "claim": claim,
            "gold_label": gold_label,
            "system": "C_finetuned",
            "prediction": pred_verdict,
            "confidence": round(float(confidence), 4),
            "status": "ok",
            "supporting_evidence_count": len(supporting),
            "contradicting_evidence_count": len(contradicting),
            "neutral_evidence_count": len(neutral),
            "max_entailment": round(float(max_entail), 4),
            "max_contradiction": round(float(max_contra), 4)
        })

    # Save System C detailed CSV
    out_dir = ROOT / "phase2" / "results" / "financial_nli"
    out_dir.mkdir(parents=True, exist_ok=True)
    c_df = pd.DataFrame(sys_c_rows)
    c_df.to_csv(out_dir / "system_c_predictions.csv", index=False)
    logger.info("Saved System C predictions to %s", out_dir / "system_c_predictions.csv")

    # Metrics
    metrics_a = compute_metrics(sys_a_df["gold_label"].tolist(), sys_a_df["prediction"].tolist())
    metrics_b = compute_metrics(sys_b_df["gold_label"].tolist(), sys_b_df["prediction"].tolist())
    metrics_c = compute_metrics(y_true, y_pred_c)

    comparison = {
        "evaluation_name": "FinGuard 675 Held-Out Claims Verification Comparison",
        "benchmark_claims_count": len(y_true),
        "systems": {
            "System_A_TFIDF_Baseline": metrics_a,
            "System_B_ZeroShot_Pipeline": metrics_b,
            "System_C_FineTuned_Pipeline": metrics_c
        },
        "deltas_c_vs_b": {
            "accuracy_delta": round(metrics_c["accuracy"] - metrics_b["accuracy"], 2),
            "macro_f1_delta": round(metrics_c["macro_f1"] - metrics_b["macro_f1"], 2),
            "weighted_f1_delta": round(metrics_c["weighted_f1"] - metrics_b["weighted_f1"], 2),
            "true_recall_delta": round(metrics_c["per_class"]["true"]["recall"] - metrics_b["per_class"]["true"]["recall"], 2),
            "true_f1_delta": round(metrics_c["per_class"]["true"]["f1"] - metrics_b["per_class"]["true"]["f1"], 2),
            "false_f1_delta": round(metrics_c["per_class"]["false"]["f1"] - metrics_b["per_class"]["false"]["f1"], 2),
            "nei_f1_delta": round(metrics_c["per_class"]["NEI"]["f1"] - metrics_b["per_class"]["NEI"]["f1"], 2)
        }
    }

    with open(out_dir / "end_to_end_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    logger.info("Saved end-to-end comparison to %s", out_dir / "end_to_end_comparison.json")

    # Print clean terminal report
    print("\n" + "=" * 80)
    print("FINGUARD 675 HELD-OUT CLAIMS: SCIENTIFIC VERIFICATION BENCHMARK")
    print("=" * 80)
    print(f"{'Metric':<24} | {'System A (Baseline)':<20} | {'System B (Zero-Shot)':<20} | {'System C (Fine-Tuned)':<20}")
    print("-" * 80)
    print(f"{'Overall Accuracy':<24} | {metrics_a['accuracy']:>18.2f}% | {metrics_b['accuracy']:>18.2f}% | {metrics_c['accuracy']:>18.2f}%")
    print(f"{'Macro-F1':<24} | {metrics_a['macro_f1']:>18.2f}% | {metrics_b['macro_f1']:>18.2f}% | {metrics_c['macro_f1']:>18.2f}%")
    print(f"{'Weighted-F1':<24} | {metrics_a['weighted_f1']:>18.2f}% | {metrics_b['weighted_f1']:>18.2f}% | {metrics_c['weighted_f1']:>18.2f}%")
    print("-" * 80)
    print(f"{'TRUE F1-Score':<24} | {metrics_a['per_class']['true']['f1']:>18.2f}% | {metrics_b['per_class']['true']['f1']:>18.2f}% | {metrics_c['per_class']['true']['f1']:>18.2f}%")
    print(f"{'TRUE Recall':<24} | {metrics_a['per_class']['true']['recall']:>18.2f}% | {metrics_b['per_class']['true']['recall']:>18.2f}% | {metrics_c['per_class']['true']['recall']:>18.2f}%")
    print(f"{'TRUE Precision':<24} | {metrics_a['per_class']['true']['precision']:>18.2f}% | {metrics_b['per_class']['true']['precision']:>18.2f}% | {metrics_c['per_class']['true']['precision']:>18.2f}%")
    print("-" * 80)
    print(f"{'FALSE F1-Score':<24} | {metrics_a['per_class']['false']['f1']:>18.2f}% | {metrics_b['per_class']['false']['f1']:>18.2f}% | {metrics_c['per_class']['false']['f1']:>18.2f}%")
    print(f"{'FALSE Recall':<24} | {metrics_a['per_class']['false']['recall']:>18.2f}% | {metrics_b['per_class']['false']['recall']:>18.2f}% | {metrics_c['per_class']['false']['recall']:>18.2f}%")
    print("-" * 80)
    print(f"{'NEI F1-Score':<24} | {metrics_a['per_class']['NEI']['f1']:>18.2f}% | {metrics_b['per_class']['NEI']['f1']:>18.2f}% | {metrics_c['per_class']['NEI']['f1']:>18.2f}%")
    print(f"{'NEI Recall':<24} | {metrics_a['per_class']['NEI']['recall']:>18.2f}% | {metrics_b['per_class']['NEI']['recall']:>18.2f}% | {metrics_c['per_class']['NEI']['recall']:>18.2f}%")
    print("-" * 80)
    print("Predicted Distribution (675 Claims Total):")
    print(f"  System A -> FALSE: {metrics_a['predicted_distribution']['false']}, TRUE: {metrics_a['predicted_distribution']['true']}, NEI: {metrics_a['predicted_distribution']['NEI']}")
    print(f"  System B -> FALSE: {metrics_b['predicted_distribution']['false']}, TRUE: {metrics_b['predicted_distribution']['true']}, NEI: {metrics_b['predicted_distribution']['NEI']}")
    print(f"  System C -> FALSE: {metrics_c['predicted_distribution']['false']}, TRUE: {metrics_c['predicted_distribution']['true']}, NEI: {metrics_c['predicted_distribution']['NEI']}")
    print("=" * 80)

if __name__ == "__main__":
    main()

