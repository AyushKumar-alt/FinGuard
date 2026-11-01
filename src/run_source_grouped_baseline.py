"""
Reproducible, Leakage-Resistant Source-Grouped Logistic Regression Baseline.
Isolates all artifacts under: outputs/experiments/source_grouped_baseline/

Experiments:
- Experiment A: Raw combined text (Claim + Raw Evidence) on source-grouped split.
- Experiment B: Cleaned combined text (Claim + Cleaned Evidence) on source-grouped split.

Protocol:
- Train strictly on grouped_train.csv.
- Tune hyperparameters (C) strictly on grouped_val.csv.
- Evaluate final model strictly on grouped_test.csv.
- No source/URL overlap, no exact duplicate claims across splits.
"""

import os
import sys
import json
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

# Output directory isolation
OUTPUT_DIR = os.path.join("outputs", "experiments", "source_grouped_baseline")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set up logging
log_file = os.path.join(OUTPUT_DIR, "experiment.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("source_grouped_baseline")

TARGET_NAMES = ["false", "true", "NEI"]
LABEL_MAP = {0: "false", 1: "true", 2: "NEI"}

def verify_data_integrity():
    """Verify split paths, row counts, label distribution, and mutual exclusivity."""
    logger.info("=== STEP 1: VERIFYING DATA INTEGRITY & MUTUAL EXCLUSIVITY ===")
    
    train_path = os.path.join("data", "grouped_train.csv")
    val_path = os.path.join("data", "grouped_val.csv")
    test_path = os.path.join("data", "grouped_test.csv")
    
    assert os.path.exists(train_path), f"Missing {train_path}"
    assert os.path.exists(val_path), f"Missing {val_path}"
    assert os.path.exists(test_path), f"Missing {test_path}"
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    logger.info(f"Loaded train: {len(train_df)} rows")
    logger.info(f"Loaded val:   {len(val_df)} rows")
    logger.info(f"Loaded test:  {len(test_df)} rows")
    logger.info(f"Total rows:   {len(train_df) + len(val_df) + len(test_df)} (FinFact total=3369)")
    
    # Check counts
    for split_name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        dist = df["label"].value_counts().to_dict()
        logger.info(f"{split_name} label counts: {dist}")
        
    # Check ID overlaps
    train_ids = set(train_df["id"])
    val_ids = set(val_df["id"])
    test_ids = set(test_df["id"])
    assert len(train_ids & val_ids) == 0, "ID overlap between train and val!"
    assert len(train_ids & test_ids) == 0, "ID overlap between train and test!"
    assert len(val_ids & test_ids) == 0, "ID overlap between val and test!"
    logger.info("ID overlap across all splits: EXACTLY 0 (Mutually exclusive)")
    
    # Check URL overlaps
    train_urls = set(train_df["url"].dropna().unique())
    val_urls = set(val_df["url"].dropna().unique())
    test_urls = set(test_df["url"].dropna().unique())
    assert len(train_urls & val_urls) == 0, "URL overlap between train and val!"
    assert len(train_urls & test_urls) == 0, "URL overlap between train and test!"
    assert len(val_urls & test_urls) == 0, "URL overlap between val and test!"
    logger.info(f"Unique URLs: train={len(train_urls)}, val={len(val_urls)}, test={len(test_urls)}")
    logger.info("URL overlap across all splits: EXACTLY 0 (Mutually exclusive sources)")
    
    # Check claim text overlaps (lowercased)
    train_claims = set(train_df["claim"].dropna().str.strip().str.lower())
    val_claims = set(val_df["claim"].dropna().str.strip().str.lower())
    test_claims = set(test_df["claim"].dropna().str.strip().str.lower())
    assert len(train_claims & val_claims) == 0, "Claim text overlap between train and val!"
    assert len(train_claims & test_claims) == 0, "Claim text overlap between train and test!"
    assert len(val_claims & test_claims) == 0, "Claim text overlap between val and test!"
    logger.info("Claim text overlap across all splits: EXACTLY 0 (No duplicate claims)")
    
    return train_df, val_df, test_df

def tune_and_train_baseline(train_df, val_df, test_df, text_column, experiment_name):
    """
    Tuning protocol:
    1. Fit TF-IDF on train_df ONLY.
    2. Tune regularization C in [0.1, 0.5, 1.0, 2.0, 5.0] using validation Macro-F1.
    3. Select best C strictly on validation set.
    4. Retrain on train_df with best C (or use fitted best model).
    5. Evaluate on test_df.
    """
    logger.info(f"\n========================================================")
    logger.info(f"STARTING {experiment_name}")
    logger.info(f"Text Column: {text_column}")
    logger.info(f"========================================================")
    
    X_train = train_df[text_column].fillna("").astype(str)
    y_train = train_df["label_id"].astype(int)
    
    X_val = val_df[text_column].fillna("").astype(str)
    y_val = val_df["label_id"].astype(int)
    
    X_test = test_df[text_column].fillna("").astype(str)
    y_test = test_df["label_id"].astype(int)
    
    c_candidates = [0.1, 0.5, 1.0, 2.0, 5.0]
    val_results = {}
    best_c = 1.0
    best_val_macro_f1 = -1.0
    best_pipe = None
    
    logger.info("Tuning hyperparameter C on validation set...")
    for c in c_candidates:
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                stop_words="english",
                sublinear_tf=True
            )),
            ("clf", LogisticRegression(
                C=c,
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            ))
        ])
        
        # Train strictly on training data
        pipe.fit(X_train, y_train)
        
        # Evaluate on validation data
        val_preds = pipe.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds)
        _, _, val_macro_f1, _ = precision_recall_fscore_support(
            y_val, val_preds, average="macro", zero_division=0
        )
        
        val_results[str(c)] = {
            "validation_accuracy": round(float(val_acc), 4),
            "validation_macro_f1": round(float(val_macro_f1), 4)
        }
        logger.info(f"  C={c:4.1f} | Val Acc: {val_acc:.4f} | Val Macro-F1: {val_macro_f1:.4f}")
        
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_c = c
            best_pipe = pipe
            
    logger.info(f"Selected Best C={best_c} based strictly on Val Macro-F1={best_val_macro_f1:.4f}")
    
    # Evaluate final model strictly on held-out test split
    logger.info("Evaluating selected model on held-out test set...")
    test_preds = best_pipe.predict(X_test)
    
    acc = accuracy_score(y_test, test_preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, test_preds, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, test_preds, average="weighted", zero_division=0
    )
    per_class_p, per_class_r, per_class_f1, per_class_supp = precision_recall_fscore_support(
        y_test, test_preds, average=None, zero_division=0
    )
    cm = confusion_matrix(y_test, test_preds).tolist()
    
    report_dict = classification_report(
        y_test, test_preds, target_names=TARGET_NAMES, output_dict=True, zero_division=0
    )
    
    logger.info(f"\n--- TEST METRICS ({experiment_name}) ---")
    logger.info(f"Accuracy:         {acc:.4f}")
    logger.info(f"Macro Precision:  {p_macro:.4f}")
    logger.info(f"Macro Recall:     {r_macro:.4f}")
    logger.info(f"Macro F1:         {f1_macro:.4f}")
    logger.info(f"Weighted F1:      {f1_weighted:.4f}")
    logger.info("\nPer-Class Breakdown:")
    for idx, cname in enumerate(TARGET_NAMES):
        logger.info(
            f"  {cname:6s} -> Precision: {per_class_p[idx]:.4f} | "
            f"Recall: {per_class_r[idx]:.4f} | F1: {per_class_f1[idx]:.4f} | "
            f"Support: {per_class_supp[idx]}"
        )
    logger.info(f"\nConfusion Matrix (rows=True, cols=Pred):\n{np.array(cm)}")
    
    metrics = {
        "experiment_name": experiment_name,
        "text_column": text_column,
        "selected_hyperparameters": {
            "C": best_c,
            "max_features": 10000,
            "ngram_range": [1, 2],
            "sublinear_tf": True,
            "stop_words": "english",
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": 42
        },
        "validation_tuning_results": val_results,
        "test_metrics": {
            "accuracy": round(float(acc), 4),
            "macro_precision": round(float(p_macro), 4),
            "macro_recall": round(float(r_macro), 4),
            "macro_f1": round(float(f1_macro), 4),
            "weighted_f1": round(float(f1_weighted), 4),
            "per_class": {
                "false": {
                    "precision": round(float(per_class_p[0]), 4),
                    "recall": round(float(per_class_r[0]), 4),
                    "f1": round(float(per_class_f1[0]), 4),
                    "support": int(per_class_supp[0])
                },
                "true": {
                    "precision": round(float(per_class_p[1]), 4),
                    "recall": round(float(per_class_r[1]), 4),
                    "f1": round(float(per_class_f1[1]), 4),
                    "support": int(per_class_supp[1])
                },
                "NEI": {
                    "precision": round(float(per_class_p[2]), 4),
                    "recall": round(float(per_class_r[2]), 4),
                    "f1": round(float(per_class_f1[2]), 4),
                    "support": int(per_class_supp[2])
                }
            },
            "confusion_matrix": cm,
            "classification_report": report_dict
        }
    }
    
    return best_pipe, metrics, np.array(cm)

def plot_confusion_matrix(cm, title, filename):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES,
        cbar=False, annot_kws={"size": 13, "weight": "bold"}
    )
    plt.title(title, fontsize=11, fontweight="bold", pad=10)
    plt.xlabel("Predicted Label", fontsize=10, fontweight="bold")
    plt.ylabel("Ground Truth Label", fontsize=10, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix figure to {out_path}")

def plot_per_class_comparison(metrics_a, metrics_b):
    classes = TARGET_NAMES
    f1_a = [metrics_a["test_metrics"]["per_class"][c]["f1"] for c in classes]
    f1_b = [metrics_b["test_metrics"]["per_class"][c]["f1"] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.35
    
    plt.figure(figsize=(7, 4.5))
    plt.bar(x - width/2, f1_a, width, label="Exp A: Raw Text", color="#2b5c8f")
    plt.bar(x + width/2, f1_b, width, label="Exp B: Cleaned Text", color="#e27c38")
    
    plt.ylabel("F1 Score", fontsize=10, fontweight="bold")
    plt.title("Per-Class F1 Score Comparison (Source-Grouped Split)", fontsize=11, fontweight="bold")
    plt.xticks(x, [c.upper() for c in classes], fontweight="bold")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    out_path = os.path.join(OUTPUT_DIR, "per_class_f1_comparison.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved per-class comparison figure to {out_path}")

def run_experiment():
    train_df, val_df, test_df = verify_data_integrity()
    
    # Experiment A: Raw combined text
    pipe_a, metrics_a, cm_a = tune_and_train_baseline(
        train_df, val_df, test_df,
        text_column="combined_text",
        experiment_name="Experiment A: Raw Combined Text (Source-Grouped)"
    )
    model_a_path = os.path.join(OUTPUT_DIR, "experiment_a_raw_model.joblib")
    joblib.dump(pipe_a, model_a_path)
    logger.info(f"Saved model A to {model_a_path}")
    plot_confusion_matrix(
        cm_a,
        "Confusion Matrix: Exp A (Raw Text, Grouped Test N=675)",
        "confusion_matrix_exp_a.png"
    )
    
    # Experiment B: Cleaned combined text
    pipe_b, metrics_b, cm_b = tune_and_train_baseline(
        train_df, val_df, test_df,
        text_column="cleaned_combined_text",
        experiment_name="Experiment B: Cleaned Combined Text (Source-Grouped)"
    )
    model_b_path = os.path.join(OUTPUT_DIR, "experiment_b_cleaned_model.joblib")
    joblib.dump(pipe_b, model_b_path)
    logger.info(f"Saved model B to {model_b_path}")
    plot_confusion_matrix(
        cm_b,
        "Confusion Matrix: Exp B (Cleaned Text, Grouped Test N=675)",
        "confusion_matrix_exp_b.png"
    )
    
    # Comparison chart
    plot_per_class_comparison(metrics_a, metrics_b)
    
    # Combined results JSON
    all_results = {
        "dataset_split": "source-grouped (data/grouped_train.csv, grouped_val.csv, grouped_test.csv)",
        "split_counts": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
            "total": len(train_df) + len(val_df) + len(test_df)
        },
        "experiment_a_raw": metrics_a,
        "experiment_b_cleaned": metrics_b
    }
    
    results_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved full results to {results_path}")
    
    print("\n========================================================")
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"Results directory: {OUTPUT_DIR}")
    print("========================================================")

if __name__ == "__main__":
    run_experiment()

