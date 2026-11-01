"""
Baseline Claim Verification Classifiers and Reproducibility Suite for FinFact.
Evaluates four classical model architectures:
1. TF-IDF + Logistic Regression (Balanced)
2. TF-IDF + Linear SVM (Balanced)
3. TF-IDF + Multinomial Naive Bayes
4. TF-IDF + Random Forest Classifier (Balanced)

Conducts controlled experimental comparisons:
- Experiment A: Standard Split (Claim + Raw Evidence) - Reproduction Test
- Experiment B: Standard Split (Claim-Only) - Prior/Plausibility Test
- Experiment C: Standard Split (Claim + Cleaned Evidence) - Meta-language Leakage Ablation
- Experiment D: Source-Grouped Split (Zero Article URL Overlap) - Generalization Test
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

TARGET_NAMES = ["false", "true", "NEI"]

def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=42))
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
            ("clf", LinearSVC(class_weight="balanced", C=1.0, random_state=42, max_iter=2000))
        ]),
        "Multinomial Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")),
            ("clf", MultinomialNB(alpha=0.5))
        ]),
        "Random Forest": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 1), stop_words="english")),
            ("clf", RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42, n_jobs=-1))
        ])
    }

def evaluate_models_on_split(train_df, test_df, text_column, experiment_name):
    print(f"\n==========================================")
    print(f"Running {experiment_name}")
    print(f"Train samples: {len(train_df)} | Test samples: {len(test_df)}")
    print(f"Test Class Distribution: {test_df['label'].value_counts().to_dict()}")
    print(f"==========================================")
    
    X_train = train_df[text_column].fillna("").astype(str)
    y_train = train_df["label_id"].astype(int)
    X_test = test_df[text_column].fillna("").astype(str)
    y_test = test_df["label_id"].astype(int)
    
    models = build_models()
    exp_results = {}
    
    best_f1 = -1.0
    best_model_name = None
    best_pipeline = None
    best_preds = None
    
    for name, pipeline in models.items():
        # Fit on training data ONLY
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        
        acc = accuracy_score(y_test, preds)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, preds, average="macro", zero_division=0)
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, preds, average="weighted", zero_division=0)
        per_class_p, per_class_r, per_class_f1, per_class_supp = precision_recall_fscore_support(y_test, preds, average=None, zero_division=0)
        
        cm = confusion_matrix(y_test, preds).tolist()
        
        exp_results[name] = {
            "Accuracy": round(float(acc), 4),
            "Macro Precision": round(float(p_macro), 4),
            "Macro Recall": round(float(r_macro), 4),
            "Macro F1": round(float(f1_macro), 4),
            "Weighted F1": round(float(f1_weighted), 4),
            "Per_Class": {
                "false": {"Precision": round(float(per_class_p[0]), 4), "Recall": round(float(per_class_r[0]), 4), "F1": round(float(per_class_f1[0]), 4), "Support": int(per_class_supp[0])},
                "true":  {"Precision": round(float(per_class_p[1]), 4), "Recall": round(float(per_class_r[1]), 4), "F1": round(float(per_class_f1[1]), 4), "Support": int(per_class_supp[1])},
                "NEI":   {"Precision": round(float(per_class_p[2]), 4), "Recall": round(float(per_class_r[2]), 4), "F1": round(float(per_class_f1[2]), 4), "Support": int(per_class_supp[2])}
            },
            "Confusion_Matrix": cm
        }
        
        print(f"[{name}] Acc: {acc:.4f} | Macro F1: {f1_macro:.4f} | Weighted F1: {f1_weighted:.4f}")
        
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_model_name = name
            best_pipeline = pipeline
            best_preds = preds
            
    return exp_results, best_model_name, best_pipeline, best_preds, y_test

def run_all_baseline_experiments(data_dir="data", output_dir="outputs", figures_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # Load standard splits
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    
    # Load grouped splits
    grouped_train_df = pd.read_csv(os.path.join(data_dir, "grouped_train.csv"))
    grouped_test_df = pd.read_csv(os.path.join(data_dir, "grouped_test.csv"))
    
    all_experiments = {}
    
    # Exp A: Standard Split (Claim + Raw Evidence) - Reproduction
    res_a, best_name_a, best_pipe_a, best_preds_a, y_test_a = evaluate_models_on_split(
        train_df, test_df, "combined_text", "Experiment A: Standard Random Split (Claim + Raw Evidence)"
    )
    all_experiments["Experiment_A_Standard_RawEvidence"] = res_a
    
    # Save best model pipeline for deployment
    joblib.dump(best_pipe_a, os.path.join(output_dir, "best_baseline_model.pkl"))
    
    # Exp B: Standard Split (Claim-Only)
    res_b, _, _, _, _ = evaluate_models_on_split(
        train_df, test_df, "claim", "Experiment B: Standard Random Split (Claim Only, No Evidence)"
    )
    all_experiments["Experiment_B_Claim_Only"] = res_b
    
    # Exp C: Standard Split (Claim + Cleaned Evidence - Stripped Verdict Cues)
    res_c, _, _, _, _ = evaluate_models_on_split(
        train_df, test_df, "cleaned_combined_text", "Experiment C: Standard Random Split (Claim + Cleaned Evidence)"
    )
    all_experiments["Experiment_C_Cleaned_Evidence"] = res_c
    
    # Exp D: Source-Grouped Split (Zero URL Overlap)
    res_d, best_name_d, best_pipe_d, best_preds_d, y_test_d = evaluate_models_on_split(
        grouped_train_df, grouped_test_df, "combined_text", "Experiment D: Source-Grouped Split (Zero Article URL Overlap)"
    )
    all_experiments["Experiment_D_Source_Grouped"] = res_d
    
    # Save comprehensive metrics JSON
    metrics_path = os.path.join(output_dir, "baseline_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_experiments, f, indent=2)
    print(f"\nAll baseline metrics successfully saved to {metrics_path}")
    
    # Plot 1: Confusion Matrix for Best Model (Exp A)
    cm_a = np.array(res_a[best_name_a]["Confusion_Matrix"])
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_a, annot=True, fmt="d", cmap="Blues",
                xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES,
                cbar=False, annot_kws={"size": 14, "weight": "bold"})
    plt.title(f"Confusion Matrix: {best_name_a}\nStandard Test Split (N={len(y_test_a)})", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
    plt.ylabel("Ground Truth Label", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "best_baseline_confusion_matrix.png"), dpi=300)
    plt.close()
    
    # Plot 2: Confusion Matrix for Source-Grouped Split (Exp D)
    cm_d = np.array(res_d[best_name_d]["Confusion_Matrix"])
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_d, annot=True, fmt="d", cmap="Greens",
                xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES,
                cbar=False, annot_kws={"size": 14, "weight": "bold"})
    plt.title(f"Confusion Matrix: {best_name_d}\nSource-Grouped Test Split (N={len(y_test_d)}, Zero URL Overlap)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
    plt.ylabel("Ground Truth Label", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "grouped_split_confusion_matrix.png"), dpi=300)
    plt.close()
    
    # Plot 3: Comparison of Model Accuracies across Experiments
    comp_data = []
    for exp_key, exp_label in [
        ("Experiment_A_Standard_RawEvidence", "Standard (Raw Evi)"),
        ("Experiment_B_Claim_Only", "Claim Only"),
        ("Experiment_C_Cleaned_Evidence", "Cleaned Evi"),
        ("Experiment_D_Source_Grouped", "Source Grouped")
    ]:
        for model_name, m in all_experiments[exp_key].items():
            comp_data.append({
                "Setting": exp_label,
                "Model": model_name,
                "Accuracy": m["Accuracy"],
                "Macro_F1": m["Macro F1"]
            })
    comp_df = pd.DataFrame(comp_data)
    
    plt.figure(figsize=(11, 5))
    ax = sns.barplot(data=comp_df, x="Setting", y="Macro_F1", hue="Model", palette="tab10")
    plt.title("Macro F1-Score Across Experimental Settings (Leakage & Split Ablation)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Experimental Setting", fontsize=11, fontweight="bold")
    plt.ylabel("Macro F1", fontsize=11, fontweight="bold")
    plt.ylim(0, 0.9)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "baseline_comparison.png"), dpi=300)
    plt.close()
    
    print(f"Figures generated: best_baseline_confusion_matrix.png, grouped_split_confusion_matrix.png, baseline_comparison.png")
    return all_experiments

if __name__ == "__main__":
    run_all_baseline_experiments()
