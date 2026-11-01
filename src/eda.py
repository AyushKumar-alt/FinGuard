"""
Exploratory Data Analysis (EDA) for FinFact Dataset
Analyzes claim veracity, issues, text length distributions, and evidence structures.
Generates publication-quality charts saved to figures/.
"""

import json
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def run_eda(json_path="finfact.json", output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print(f"Loaded {len(data)} total records from {json_path}")
    
    # Flatten records for tabular analysis
    rows = []
    all_issues = []
    
    for idx, item in enumerate(data):
        label = item.get("label", "unknown")
        claim = item.get("claim", "") or ""
        justification = item.get("justification", "") or ""
        sci_digest = " ".join(item.get("sci_digest", [])) if isinstance(item.get("sci_digest"), list) else (item.get("sci_digest") or "")
        evidence_list = item.get("evidence", []) or []
        issues = item.get("issues", []) or []
        images = item.get("image_data", []) or []
        
        # Combine evidence sentences
        evidence_texts = [e.get("sentence", "") for e in evidence_list if isinstance(e, dict) and e.get("sentence")]
        combined_evidence = " ".join(evidence_texts)
        
        # Issue tracking
        for iss in issues:
            all_issues.append(iss.strip())
            
        rows.append({
            "id": idx,
            "url": item.get("url", ""),
            "author": item.get("author", "Unknown"),
            "posted": item.get("posted", ""),
            "claim": claim,
            "claim_words": len(claim.split()),
            "justification": justification,
            "justification_words": len(justification.split()),
            "sci_digest": sci_digest,
            "sci_digest_words": len(sci_digest.split()),
            "evidence_count": len(evidence_list),
            "evidence_words": len(combined_evidence.split()),
            "combined_evidence": combined_evidence,
            "has_images": len(images) > 0,
            "image_count": len(images),
            "primary_issue": issues[0] if len(issues) > 0 else "None",
            "label": label
        })
        
    df = pd.DataFrame(rows)
    print("\nDataset Summary DataFrame:")
    print(df[["claim_words", "justification_words", "evidence_count", "label"]].describe())
    
    # 1. Label Distribution Plot
    plt.figure(figsize=(8, 5))
    label_counts = df["label"].value_counts()
    colors = ["#e74c3c" if l == "false" else "#2ecc71" if l == "true" else "#f39c12" for l in label_counts.index]
    ax = sns.barplot(x=label_counts.index, y=label_counts.values, palette=colors)
    plt.title("FinFact Label Veracity Distribution", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Veracity Label", fontsize=12)
    plt.ylabel("Number of Claims", fontsize=12)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}\n({p.get_height()/len(df)*100:.1f}%)",
                    (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                    ha="center", va="center", color="white", fontweight="bold", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "label_distribution.png"), dpi=300)
    plt.close()
    
    # 2. Top Financial Issues
    issue_counts = Counter(all_issues)
    top_issues = pd.DataFrame(issue_counts.most_common(12), columns=["Issue", "Count"])
    # Clean case sensitivity duplicates
    top_issues["Issue"] = top_issues["Issue"].str.title()
    top_issues = top_issues.groupby("Issue", as_index=False)["Count"].sum().sort_values(by="Count", ascending=False)
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=top_issues, y="Issue", x="Count", palette="Blues_r")
    plt.title("Top 10 Financial & Economic Issues in FinFact", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Number of Occurrences", fontsize=12)
    plt.ylabel("Topic / Issue", fontsize=12)
    for p in ax.patches:
        ax.annotate(f" {int(p.get_width()):,}",
                    (p.get_width(), p.get_y() + p.get_height() / 2.),
                    va="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_financial_issues.png"), dpi=300)
    plt.close()
    
    # 3. Text Length Distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Claim length by label
    sns.boxplot(data=df, x="label", y="claim_words", palette=["#e74c3c", "#2ecc71", "#f39c12"], ax=axes[0])
    axes[0].set_title("Claim Word Length by Veracity Label", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Label", fontsize=11)
    axes[0].set_ylabel("Word Count", fontsize=11)
    axes[0].set_ylim(0, 45)
    
    # Evidence sentence count distribution
    sns.histplot(df["evidence_count"], bins=20, kde=True, color="#3498db", ax=axes[1])
    axes[1].set_title("Distribution of Supporting Evidence Sentences per Claim", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Evidence Count", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "length_and_evidence_distribution.png"), dpi=300)
    plt.close()
    
    # Summary JSON output
    summary = {
        "total_records": len(df),
        "labels": label_counts.to_dict(),
        "avg_claim_words": round(float(df["claim_words"].mean()), 2),
        "avg_justification_words": round(float(df["justification_words"].mean()), 2),
        "avg_evidence_count": round(float(df["evidence_count"].mean()), 2),
        "items_with_images": int((df["image_count"] > 0).sum()),
        "top_issues": top_issues.head(10).to_dict(orient="records")
    }
    
    summary_path = os.path.join(output_dir, "eda_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\nEDA completed successfully! Figures & summary saved to {output_dir}/")
    return df, summary

if __name__ == "__main__":
    run_eda()
