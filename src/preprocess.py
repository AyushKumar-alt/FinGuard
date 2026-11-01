"""
Data Preprocessing & Dataset Splitting for FinFact.
Handles text cleaning, evidence extraction, provenance tracking, and generates:
1. Stratified random splits (train/val/test)
2. Source-grouped splits (grouped by article URL using StratifiedGroupKFold to prevent source leakage)
"""

import json
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedGroupKFold

LABEL2ID = {"false": 0, "true": 1, "NEI": 2}
ID2LABEL = {0: "false", 1: "true", 2: "NEI"}

def clean_text(text: str) -> str:
    """Normalize whitespace and strip spurious characters."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\u2014', ' - ').replace('\u201c', '"').replace('\u201d', '"')
    return text.strip()

def strip_verdict_cues(text: str) -> str:
    """
    Strips explicit fact-checker verdict cues to test model resilience against lexical leakage.
    Removes phrases like 'falsely claimed', 'false claim', 'debunked', etc.
    """
    patterns = [
        r'\bfalsely\s+claimed\b',
        r'\bfalse\s+claims?\b',
        r'\bdebunked\b',
        r'\bhoax\b',
        r'\bmisleading\s+claims?\b',
        r'\bfact-check\b',
        r'\bincorrectly\s+stated\b',
        r'\buntrue\b'
    ]
    cleaned = text
    for pat in patterns:
        cleaned = re.sub(pat, 'stated', cleaned, flags=re.IGNORECASE)
    return clean_text(cleaned)

def preprocess_finfact(json_path="finfact.json", data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    records = []
    for idx, item in enumerate(raw_data):
        label = item.get("label", "").strip()
        if label not in LABEL2ID:
            continue
            
        claim = clean_text(item.get("claim", ""))
        justification = clean_text(item.get("justification", ""))
        url = clean_text(item.get("url", ""))
        author = clean_text(item.get("author", "Unknown"))
        posted = clean_text(item.get("posted", ""))
        
        # Extract evidence sentences and their source URLs
        evidence_list = item.get("evidence", []) or []
        evidence_sents = []
        evidence_urls = []
        for e in evidence_list:
            if isinstance(e, dict):
                s = clean_text(e.get("sentence", ""))
                if s:
                    evidence_sents.append(s)
                hrefs = e.get("hrefs", [])
                if isinstance(hrefs, list):
                    evidence_urls.extend(hrefs)
                    
        evidence_text = " ".join(evidence_sents)
        cleaned_evidence = strip_verdict_cues(evidence_text)
        
        issues = item.get("issues", []) or []
        image_data = item.get("image_data", []) or []
        
        # Formulate joint inputs
        combined_text = f"Claim: {claim} Evidence: {evidence_text if evidence_text else 'No direct evidence provided.'}"
        cleaned_combined_text = f"Claim: {claim} Evidence: {cleaned_evidence if cleaned_evidence else 'No direct evidence provided.'}"
        
        records.append({
            "id": idx,
            "url": url,
            "author": author,
            "posted": posted,
            "claim": claim,
            "evidence_text": evidence_text,
            "cleaned_evidence_text": cleaned_evidence,
            "combined_text": combined_text,
            "cleaned_combined_text": cleaned_combined_text,
            "justification": justification,
            "evidence_count": len(evidence_sents),
            "evidence_urls_count": len(evidence_urls),
            "issues": ", ".join(issues),
            "image_count": len(image_data),
            "label": label,
            "label_id": LABEL2ID[label]
        })
        
    df = pd.DataFrame(records)
    print(f"Total valid samples: {len(df)}")
    print("Label distribution:\n", df["label"].value_counts())
    
    # 1. Stratified Random Split: 70% Train, 15% Validation, 15% Test
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["label_id"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["label_id"]
    )
    
    # 2. Source-Grouped Split (Grouped by URL to prevent article-level leakage)
    # 5-fold StratifiedGroupKFold gives an 80/20 train/test split; we split the 80 into 70/10 train/val
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    grouped_train_idx, grouped_test_idx = next(sgkf.split(df, df["label_id"], groups=df["url"]))
    
    grouped_train_val = df.iloc[grouped_train_idx].copy().reset_index(drop=True)
    grouped_test = df.iloc[grouped_test_idx].copy().reset_index(drop=True)
    
    # Split train_val into train (approx 70%) and val (approx 10%)
    sgkf_inner = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=42)
    inner_train_idx, inner_val_idx = next(sgkf_inner.split(grouped_train_val, grouped_train_val["label_id"], groups=grouped_train_val["url"]))
    grouped_train = grouped_train_val.iloc[inner_train_idx].copy().reset_index(drop=True)
    grouped_val = grouped_train_val.iloc[inner_val_idx].copy().reset_index(drop=True)
    
    # Save standard splits
    df.to_csv(os.path.join(data_dir, "processed_finfact.csv"), index=False, encoding="utf-8")
    train_df.to_csv(os.path.join(data_dir, "train.csv"), index=False, encoding="utf-8")
    val_df.to_csv(os.path.join(data_dir, "val.csv"), index=False, encoding="utf-8")
    test_df.to_csv(os.path.join(data_dir, "test.csv"), index=False, encoding="utf-8")
    
    # Save grouped splits
    grouped_train.to_csv(os.path.join(data_dir, "grouped_train.csv"), index=False, encoding="utf-8")
    grouped_val.to_csv(os.path.join(data_dir, "grouped_val.csv"), index=False, encoding="utf-8")
    grouped_test.to_csv(os.path.join(data_dir, "grouped_test.csv"), index=False, encoding="utf-8")
    
    print("\n[Standard Random Split] Saved to data/:")
    print(f" - Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f" - Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f" - Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    print("\n[Source-Grouped Split (Zero URL Overlap)] Saved to data/:")
    print(f" - Grouped Train: {len(grouped_train)} ({len(grouped_train)/len(df)*100:.1f}%)")
    print(f" - Grouped Val:   {len(grouped_val)} ({len(grouped_val)/len(df)*100:.1f}%)")
    print(f" - Grouped Test:  {len(grouped_test)} ({len(grouped_test)/len(df)*100:.1f}%)")
    
    # Verify zero URL overlap in grouped split
    gt_urls = set(grouped_train["url"])
    gtest_urls = set(grouped_test["url"])
    overlap = len(gt_urls.intersection(gtest_urls))
    print(f"Grouped Train/Test URL Overlap: {overlap} (Must be 0)")
    assert overlap == 0, "Error: Grouped split leaked URLs across train and test!"
    
    return train_df, val_df, test_df, grouped_train, grouped_val, grouped_test

if __name__ == "__main__":
    preprocess_finfact()
