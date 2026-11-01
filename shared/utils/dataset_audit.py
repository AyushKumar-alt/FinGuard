"""
Comprehensive Dataset Audit for FinFact Benchmark
Audits schema, records, label distributions, missing fields, duplicates, evidence,
and split leakage.
"""

import json
import os
from collections import Counter, defaultdict
import urllib.parse
import pandas as pd

def audit_finfact(json_path="finfact.json", output_path="outputs/dataset_audit.json"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    total_records = len(data)
    
    # 1. Field presence and types
    field_counts = Counter()
    missing_fields = defaultdict(int)
    empty_fields = defaultdict(int)
    
    # Label analysis
    raw_labels = Counter()
    
    # Duplicates tracking
    exact_claims = Counter()
    norm_claims = Counter()
    urls = Counter()
    url_domains = Counter()
    
    # Content statistics
    total_evidence_sentences = 0
    total_evidence_hrefs = 0
    records_with_zero_evidence = 0
    records_with_images = 0
    records_with_visualization_bias = 0
    
    # Text leakage cue detection (e.g., explicit verdict words inside evidence sentences)
    leakage_cues = ["false", "falsely", "true", "debunk", "incorrect", "misleading", "untrue", "accurate", "hoax", "fact-check"]
    records_with_leakage_in_evidence = 0
    
    for idx, item in enumerate(data):
        for k in item.keys():
            field_counts[k] += 1
            
        label = item.get("label")
        raw_labels[str(label)] += 1
        
        claim = item.get("claim")
        if not claim or not str(claim).strip():
            empty_fields["claim"] += 1
        else:
            exact_claims[claim] += 1
            norm_claims[claim.strip().lower()] += 1
            
        justification = item.get("justification")
        if not justification or not str(justification).strip():
            empty_fields["justification"] += 1
            
        sci_digest = item.get("sci_digest")
        if not sci_digest or (isinstance(sci_digest, list) and len(sci_digest) == 0):
            empty_fields["sci_digest"] += 1
            
        url = item.get("url")
        if not url:
            empty_fields["url"] += 1
        else:
            urls[url] += 1
            domain = urllib.parse.urlparse(url).netloc
            url_domains[domain] += 1
            
        # Evidence analysis
        evidence = item.get("evidence", [])
        if not evidence or len(evidence) == 0:
            records_with_zero_evidence += 1
        else:
            item_has_leakage = False
            for ev in evidence:
                if isinstance(ev, dict):
                    sent = ev.get("sentence", "")
                    if sent:
                        total_evidence_sentences += 1
                        sent_lower = sent.lower()
                        if any(cue in sent_lower for cue in leakage_cues):
                            item_has_leakage = True
                    hrefs = ev.get("hrefs", [])
                    if isinstance(hrefs, list):
                        total_evidence_hrefs += len(hrefs)
            if item_has_leakage:
                records_with_leakage_in_evidence += 1
                
        if item.get("image_data"):
            records_with_images += 1
            
        if item.get("visualization_bias"):
            records_with_visualization_bias += 1

    duplicate_exact_claims = {k: v for k, v in exact_claims.items() if v > 1}
    duplicate_norm_claims = {k: v for k, v in norm_claims.items() if v > 1}
    duplicate_urls = {k: v for k, v in urls.items() if v > 1}

    # Audit existing splits in data/
    split_audit = {}
    if os.path.exists("data/train.csv") and os.path.exists("data/val.csv") and os.path.exists("data/test.csv"):
        train_df = pd.read_csv("data/train.csv")
        val_df = pd.read_csv("data/val.csv")
        test_df = pd.read_csv("data/test.csv")
        
        train_claims = set(train_df["claim"].dropna().str.strip().str.lower())
        val_claims = set(val_df["claim"].dropna().str.strip().str.lower())
        test_claims = set(test_df["claim"].dropna().str.strip().str.lower())
        
        claim_overlap_train_val = len(train_claims.intersection(val_claims))
        claim_overlap_train_test = len(train_claims.intersection(test_claims))
        claim_overlap_val_test = len(val_claims.intersection(test_claims))
        
        split_audit = {
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "total_split_rows": len(train_df) + len(val_df) + len(test_df),
            "train_label_distribution": train_df["label"].value_counts().to_dict(),
            "val_label_distribution": val_df["label"].value_counts().to_dict(),
            "test_label_distribution": test_df["label"].value_counts().to_dict(),
            "claim_overlap_train_val": claim_overlap_train_val,
            "claim_overlap_train_test": claim_overlap_train_test,
            "claim_overlap_val_test": claim_overlap_val_test,
        }

    audit_results = {
        "dataset_name": "FinFact",
        "total_records": total_records,
        "schema_keys": list(field_counts.keys()),
        "label_distribution": dict(raw_labels),
        "empty_or_missing_fields": dict(empty_fields),
        "duplicate_analysis": {
            "unique_exact_claims": len(exact_claims),
            "duplicate_exact_claims_count": len(duplicate_exact_claims),
            "unique_normalized_claims": len(norm_claims),
            "duplicate_normalized_claims_count": len(duplicate_norm_claims),
            "unique_urls": len(urls),
            "duplicate_urls_count": len(duplicate_urls),
            "url_domains": dict(url_domains.most_common(10))
        },
        "evidence_analysis": {
            "total_evidence_sentences": total_evidence_sentences,
            "total_evidence_hrefs": total_evidence_hrefs,
            "records_with_zero_evidence": records_with_zero_evidence,
            "avg_evidence_sentences_per_record": round(total_evidence_sentences / total_records, 2),
            "records_with_leakage_cues_in_evidence": records_with_leakage_in_evidence,
            "percentage_records_with_evidence_leakage_cues": round(records_with_leakage_in_evidence / total_records * 100, 2)
        },
        "multimodal_analysis": {
            "records_with_images": records_with_images,
            "records_with_visualization_bias": records_with_visualization_bias
        },
        "existing_split_audit": split_audit
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)
        
    print(f"Dataset audit successfully written to {output_path}")
    print("\n--- Key Audit Findings ---")
    print(f"Total Records: {total_records}")
    print(f"Labels: {dict(raw_labels)}")
    print(f"Empty claims: {empty_fields['claim']}, Empty justifications: {empty_fields['justification']}")
    print(f"Duplicate normalized claims: {len(duplicate_norm_claims)}")
    print(f"Unique source URLs: {len(urls)} across {len(url_domains)} domains")
    print(f"Records with zero evidence sentences: {records_with_zero_evidence}")
    print(f"Records with verdict leakage cues in evidence: {records_with_leakage_in_evidence} ({records_with_leakage_in_evidence/total_records*100:.1f}%)")
    if split_audit:
        print(f"Existing Split Claim Overlaps: Train-Test={split_audit['claim_overlap_train_test']}, Train-Val={split_audit['claim_overlap_train_val']}")
    
    return audit_results

if __name__ == "__main__":
    audit_finfact()
