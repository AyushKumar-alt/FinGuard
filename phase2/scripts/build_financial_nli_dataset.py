"""
FinGuard Phase 2: Domain-Specific Financial NLI Dataset Builder.

Constructs rigorously audited, leakage-free premise-hypothesis pairs from
the FinFact dataset using explicit, rule-based provenance.

Premise: Candidate financial evidence passage.
Hypothesis: Financial claim under evaluation.
Target Labels:
  0 = Contradiction
  1 = Entailment
  2 = Neutral

Enforces:
1. Strict split boundaries (train from grouped_train, val from grouped_val, test from grouped_test).
2. Claim-level clumping (all pairs from a claim remain in the same split).
3. Zero test contamination.
4. Transparent rule-based provenance and separation of high-confidence vs weakly-supervised pairs.
5. Extraction of ambiguous cases into human review queues with formal annotation guidelines.
"""

import os
import sys
import re
import json
import logging
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("build_financial_nli")

# Refutation cues (strict regular expression)
REFUTE_PATTERNS = [
    r"\b(false|untrue|incorrect|misleading|fabricated|fake|debunk|myth|hoax|disproved|discredited)\b",
    r"\b(no evidence|not true|never happened|did not|didn't|was not|wasn't|has not|hasn't|had not|hadn't|were not|weren't|does not|doesn't|cannot)\b",
    r"\b(in fact|actually|contrary to|instead|rather than|reality is|disputed|refuted|denied|refused|rejected|contradicts|misrepresented|inaccurate)\b"
]
REFUTE_RE = re.compile("|".join(REFUTE_PATTERNS), re.IGNORECASE)

# Disclaimer and introductory noise patterns to exclude
DISCLAIMER_PATTERNS = [
    r"^(in july and august|facebook users shared|readers asked snopes|contact us|click here|read more|advertisement|sign up|subscribe)",
    r"^(photo credit|image credit|source:|via:|follow us|copyright|all rights reserved)"
]
DISCLAIMER_RE = re.compile("|".join(DISCLAIMER_PATTERNS), re.IGNORECASE)

STOPWORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with", "by", "of",
    "is", "was", "are", "were", "it", "its", "that", "this", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "from", "as", "at", "about"
}

def get_content_words(text: str) -> set:
    """Extracts non-stopword alphabetic tokens (length >= 3) for lexical overlap checks."""
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return set(w for w in words if w not in STOPWORDS)

def build_provenance_registry() -> Dict[str, Any]:
    """Returns metadata descriptions of all labeling rules."""
    return {
        "R_ENTAIL_HIGH_OVERLAP": {
            "target_label": 1,
            "label_name": "entailment",
            "source_claim_gold_label": "true",
            "description": "Evidence sentence from verified TRUE claim with >= 2 matching content tokens and zero refutation cues.",
            "confidence": 0.90,
            "is_weakly_supervised": False
        },
        "R_ENTAIL_MODERATE_OVERLAP": {
            "target_label": 1,
            "label_name": "entailment",
            "source_claim_gold_label": "true",
            "description": "Evidence sentence from verified TRUE claim with exactly 1 matching content token and zero refutation cues.",
            "confidence": 0.75,
            "is_weakly_supervised": True
        },
        "R_CONTRA_EXPLICIT_REFUTATION": {
            "target_label": 0,
            "label_name": "contradiction",
            "source_claim_gold_label": "false",
            "description": "Evidence sentence from verified FALSE claim containing explicit negation/refutation markers and >= 1 matching content token.",
            "confidence": 0.88,
            "is_weakly_supervised": False
        },
        "R_NEUTRAL_ARTICLE_CONTEXT": {
            "target_label": 2,
            "label_name": "neutral",
            "source_claim_gold_label": "true",
            "description": "Contextual sentence from the same article of a TRUE claim with 0 content word overlap, providing setting or background without affirming the assertion.",
            "confidence": 0.85,
            "is_weakly_supervised": False
        },
        "R_NEUTRAL_FALSE_CONTEXT": {
            "target_label": 2,
            "label_name": "neutral",
            "source_claim_gold_label": "false",
            "description": "Background sentence from a FALSE claim article with zero refutation cues and 0 content word overlap.",
            "confidence": 0.85,
            "is_weakly_supervised": False
        },
        "R_NEUTRAL_NEI_CONTEXT": {
            "target_label": 2,
            "label_name": "neutral",
            "source_claim_gold_label": "NEI",
            "description": "Evidence sentence from an inconclusive (NEI) claim that provides non-definitive background and lacks refutation cues.",
            "confidence": 0.88,
            "is_weakly_supervised": False
        }
    }

def extract_pairs(split_df: pd.DataFrame, raw_map: Dict[str, Any], split_name: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extracts premise-hypothesis pairs for a given split according to the provenance rules.
    """
    pairs = []
    ambiguous = []
    
    for _, row in split_df.iterrows():
        claim_id = int(row["id"])
        claim_text = str(row["claim"]).strip()
        claim_clean = claim_text.lower()
        gold_label = str(row["label"]).strip()
        source_url = str(row["url"]).strip() if pd.notnull(row["url"]) else ""
        
        raw_item = raw_map.get(claim_clean, {})
        evidence_list = raw_item.get("evidence", [])
        claim_words = get_content_words(claim_text)
        
        for ev_idx, ev in enumerate(evidence_list):
            sent = ev.get("sentence", "").strip()
            word_count = len(sent.split())
            if word_count < 8 or word_count > 180:
                continue
            if DISCLAIMER_RE.search(sent):
                continue
                
            sent_words = get_content_words(sent)
            overlap = len(claim_words.intersection(sent_words))
            has_refute = bool(REFUTE_RE.search(sent))
            
            # 1. Gold TRUE Claims
            if gold_label == "true":
                if overlap >= 2 and not has_refute:
                    pairs.append({
                        "split": split_name,
                        "claim_id": claim_id,
                        "evidence_idx": ev_idx,
                        "claim": claim_text,
                        "premise": sent,
                        "label": 1,
                        "label_name": "entailment",
                        "provenance_rule": "R_ENTAIL_HIGH_OVERLAP",
                        "confidence": 0.90,
                        "is_weakly_supervised": False,
                        "overlap_token_count": overlap,
                        "source_url": source_url
                    })
                elif overlap >= 1 and not has_refute:
                    pairs.append({
                        "split": split_name,
                        "claim_id": claim_id,
                        "evidence_idx": ev_idx,
                        "claim": claim_text,
                        "premise": sent,
                        "label": 1,
                        "label_name": "entailment",
                        "provenance_rule": "R_ENTAIL_MODERATE_OVERLAP",
                        "confidence": 0.75,
                        "is_weakly_supervised": True,
                        "overlap_token_count": overlap,
                        "source_url": source_url
                    })
                elif overlap == 0 and not has_refute:
                    pairs.append({
                        "split": split_name,
                        "claim_id": claim_id,
                        "evidence_idx": ev_idx,
                        "claim": claim_text,
                        "premise": sent,
                        "label": 2,
                        "label_name": "neutral",
                        "provenance_rule": "R_NEUTRAL_ARTICLE_CONTEXT",
                        "confidence": 0.85,
                        "is_weakly_supervised": False,
                        "overlap_token_count": 0,
                        "source_url": source_url
                    })
                else:
                    # Refutation cues inside a true claim fact check
                    ambiguous.append({
                        "split": split_name,
                        "claim_id": claim_id,
                        "evidence_idx": ev_idx,
                        "claim": claim_text,
                        "premise": sent,
                        "reason": "TRUE claim containing refutation markers in evidence sentence",
                        "source_url": source_url
                    })
                    
            # 2. Gold FALSE Claims
            elif gold_label == "false":
                if has_refute and overlap >= 1:
                    pairs.append({
                        "split": split_name,
                        "claim_id": claim_id,
                        "evidence_idx": ev_idx,
                        "claim": claim_text,
                        "premise": sent,
                        "label": 0,
                        "label_name": "contradiction",
                        "provenance_rule": "R_CONTRA_EXPLICIT_REFUTATION",
                        "confidence": 0.88,
                        "is_weakly_supervised": False,
                        "overlap_token_count": overlap,
                        "source_url": source_url
                    })
                elif has_refute and overlap == 0:
                    ambiguous.append({
                        "split": split_name,
                        "claim_id": claim_id,
                        "evidence_idx": ev_idx,
                        "claim": claim_text,
                        "premise": sent,
                        "reason": "FALSE claim with refutation cues but zero lexical overlap",
                        "source_url": source_url
                    })
                elif not has_refute and overlap == 0:
                    pairs.append({
                        "split": split_name,
                        "claim_id": claim_id,
                        "evidence_idx": ev_idx,
                        "claim": claim_text,
                        "premise": sent,
                        "label": 2,
                        "label_name": "neutral",
                        "provenance_rule": "R_NEUTRAL_FALSE_CONTEXT",
                        "confidence": 0.85,
                        "is_weakly_supervised": False,
                        "overlap_token_count": 0,
                        "source_url": source_url
                    })
                else:
                    # Overlap > 0 but no refutation cues: high risk of quoting the false rumor!
                    ambiguous.append({
                        "split": split_name,
                        "claim_id": claim_id,
                        "evidence_idx": ev_idx,
                        "claim": claim_text,
                        "premise": sent,
                        "reason": "FALSE claim article sentence without refutation cues; potential rumor quote",
                        "source_url": source_url
                    })
                    
            # 3. Gold NEI Claims
            elif gold_label == "NEI":
                if not has_refute:
                    pairs.append({
                        "split": split_name,
                        "claim_id": claim_id,
                        "evidence_idx": ev_idx,
                        "claim": claim_text,
                        "premise": sent,
                        "label": 2,
                        "label_name": "neutral",
                        "provenance_rule": "R_NEUTRAL_NEI_CONTEXT",
                        "confidence": 0.88,
                        "is_weakly_supervised": False,
                        "overlap_token_count": overlap,
                        "source_url": source_url
                    })
                else:
                    ambiguous.append({
                        "split": split_name,
                        "claim_id": claim_id,
                        "evidence_idx": ev_idx,
                        "claim": claim_text,
                        "premise": sent,
                        "reason": "NEI claim containing partial refutation cue",
                        "source_url": source_url
                    })
                    
    return pairs, ambiguous

def write_jsonl(records: List[Dict[str, Any]], filepath: Path):
    with open(filepath, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records to %s", len(records), filepath)

def main():
    parser = argparse.ArgumentParser(description="Build Financial NLI Dataset")
    parser.add_argument("--output_dir", default="phase2/data/financial_nli", help="Output directory")
    args = parser.parse_args()
    
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading master FinFact raw data...")
    with open(ROOT / "data" / "raw" / "finfact.json", "r", encoding="utf-8") as f:
        raw_finfact = json.load(f)
    raw_map = {item.get("claim", "").strip().lower(): item for item in raw_finfact}
    
    logger.info("Loading split definitions...")
    train_df = pd.read_csv(ROOT / "data" / "grouped_train.csv")
    val_df = pd.read_csv(ROOT / "data" / "grouped_val.csv")
    test_df = pd.read_csv(ROOT / "data" / "grouped_test.csv")
    
    # Split integrity assertions
    train_ids = set(train_df["id"].astype(int))
    val_ids = set(val_df["id"].astype(int))
    test_ids = set(test_df["id"].astype(int))
    
    assert len(train_ids.intersection(val_ids)) == 0, "Train and Val IDs overlap!"
    assert len(train_ids.intersection(test_ids)) == 0, "Train and Test IDs overlap!"
    assert len(val_ids.intersection(test_ids)) == 0, "Val and Test IDs overlap!"
    logger.info("Split integrity verified: 0 ID overlap across Train, Val, and Test.")
    
    # Extract pairs
    train_pairs, train_ambiguous = extract_pairs(train_df, raw_map, "train")
    val_pairs, val_ambiguous = extract_pairs(val_df, raw_map, "validation")
    test_pairs, test_ambiguous = extract_pairs(test_df, raw_map, "test")
    
    # Save files
    write_jsonl(train_pairs, out_dir / "train.jsonl")
    write_jsonl(val_pairs, out_dir / "validation.jsonl")
    write_jsonl(test_pairs, out_dir / "test.jsonl")
    write_jsonl(train_ambiguous + val_ambiguous, out_dir / "human_review_queue.jsonl")
    
    # Statistics
    stats = {
        "dataset_name": "FinGuard Financial NLI Benchmark",
        "created_at": "2026-11-01",
        "label_mapping": {
            0: "contradiction",
            1: "entailment",
            2: "neutral"
        },
        "splits": {
            "train": {
                "total_pairs": len(train_pairs),
                "class_distribution": dict(Counter(p["label_name"] for p in train_pairs)),
                "high_confidence_pairs": sum(1 for p in train_pairs if not p["is_weakly_supervised"]),
                "weakly_supervised_pairs": sum(1 for p in train_pairs if p["is_weakly_supervised"]),
                "ambiguous_quarantined_pairs": len(train_ambiguous)
            },
            "validation": {
                "total_pairs": len(val_pairs),
                "class_distribution": dict(Counter(p["label_name"] for p in val_pairs)),
                "high_confidence_pairs": sum(1 for p in val_pairs if not p["is_weakly_supervised"]),
                "weakly_supervised_pairs": sum(1 for p in val_pairs if p["is_weakly_supervised"]),
                "ambiguous_quarantined_pairs": len(val_ambiguous)
            },
            "test": {
                "total_pairs": len(test_pairs),
                "class_distribution": dict(Counter(p["label_name"] for p in test_pairs)),
                "high_confidence_pairs": sum(1 for p in test_pairs if not p["is_weakly_supervised"]),
                "weakly_supervised_pairs": sum(1 for p in test_pairs if p["is_weakly_supervised"]),
                "ambiguous_quarantined_pairs": len(test_ambiguous)
            }
        },
        "integrity_checks": {
            "train_val_claim_overlap": len(train_ids.intersection(val_ids)),
            "train_test_claim_overlap": len(train_ids.intersection(test_ids)),
            "test_claims_used_in_training": False
        }
    }
    
    with open(out_dir / "dataset_statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info("Saved dataset statistics to %s", out_dir / "dataset_statistics.json")
    
    provenance = build_provenance_registry()
    with open(out_dir / "labeling_provenance.json", "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    logger.info("Saved labeling provenance to %s", out_dir / "labeling_provenance.json")

    # Write annotation guidelines
    guidelines_content = """# Financial NLI Human Annotation & Review Guidelines

## Overview
This document specifies the scientific protocol for adjudicating ambiguous claim-evidence pairs in FinGuard Phase 2.

## Label Taxonomy
1. **Contradiction (Label 0)**
   - The premise evidence directly conflicts with, refutes, or negates the key factual assertions of the financial claim.
   - Examples: Direct negation ("did not happen"), conflicting monetary amount ("$10B instead of $20B"), incorrect fiscal year, or debunking of a fabricated rumor.

2. **Entailment (Label 1)**
   - The premise evidence directly substantiates, corroborates, or proves the factual premise of the financial claim.
   - Both the entity, financial metric, direction, and magnitude must align.

3. **Neutral (Label 2)**
   - The premise evidence discusses relevant context, entity history, regulatory background, or general definitions, but neither establishes truth nor refutes the claim.
   - Also assigned when evidence is tangential, incomplete, or refers to a different fiscal period without explicit conflict.

## Ambiguity Adjudication Protocol
- When an evidence passage quotes a false claim without debunking language, annotate as **Neutral** unless accompanied by explicit refutation.
- When an evidence passage mentions the same company but an unrelated topic, annotate as **Neutral**.
"""
    with open(out_dir / "annotation_guidelines.md", "w", encoding="utf-8") as f:
        f.write(guidelines_content)
    logger.info("Saved annotation guidelines to %s", out_dir / "annotation_guidelines.md")

if __name__ == "__main__":
    main()

