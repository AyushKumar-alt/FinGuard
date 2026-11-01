"""
FinGuard Benchmark Adjudication & Annotation Workflow Module.

Provides:
1. Review and adjudication routines for pooled candidate records.
2. Transparent, rule-assisted relevance suggestions (strictly labeled as 'model_assisted', pending human adjudication).
3. Frozen judgment store exporting to data/benchmark/v1/relevance_judgments.json and .csv.
4. Clear separation between silver-reference dataset evidence, model-assisted suggestions, and adjudicated judgments.
"""

import os
import sys
import json
import re
import pandas as pd
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gold_benchmark_schema import (
    EvidenceAnnotationRecord,
    normalize_text_for_matching,
    VALID_RELEVANCE_GRADES,
    VALID_SUPPORT_RELATIONS,
    BENCHMARK_VERSION
)
from src.claim_understanding import extract_financial_entities

class BenchmarkAdjudicator:
    def __init__(self,
                 pooled_jsonl_path: str = "data/benchmark/v1/pooled_candidates_annotation.jsonl",
                 judgments_output_dir: str = "data/benchmark/v1"):
        self.pooled_jsonl_path = pooled_jsonl_path
        self.output_dir = judgments_output_dir
        self.records: Dict[str, EvidenceAnnotationRecord] = {}
        self._load_pooled_records()

    def _load_pooled_records(self):
        if not os.path.exists(self.pooled_jsonl_path):
            raise FileNotFoundError(f"Pooled annotation file missing: {self.pooled_jsonl_path}")

        with open(self.pooled_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line.strip())
                    rec = EvidenceAnnotationRecord.from_dict(d)
                    key = f"{rec.claim_id}::{rec.evidence_id}"
                    self.records[key] = rec

        print(f"Loaded {len(self.records):,} pooled candidate records for adjudication.", flush=True)

    def apply_rule_assisted_suggestions(self, overwrite_existing_judgments: bool = False):
        """
        Applies transparent, deterministic financial relevance heuristics to suggest grades
        for UNJUDGED candidates.
        
        Strict Scientific Controls:
        - NEVER labels an automated suggestion as 'human-verified gold'.
        - Sets annotation_source='model_assisted' and adjudication_status='pending'.
        - Preserves unreviewed records as unjudged unless explicit evidence indicators are matched.
        """
        print("Applying transparent financial-entity heuristic suggestions...", flush=True)
        suggested_count = 0

        for key, rec in self.records.items():
            # Never overwrite silver reference evidence or already approved judgments
            if rec.judgment_status == "silver_reference" or rec.adjudication_status == "approved":
                continue
            if not overwrite_existing_judgments and rec.relevance_grade is not None:
                continue

            claim = rec.claim_text or ""
            evidence = rec.evidence_text or ""
            if not claim or not evidence:
                continue

            # Extract entities and metrics from claim
            parsed = extract_financial_entities(claim)
            entities = parsed.get("entities", [])
            metrics = parsed.get("financial_metrics", [])
            monetary = parsed.get("monetary_amounts", [])
            percentages = parsed.get("percentages", [])
            periods = parsed.get("fiscal_periods", [])

            norm_ev = normalize_text_for_matching(evidence)

            # Heuristic 1: Check entity and monetary/metric match
            matched_entities = [e for e in entities if normalize_text_for_matching(e) in norm_ev]
            matched_monetary = [m for m in monetary if normalize_text_for_matching(m) in norm_ev]
            matched_pct = [p for p in percentages if normalize_text_for_matching(p) in norm_ev]
            matched_periods = [t for t in periods if normalize_text_for_matching(t) in norm_ev]
            matched_metrics = [m for m in metrics if normalize_text_for_matching(m) in norm_ev]

            # Assess relevance level
            strong_signals = len(matched_entities) + len(matched_monetary) + len(matched_pct) + len(matched_periods)
            has_metric_overlap = len(matched_metrics) > 0

            # Direct decisive indicator (Grade 3 suggestion)
            if (matched_entities and (matched_monetary or matched_pct) and (matched_periods or has_metric_overlap)):
                rec.relevance_grade = 3
                rec.judgment_status = "reviewed"
                rec.annotation_source = "model_assisted"
                rec.annotation_method = "financial_rule_heuristic_v1"
                rec.annotator_or_reviewer = "rule_assisted_relevance_engine"
                rec.adjudication_status = "pending"
                rec.support_relation = self._infer_relation_heuristic(norm_ev, parsed)
                rec.rationale = f"Heuristic match: Contains entity {matched_entities}, numerical values {matched_monetary + matched_pct}, and period/metric {matched_periods + matched_metrics}."
                suggested_count += 1
            # Substantial relevant indicator (Grade 2 suggestion)
            elif (matched_entities and (matched_monetary or matched_pct or has_metric_overlap)):
                rec.relevance_grade = 2
                rec.judgment_status = "reviewed"
                rec.annotation_source = "model_assisted"
                rec.annotation_method = "financial_rule_heuristic_v1"
                rec.annotator_or_reviewer = "rule_assisted_relevance_engine"
                rec.adjudication_status = "pending"
                rec.support_relation = "context_only"
                rec.rationale = f"Partial entity ({matched_entities}) and financial metric overlap without full temporal/numerical resolution."
                suggested_count += 1
            # Contextual relevant indicator (Grade 1 suggestion)
            elif matched_entities or (has_metric_overlap and (matched_monetary or matched_pct)):
                rec.relevance_grade = 1
                rec.judgment_status = "reviewed"
                rec.annotation_source = "model_assisted"
                rec.annotation_method = "financial_rule_heuristic_v1"
                rec.annotator_or_reviewer = "rule_assisted_relevance_engine"
                rec.adjudication_status = "pending"
                rec.support_relation = "context_only"
                rec.rationale = "Mentions target organization or financial metric in broader economic context."
                suggested_count += 1
            # If no entity/metric overlap, keep as unjudged (NOT automatically grade 0)
            else:
                # Keep explicitly unjudged to avoid false negative assumptions
                rec.relevance_grade = None
                rec.judgment_status = "unjudged"
                rec.support_relation = "unknown"
                rec.annotation_source = "unjudged_pool"

        print(f"Applied heuristic suggestions to {suggested_count:,} candidates (relevance grades 1-3). Remaining {len(self.records) - suggested_count:,} candidates preserved explicitly as unjudged/reference.", flush=True)

    def _infer_relation_heuristic(self, norm_ev: str, parsed_claim: Dict[str, Any]) -> str:
        """Determines preliminary support direction based on negation and directional polarity."""
        directions = parsed_claim.get("directional_language", [])
        negations = parsed_claim.get("negations", [])

        has_negation_in_ev = any(w in norm_ev for w in ["not", "never", "false", "debunked", "untrue", "denied", "no evidence"])
        has_affirmation_in_ev = any(w in norm_ev for w in ["confirmed", "announced", "reported", "shows that", "official records", "increased by"])

        if has_negation_in_ev and not has_affirmation_in_ev:
            return "refutes"
        elif has_affirmation_in_ev and not has_negation_in_ev:
            return "supports"
        elif has_negation_in_ev and has_affirmation_in_ev:
            return "mixed"
        return "context_only"

    def adjudicate_record(self,
                           claim_id: str,
                           evidence_id: str,
                           relevance_grade: Optional[int],
                           support_relation: str,
                           reviewer: str,
                           rationale: str,
                           status: str = "approved") -> EvidenceAnnotationRecord:
        """
        Manually adjudicates a specific candidate pair.
        """
        key = f"{claim_id}::{evidence_id}"
        if key not in self.records:
            raise KeyError(f"Record {key} not found in pooled candidates.")

        rec = self.records[key]
        rec.relevance_grade = relevance_grade
        rec.support_relation = support_relation
        rec.judgment_status = "adjudicated" if relevance_grade is not None else "unjudged"
        rec.annotation_source = "manual_annotation"
        rec.annotation_method = "manual_review"
        rec.annotator_or_reviewer = reviewer
        rec.adjudication_status = status
        rec.rationale = rationale
        rec.validate()
        return rec

    def freeze_judgments(self,
                         output_json: str = "data/benchmark/v1/relevance_judgments.json",
                         output_csv: str = "data/benchmark/v1/relevance_judgments.csv"):
        """
        Freezes current judgments to versioned benchmark judgment store.
        """
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        judgments_list = [r.to_dict() for r in self.records.values()]

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(judgments_list, f, indent=2)

        # Flatten for CSV
        df_rows = []
        for d in judgments_list:
            row = dict(d)
            row["retrieved_by_systems"] = ", ".join(d["retrieved_by_systems"])
            row["system_ranks"] = json.dumps(d["system_ranks"])
            df_rows.append(row)

        df = pd.DataFrame(df_rows)
        df.to_csv(output_csv, index=False, encoding="utf-8")

        # Stats
        grade_counts = pd.Series([r.relevance_grade for r in self.records.values()]).value_counts(dropna=False).to_dict()
        status_counts = pd.Series([r.judgment_status for r in self.records.values()]).value_counts().to_dict()

        print(f"\nFrozen benchmark judgments saved to {output_json} and {output_csv}:", flush=True)
        print(f"  Total candidate pairs: {len(self.records):,}")
        print(f"  Relevance grade distribution: {grade_counts}")
        print(f"  Judgment status distribution: {status_counts}")

if __name__ == "__main__":
    adjudicator = BenchmarkAdjudicator()
    adjudicator.apply_rule_assisted_suggestions()
    adjudicator.freeze_judgments()
