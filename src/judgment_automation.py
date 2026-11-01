"""
FinGuard Research-Grounded Relevance Judgment Automation Pipeline.

Implements the LARA-inspired (Takehi et al., SIGIR 2025) and uncertainty-aware
dual-assessor relevance assessment workflow for financial claim-evidence pairs.

Core Features:
1. Strict Provenance & Schema Validation: Retains canonical IDs, full text, system origins,
   assessor versions, confidence/uncertainty reasons, and adjudication status.
2. Financial Relevance Rubric: Evaluates entity, metric, fiscal period, numerical/scale,
   and directional polarity dimensions.
3. Dual-Assessor Architecture:
   - Assessor 1: Cross-Encoder joint sequence classification (continuous relevance score).
   - Assessor 2: Rule-assisted financial entity and metric alignment engine.
4. Agreement & Uncertainty Analysis: Quantifies assessor concordance, flags disagreements
   and borderline cases for human adjudication.
5. Selective Cost-Aware Human Review Queue: Ranks ambiguous/contested pairs by expected
   information gain for targeted human review (CSV/JSONL export/import).
6. Non-Leakage & Integrity Guarantees: Never uses claim True/False/NEI veracity labels
   to assess evidence relevance; preserves frozen v1 immutability by outputting to v2.
"""

import os
import sys
import json
import math
import time
import hashlib
from datetime import datetime, timezone
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gold_benchmark_schema import (
    EvidenceAnnotationRecord,
    normalize_text_for_matching,
    format_claim_id,
    format_evidence_id,
    parse_claim_id,
    parse_evidence_id,
    VALID_RELEVANCE_GRADES,
    VALID_SUPPORT_RELATIONS,
    BENCHMARK_VERSION
)
from src.claim_understanding import extract_financial_entities
from src.reranker import PretrainedCrossEncoderReranker

AUTOMATION_PIPELINE_VERSION = "2.0.0"

class BaseRelevanceAssessor:
    """Abstract base class for relevance assessors."""
    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version

    def assess_pair(self, claim_text: str, evidence_text: str) -> Dict[str, Any]:
        raise NotImplementedError

class CrossEncoderAssessor(BaseRelevanceAssessor):
    """
    Assessor 1: Neural Cross-Encoder joint attention relevance scorer.
    Uses 'cross-encoder/ms-marco-MiniLM-L-6-v2' to evaluate direct query-passage alignment.
    """
    def __init__(self, reranker: Optional[PretrainedCrossEncoderReranker] = None):
        super().__init__(name="cross_encoder_minilm_v2", version="ms-marco-MiniLM-L-6-v2")
        self.reranker = reranker or PretrainedCrossEncoderReranker()

    def assess_pair(self, claim_text: str, evidence_text: str) -> Dict[str, Any]:
        if not self.reranker or not self.reranker.is_available:
            return {
                "score": 0.5,
                "confidence": 0.0,
                "is_available": False,
                "reasoning": "Cross-encoder unavailable; operating in neutral fallback mode."
            }

        candidate = [{"doc_id": 0, "text": evidence_text}]
        ranked = self.reranker.rerank(claim_text, candidate, top_k=1)
        if ranked:
            score = float(ranked[0].get("rerank_score", 0.0))
        else:
            score = 0.0

        return {
            "score": score,
            "confidence": round(abs(score - 0.5) * 2.0, 4), # Higher distance from 0.5 = higher confidence
            "is_available": True,
            "reasoning": f"Cross-encoder joint attention score: {score:.4f}."
        }

    def assess_batch(self, pairs: List[List[str]], batch_size: int = 64) -> List[Dict[str, Any]]:
        """Scores a batch of [claim_text, evidence_text] pairs efficiently."""
        if not self.reranker or not self.reranker.is_available or not pairs:
            return [{
                "score": 0.5,
                "confidence": 0.0,
                "is_available": False,
                "reasoning": "Cross-encoder unavailable; neutral fallback."
            } for _ in pairs]

        import torch
        scores = []
        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                encoded = self.reranker.tokenizer(
                    batch_pairs, padding=True, truncation=True, max_length=160, return_tensors='pt'
                ).to(self.reranker.device)
                logits = self.reranker.model(**encoded).logits.flatten()
                probs = torch.sigmoid(logits).cpu().numpy().tolist()
                scores.extend(probs)

        results = []
        for s in scores:
            score = float(s)
            results.append({
                "score": round(score, 4),
                "confidence": round(abs(score - 0.5) * 2.0, 4),
                "is_available": True,
                "reasoning": f"Cross-encoder joint attention score: {score:.4f}."
            })
        return results

class FinancialRubricAssessor(BaseRelevanceAssessor):
    """
    Assessor 2: Financial domain rubric and entity-metric alignment engine.
    Evaluates entity match, financial metric match, fiscal period, numerical values, and polarity.
    """
    def __init__(self):
        super().__init__(name="financial_rubric_engine_v2", version="2.0.0")

    def assess_pair(self, claim_text: str, evidence_text: str) -> Dict[str, Any]:
        norm_ev = normalize_text_for_matching(evidence_text)
        parsed = extract_financial_entities(claim_text)

        entities = parsed.get("organizations", []) or parsed.get("entities", [])
        metrics = parsed.get("financial_metrics", [])
        monetary = parsed.get("monetary_amounts", [])
        percentages = parsed.get("percentages", [])
        periods = parsed.get("periods", []) or parsed.get("fiscal_periods", [])

        # Additional fallback: extract raw currency and numerical tokens from claim text
        if not monetary:
            raw_nums = re.findall(r'[\$€£₹]\s*\d+(?:,\d{3})*(?:\.\d+)?|\b\d+(?:,\d{3})*(?:\.\d+)?\b', claim_text)
            monetary = [n.strip() for n in raw_nums if n.strip() not in periods]

        # Match dimensions
        matched_entities = [e for e in entities if normalize_text_for_matching(e) in norm_ev]
        matched_monetary = [m for m in monetary if normalize_text_for_matching(m) in norm_ev]
        matched_pct = [p for p in percentages if normalize_text_for_matching(p) in norm_ev]
        matched_periods = [t for t in periods if normalize_text_for_matching(t) in norm_ev]
        matched_metrics = [m for m in metrics if normalize_text_for_matching(m) in norm_ev]

        has_entity = len(matched_entities) > 0
        has_metric = len(matched_metrics) > 0
        has_num = len(matched_monetary) > 0 or len(matched_pct) > 0
        has_period = len(matched_periods) > 0

        # Infer directional support relation
        relation = self._infer_relation(norm_ev, parsed)

        # Rubric decision logic
        if has_entity and has_num and (has_period or has_metric):
            grade = 3
            rationale = (f"Directly Decisive: Confirmed entity ({matched_entities}), "
                         f"numerical values ({matched_monetary + matched_pct}), and metric/period ({matched_periods + matched_metrics}).")
            confidence = 0.90
        elif has_entity and (has_num or has_metric):
            grade = 2
            rationale = (f"Substantially Relevant: Entity confirmed ({matched_entities}) with financial metric/numerical overlap, "
                         f"but lacks full temporal/period resolution.")
            confidence = 0.75
        elif has_entity or (has_metric and has_num):
            grade = 1
            rationale = "Contextually Relevant: Topical background mentioning entity or general market sector."
            confidence = 0.60
        else:
            grade = 0
            rationale = "Not Relevant: No verified entity or financial metric overlap with claim assertion."
            confidence = 0.85

        return {
            "grade": grade,
            "confidence": confidence,
            "support_relation": relation,
            "rationale": rationale,
            "matched_dimensions": {
                "entities": matched_entities,
                "metrics": matched_metrics,
                "monetary": matched_monetary,
                "percentages": matched_pct,
                "periods": matched_periods
            }
        }

    def _infer_relation(self, norm_ev: str, parsed_claim: Dict[str, Any]) -> str:
        neg_cues = ["not", "never", "false", "debunked", "untrue", "denied", "no evidence", "refuted", "inaccurate"]
        pos_cues = ["confirmed", "announced", "reported", "shows that", "official records", "increased by", "earned", "paid"]

        has_neg = any(w in norm_ev for w in neg_cues)
        has_pos = any(w in norm_ev for w in pos_cues)

        if has_neg and not has_pos:
            return "refutes"
        elif has_pos and not has_neg:
            return "supports"
        elif has_neg and has_pos:
            return "mixed"
        return "context_only"

class RelevanceJudgmentAutomator:
    """
    Orchestrates the research-grounded automated assessment pipeline,
    dual-assessor agreement analysis, and selective human-review queue generation.
    """
    def __init__(self,
                 input_judgments_json: str = "data/benchmark/v1/relevance_judgments.json",
                 output_dir: str = "data/benchmark/v2",
                 cross_encoder_assessor: Optional[CrossEncoderAssessor] = None,
                 rubric_assessor: Optional[FinancialRubricAssessor] = None):
        self.input_json = input_judgments_json
        self.output_dir = output_dir
        self.assessor1 = cross_encoder_assessor or CrossEncoderAssessor()
        self.assessor2 = rubric_assessor or FinancialRubricAssessor()

        self.records: Dict[str, Dict[str, Any]] = {}
        self._load_input_pool()

    def _load_input_pool(self):
        if not os.path.exists(self.input_json):
            raise FileNotFoundError(f"Input judgments file not found: {self.input_json}")

        with open(self.input_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        for d in data:
            key = f"{d['claim_id']}::{d['evidence_id']}"
            self.records[key] = dict(d)

        print(f"Loaded {len(self.records):,} pooled candidate records from {self.input_json}.", flush=True)

    def execute_assessment_pipeline(self) -> Dict[str, Any]:
        """
        Executes the uncertainty-aware dual-assessor pipeline over all unjudged candidates.
        """
        print(f"\n=== Executing FinGuard Relevance Automation Pipeline v{AUTOMATION_PIPELINE_VERSION} ===", flush=True)
        start_time = time.time()

        assessed_count = 0
        silver_count = 0
        accepted_count = 0
        unjudged_queue_count = 0
        disagreement_count = 0

        # Stats tracking
        assessor1_scores = []
        assessor2_grades = []
        agreement_records = []

        total_pairs = len(self.records)
        # Step 1: Pre-collect unjudged pairs for batch neural scoring
        pairs_to_score = []
        keys_to_score = []
        for key, rec in self.records.items():
            if rec.get("judgment_status") == "silver_reference" or rec.get("adjudication_status") == "approved":
                continue
            c_text = rec.get("claim_text", "")
            e_text = rec.get("evidence_text", "")
            if c_text and e_text:
                pairs_to_score.append([c_text, e_text])
                keys_to_score.append(key)

        print(f"Batch scoring {len(pairs_to_score):,} candidate pairs with neural Cross-Encoder (batch size 64)...", flush=True)
        ce_batch_results = self.assessor1.assess_batch(pairs_to_score, batch_size=64) if hasattr(self.assessor1, "assess_batch") else [self.assessor1.assess_pair(p[0], p[1]) for p in pairs_to_score]
        ce_map = dict(zip(keys_to_score, ce_batch_results))
        print("Neural Cross-Encoder batch scoring complete!", flush=True)

        processed = 0
        for key, rec in self.records.items():
            processed += 1
            if processed % 1000 == 0 or processed == total_pairs:
                print(f"  Evaluating candidate rules & concordance: {processed:,} / {total_pairs:,}...", flush=True)

            # Preserve silver-reference dataset evidence without alteration
            if rec.get("judgment_status") == "silver_reference":
                silver_count += 1
                continue

            # Preserve already approved manual adjudications
            if rec.get("adjudication_status") == "approved":
                continue

            claim_text = rec.get("claim_text", "")
            evidence_text = rec.get("evidence_text", "")

            # Deterministic check: missing text cannot be assessed
            if not claim_text or not evidence_text:
                rec["relevance_grade"] = None
                rec["judgment_status"] = "unjudged"
                rec["support_relation"] = "unknown"
                rec["rationale"] = "Deterministic check: Missing claim or evidence text."
                unjudged_queue_count += 1
                continue

            # Retrieve Cross-Encoder score and run Financial Rubric Assessor
            res1 = ce_map.get(key, {"score": 0.0, "confidence": 0.0, "is_available": False, "reasoning": "Unscored"})
            res2 = self.assessor2.assess_pair(claim_text, evidence_text)

            score_ce = res1["score"]
            grade_rule = res2["grade"]
            rel_rule = res2["support_relation"]

            assessor1_scores.append(score_ce)
            assessor2_grades.append(grade_rule)
            assessed_count += 1

            # Compute concordance & disagreement
            # Map rule grade to [0, 1] scale for continuous comparison
            normalized_grade_rule = grade_rule / 3.0
            disagreement_score = round(abs(score_ce - normalized_grade_rule), 4)

            # Information Gain Priority for Human Review Queue:
            # High priority if:
            # 1. High disagreement between assessors
            # 2. Retrieved at high rank by retrievers
            # 3. Retrieved by multiple systems
            ranks = list(rec.get("system_ranks", {}).values())
            best_rank = min(ranks) if ranks else 20
            rank_weight = 1.0 / math.sqrt(best_rank) # rank 1 -> 1.0, rank 4 -> 0.5, rank 20 -> 0.22
            sys_consensus = len(rec.get("retrieved_by_systems", [])) / 8.0

            priority_score = round(0.50 * disagreement_score + 0.35 * rank_weight + 0.15 * sys_consensus, 4)

            # Structured metadata recording
            rec["assessor1_model"] = self.assessor1.version
            rec["assessor1_score"] = round(score_ce, 4)
            rec["assessor2_model"] = self.assessor2.version
            rec["assessor2_grade"] = grade_rule
            rec["disagreement_score"] = disagreement_score
            rec["review_priority_score"] = priority_score

            # Acceptance and Uncertainty Decision Rules
            # Case A: High-Concordance Decisive/Substantial Relevant
            if score_ce >= 0.70 and grade_rule >= 2:
                final_grade = 3 if grade_rule == 3 else 2
                rec["relevance_grade"] = final_grade
                rec["judgment_status"] = "automatically_accepted"
                rec["support_relation"] = rel_rule
                rec["annotation_source"] = "model_assisted"
                rec["annotation_method"] = "dual_assessor_high_concordance"
                rec["annotator_or_reviewer"] = "finguard_dual_assessor_v2"
                rec["adjudication_status"] = "pending"
                rec["rationale"] = (f"Dual-Assessor Concordant Relevant: Cross-Encoder score {score_ce:.4f} "
                                    f"and Financial Rubric Grade {grade_rule}. {res2['rationale']}")
                accepted_count += 1

            # Case B: High-Concordance Contextual Relevant
            elif (0.40 <= score_ce < 0.70) and grade_rule == 1:
                rec["relevance_grade"] = 1
                rec["judgment_status"] = "automatically_accepted"
                rec["support_relation"] = "context_only"
                rec["annotation_source"] = "model_assisted"
                rec["annotation_method"] = "dual_assessor_high_concordance"
                rec["annotator_or_reviewer"] = "finguard_dual_assessor_v2"
                rec["adjudication_status"] = "pending"
                rec["rationale"] = (f"Dual-Assessor Concordant Contextual: Cross-Encoder score {score_ce:.4f} "
                                    f"and Financial Rubric Grade 1. Mentions financial sector/entity in broader context.")
                accepted_count += 1

            # Case C: High-Concordance Confirmed Non-relevant
            elif score_ce < 0.15 and grade_rule == 0:
                rec["relevance_grade"] = 0
                rec["judgment_status"] = "automatically_accepted"
                rec["support_relation"] = "unknown"
                rec["annotation_source"] = "model_assisted"
                rec["annotation_method"] = "dual_assessor_high_concordance"
                rec["annotator_or_reviewer"] = "finguard_dual_assessor_v2"
                rec["adjudication_status"] = "pending"
                rec["rationale"] = (f"Dual-Assessor Concordant Non-Relevant: Cross-Encoder score {score_ce:.4f} "
                                    f"and Financial Rubric Grade 0. Zero entity or metric overlap.")
                accepted_count += 1

            # Case D: Disagreement or Borderline Uncertainty -> Explicitly Preserve UNJUDGED
            else:
                disagreement_count += 1
                unjudged_queue_count += 1
                rec["relevance_grade"] = None
                rec["judgment_status"] = "unjudged"
                rec["support_relation"] = "unknown"
                rec["annotation_source"] = "unjudged_pool"
                rec["annotation_method"] = "unjudged"
                rec["annotator_or_reviewer"] = "unassigned"
                rec["adjudication_status"] = "none"

                uncertainty_reason = []
                if (score_ce >= 0.60 and grade_rule == 0):
                    uncertainty_reason.append("Neural model detects high relevance but rule engine finds no entity/metric match")
                elif (score_ce < 0.25 and grade_rule >= 2):
                    uncertainty_reason.append("Rule engine finds entity/metric match but neural cross-encoder finds low query alignment")
                elif 0.15 <= score_ce < 0.40:
                    uncertainty_reason.append("Borderline cross-encoder score in uncertainty corridor [0.15, 0.40)")
                else:
                    uncertainty_reason.append("Partial signal requiring human arbitration")

                rec["rationale"] = (f"Uncertainty Flag: {'; '.join(uncertainty_reason)}. "
                                    f"Cross-Encoder score: {score_ce:.4f}, Rule Grade: {grade_rule}. "
                                    f"Disagreement score: {disagreement_score:.4f}.")

        elapsed = time.time() - start_time
        print(f"\nPipeline execution finished in {elapsed:.2f}s.", flush=True)
        print(f"  Total candidate pairs: {total_pairs:,}")
        print(f"  Preserved Silver References (Grade 3): {silver_count:,}")
        print(f"  Automatically Accepted (Concordant Grades 0-3): {accepted_count:,}")
        print(f"  Preserved Unjudged (Targeted for Human Review Queue): {unjudged_queue_count:,}")
        print(f"  Assessor Disagreements / Borderline Cases: {disagreement_count:,}")

        return {
            "total_pairs": total_pairs,
            "silver_count": silver_count,
            "accepted_count": accepted_count,
            "unjudged_count": unjudged_queue_count,
            "disagreement_count": disagreement_count,
            "elapsed_seconds": round(elapsed, 2)
        }

    def export_benchmark_v2(self,
                            output_dir: Optional[str] = None,
                            review_queue_csv_name: str = "human_review_queue.csv",
                            judgments_json_name: str = "relevance_judgments.json",
                            judgments_csv_name: str = "relevance_judgments.csv") -> Dict[str, str]:
        """
        Exports versioned Benchmark v2 artifacts:
        - Full relevance judgments store (JSON and CSV)
        - Targeted Human Review Queue sorted by Information Gain priority (CSV and JSONL)
        """
        out_dir = output_dir or self.output_dir
        os.makedirs(out_dir, exist_ok=True)

        records_list = list(self.records.values())

        # 1. Save Full Relevance Judgments (JSON)
        json_path = os.path.join(out_dir, judgments_json_name)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records_list, f, indent=2)

        # 2. Save Full Relevance Judgments (CSV)
        csv_path = os.path.join(out_dir, judgments_csv_name)
        df_full = []
        for d in records_list:
            row = dict(d)
            if isinstance(row.get("retrieved_by_systems"), list):
                row["retrieved_by_systems"] = ", ".join(row["retrieved_by_systems"])
            if isinstance(row.get("system_ranks"), dict):
                row["system_ranks"] = json.dumps(row["system_ranks"])
            df_full.append(row)
        pd.DataFrame(df_full).to_csv(csv_path, index=False, encoding="utf-8")

        # 3. Export Targeted Human Review Queue
        # Filter records that remain unjudged, sorted by review_priority_score descending
        unjudged_records = [r for r in records_list if r.get("judgment_status") == "unjudged"]
        unjudged_records.sort(key=lambda x: x.get("review_priority_score", 0.0), reverse=True)

        queue_csv_path = os.path.join(out_dir, review_queue_csv_name)
        queue_jsonl_path = os.path.join(out_dir, review_queue_csv_name.replace(".csv", ".jsonl"))

        queue_rows = []
        for r in unjudged_records:
            row = {
                "claim_id": r["claim_id"],
                "evidence_id": r["evidence_id"],
                "review_priority_score": r.get("review_priority_score", 0.0),
                "disagreement_score": r.get("disagreement_score", 0.0),
                "assessor1_score": r.get("assessor1_score"),
                "assessor2_grade": r.get("assessor2_grade"),
                "claim_text": r.get("claim_text", ""),
                "evidence_text": r.get("evidence_text", ""),
                "relevance_grade": "", # Empty for reviewer input (0, 1, 2, 3)
                "support_relation": "", # Empty for reviewer input (supports, refutes, mixed, context_only)
                "annotator_or_reviewer": "human_reviewer",
                "adjudication_status": "pending",
                "human_rationale": "",
                "source_url": r.get("source_url", ""),
                "retrieved_by_systems": ", ".join(r.get("retrieved_by_systems", [])) if isinstance(r.get("retrieved_by_systems"), list) else r.get("retrieved_by_systems", ""),
                "system_ranks": json.dumps(r.get("system_ranks", {})) if isinstance(r.get("system_ranks"), dict) else r.get("system_ranks", "")
            }
            queue_rows.append(row)

        df_queue = pd.DataFrame(queue_rows)
        df_queue.to_csv(queue_csv_path, index=False, encoding="utf-8")

        with open(queue_jsonl_path, "w", encoding="utf-8") as f:
            for row in queue_rows:
                f.write(json.dumps(row) + "\n")

        print(f"\nBenchmark v2 artifacts exported successfully to {out_dir}:")
        print(f"  Judgments JSON: {json_path} ({len(records_list):,} records)")
        print(f"  Judgments CSV:  {csv_path}")
        print(f"  Review Queue CSV:   {queue_csv_path} ({len(queue_rows):,} unjudged pairs prioritized)")
        print(f"  Review Queue JSONL: {queue_jsonl_path}")

        return {
            "judgments_json": json_path,
            "judgments_csv": csv_path,
            "review_queue_csv": queue_csv_path,
            "review_queue_jsonl": queue_jsonl_path
        }

    def import_human_reviews(self, review_csv_path: str) -> int:
        """
        Imports human-reviewed decisions from an edited Review Queue CSV.
        Validates schema rules, updates status to 'human_adjudicated' or 'reviewed',
        and preserves provenance.
        """
        if not os.path.exists(review_csv_path):
            raise FileNotFoundError(f"Review CSV not found: {review_csv_path}")

        df = pd.read_csv(review_csv_path)
        imported_count = 0

        for _, row in df.iterrows():
            cid = str(row["claim_id"]).strip()
            eid = str(row["evidence_id"]).strip()
            key = f"{cid}::{eid}"

            if key not in self.records:
                continue

            raw_grade = row.get("relevance_grade")
            if pd.isna(raw_grade) or str(raw_grade).strip() == "":
                continue # Still unreviewed

            grade = int(raw_grade)
            if grade not in {0, 1, 2, 3}:
                raise ValueError(f"Invalid relevance_grade '{grade}' for {key}. Must be 0, 1, 2, or 3.")

            relation = str(row.get("support_relation", "context_only")).strip()
            if relation not in VALID_SUPPORT_RELATIONS:
                relation = "context_only"

            reviewer = str(row.get("annotator_or_reviewer", "human_reviewer")).strip()
            human_rationale = str(row.get("human_rationale", "")).strip() or "Human verified relevance assessment."

            rec = self.records[key]
            rec["relevance_grade"] = grade
            rec["judgment_status"] = "adjudicated"
            rec["support_relation"] = relation
            rec["annotation_source"] = "manual_annotation"
            rec["annotation_method"] = "human_review"
            rec["annotator_or_reviewer"] = reviewer
            rec["adjudication_status"] = "approved"
            rec["rationale"] = f"Human Adjudication: {human_rationale}"

            imported_count += 1

        print(f"Imported {imported_count:,} human-adjudicated judgments from {review_csv_path}.", flush=True)
        return imported_count

if __name__ == "__main__":
    automator = RelevanceJudgmentAutomator(
        input_judgments_json="data/benchmark/v1/relevance_judgments.json",
        output_dir="data/benchmark/v2"
    )
    automator.execute_assessment_pipeline()
    automator.export_benchmark_v2()
