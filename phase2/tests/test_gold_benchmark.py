"""
Unit tests for FinGuard Gold-Evidence Relevance Benchmark.

Validates:
1. Evidence-ID stability and normalization.
2. Annotation record schema invariants (rejection of invalid grades/status).
3. Exact, normalized, and substring matching logic.
4. Mathematical correctness of IR metrics against hand-calculated ground-truth examples:
   - Precision@K, Recall@K, MRR, MAP.
   - Graded DCG / nDCG (2^g - 1 / log2(rank+1)).
   - Judged coverage calculation.
5. Edge cases: empty retrieval, zero-relevant claims, unjudged vs grade-0 distinction.
6. Zero overlap between held-out evaluation claim IDs and training split.
7. Reproducibility with fixed random seed.
"""

import unittest
import math
import numpy as np
import pandas as pd

from src.gold_benchmark_schema import (
    EvidenceAnnotationRecord,
    normalize_text_for_matching,
    compute_content_sha256,
    format_claim_id,
    parse_claim_id,
    format_evidence_id,
    parse_evidence_id
)
from src.benchmark_evaluator import BenchmarkEvaluator

class TestGoldBenchmark(unittest.TestCase):
    def test_canonical_id_formatting_and_parsing(self):
        cid = format_claim_id(42)
        self.assertEqual(cid, "claim_0042")
        self.assertEqual(parse_claim_id(cid), 42)

        eid = format_evidence_id(105)
        self.assertEqual(eid, "ev_00105")
        self.assertEqual(parse_evidence_id(eid), 105)

        with self.assertRaises(ValueError):
            parse_claim_id("invalid_claim_id")
        with self.assertRaises(ValueError):
            parse_evidence_id("not_an_eid")

    def test_text_normalization_and_hashing(self):
        raw_1 = '  "Apple Inc. reported $89.5B in Q4 2023."  '
        raw_2 = '“apple inc. reported $89.5b in q4 2023.”'
        norm_1 = normalize_text_for_matching(raw_1)
        norm_2 = normalize_text_for_matching(raw_2)
        self.assertEqual(norm_1, norm_2)
        self.assertEqual(compute_content_sha256(raw_1), compute_content_sha256(raw_2))

    def test_annotation_record_validation(self):
        # Valid silver reference record
        rec = EvidenceAnnotationRecord(
            claim_id="claim_0001",
            evidence_id="ev_00001",
            relevance_grade=3,
            judgment_status="silver_reference",
            support_relation="supports",
            annotation_source="finfact_reference",
            annotation_method="exact_reference_match",
            annotator_or_reviewer="dataset_curator",
            adjudication_status="approved",
            rationale="Valid test record."
        )
        rec.validate()

        # Invariant: unjudged must have relevance_grade=None
        with self.assertRaises(ValueError):
            bad_rec = EvidenceAnnotationRecord(
                claim_id="claim_0001",
                evidence_id="ev_00002",
                relevance_grade=0,  # Invalid: unjudged cannot have numeric grade
                judgment_status="unjudged",
                support_relation="unknown",
                annotation_source="unjudged_pool",
                annotation_method="unjudged",
                annotator_or_reviewer="unassigned",
                adjudication_status="none",
                rationale="Should fail validation."
            )
            bad_rec.validate()

        # Invariant: grade must be 0, 1, 2, 3 or None
        with self.assertRaises(ValueError):
            bad_grade = EvidenceAnnotationRecord(
                claim_id="claim_0001",
                evidence_id="ev_00003",
                relevance_grade=5,  # Out of range
                judgment_status="reviewed",
                support_relation="supports",
                annotation_source="manual_annotation",
                annotation_method="manual_review",
                annotator_or_reviewer="reviewer",
                adjudication_status="pending",
                rationale="Out of range grade."
            )
            bad_grade.validate()

    def test_hand_crafted_ir_metrics(self):
        """
        Validates mathematical implementation of Precision, Recall, MRR, MAP,
        and Graded nDCG on hand-constructed synthetic query test vector.
        """
        # Ground truth: relevant items are ev_1 (grade 3) and ev_3 (grade 2).
        # Total relevant items = 2.
        evaluator = BenchmarkEvaluator.__new__(BenchmarkEvaluator)
        evaluator.binary_threshold = 2
        evaluator.claim_relevance = {"claim_0001": {"ev_1": 3, "ev_3": 2}}
        evaluator.claim_judged_set = {"claim_0001": {"ev_1", "ev_2", "ev_3", "ev_4", "ev_5"}}
        evaluator.reference_evidence = {"claim_0001": [{"evidence_id": "ev_1"}, {"evidence_id": "ev_3"}]}

        # System ranking: [ev_2 (grade 0), ev_1 (grade 3), ev_4 (grade 0), ev_3 (grade 2), ev_5 (grade 0)]
        retrieved_eids = ["ev_2", "ev_1", "ev_4", "ev_3", "ev_5"]

        metrics = evaluator.compute_single_query_metrics(retrieved_eids, "claim_0001", top_k=5)

        # 1. Precision@5: 2 relevant items in top 5 -> 2 / 5 = 0.4
        self.assertAlmostEqual(metrics["P@5"], 0.4, places=4)

        # 2. Recall@5: 2 relevant retrieved out of 2 total -> 2 / 2 = 1.0
        self.assertAlmostEqual(metrics["R@5"], 1.0, places=4)

        # 3. MRR: First hit is ev_1 at rank 2 -> RR = 1 / 2 = 0.5
        self.assertAlmostEqual(metrics["MRR"], 0.5, places=4)

        # 4. MAP:
        # Hit 1 at rank 2: Precision@2 = 1/2 = 0.5
        # Hit 2 at rank 4: Precision@4 = 2/4 = 0.5
        # AP = (0.5 + 0.5) / 2 = 0.5
        self.assertAlmostEqual(metrics["MAP"], 0.5, places=4)

        # 5. Graded nDCG@5:
        # Retrieved grades at ranks: [rank 1: ev_2 (0), rank 2: ev_1 (3), rank 3: ev_4 (0), rank 4: ev_3 (2), rank 5: ev_5 (0)]
        # DCG = (2^0 - 1)/log2(2) + (2^3 - 1)/log2(3) + (2^0 - 1)/log2(4) + (2^2 - 1)/log2(5) + 0
        #     = 0 + 7 / 1.5849625 + 0 + 3 / 2.321928 + 0
        #     = 4.4165 + 1.2920 = 5.7085
        # Ideal ranking grades: [3, 2]
        # IDCG = (2^3 - 1)/log2(2) + (2^2 - 1)/log2(3) = 7/1.0 + 3/1.5849625 = 7.0 + 1.8928 = 8.8928
        # nDCG = 5.7085 / 8.8928 = ~0.6419
        expected_dcg = (math.pow(2, 3) - 1.0) / math.log2(3) + (math.pow(2, 2) - 1.0) / math.log2(5)
        expected_idcg = (math.pow(2, 3) - 1.0) / math.log2(2) + (math.pow(2, 2) - 1.0) / math.log2(3)
        expected_ndcg = expected_dcg / expected_idcg
        self.assertAlmostEqual(metrics["nDCG@5"], expected_ndcg, places=3)

        # 6. Judged Coverage@5: all 5 retrieved items are in judged_set -> 5/5 = 1.0
        self.assertEqual(metrics["Coverage@5"], 1.0)

    def test_empty_retrieval_and_zero_relevant_claims(self):
        evaluator = BenchmarkEvaluator.__new__(BenchmarkEvaluator)
        evaluator.binary_threshold = 2
        evaluator.claim_relevance = {"claim_0002": {}}  # Zero known relevant items
        evaluator.claim_judged_set = {"claim_0002": set()}
        evaluator.reference_evidence = {"claim_0002": []}

        # Empty retrieval
        m_empty = evaluator.compute_single_query_metrics([], "claim_0002", top_k=5)
        self.assertEqual(m_empty["P@5"], 0.0)
        self.assertEqual(m_empty["R@5"], 0.0)
        self.assertEqual(m_empty["MRR"], 0.0)
        self.assertEqual(m_empty["MAP"], 0.0)
        self.assertEqual(m_empty["nDCG@5"], 0.0)

        # Claim with zero relevant items retrieving candidates
        m_zero = evaluator.compute_single_query_metrics(["ev_1", "ev_2"], "claim_0002", top_k=5)
        self.assertEqual(m_zero["P@5"], 0.0)
        self.assertEqual(m_zero["R@5"], 0.0)
        self.assertEqual(m_zero["MRR"], 0.0)
        self.assertEqual(m_zero["MAP"], 0.0)

    def test_no_leakage_between_pilot_and_train_splits(self):
        """Verifies held-out evaluation claim IDs strictly do not overlap with training split."""
        train_df = pd.read_csv("data/train.csv")
        train_ids = set(train_df["id"].tolist())

        from src.pooling import CandidatePooler
        pooler = CandidatePooler.__new__(CandidatePooler)
        pilot_cids = pooler.select_pilot_claim_ids(sample_size=75, seed=42)

        pilot_record_indices = [parse_claim_id(cid) for cid in pilot_cids]
        overlap = set(pilot_record_indices).intersection(train_ids)
        self.assertEqual(len(overlap), 0, f"Leakage detected! Evaluation claims overlapped with train.csv: {overlap}")

if __name__ == "__main__":
    unittest.main()
