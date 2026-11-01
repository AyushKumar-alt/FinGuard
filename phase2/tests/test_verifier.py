"""
Unit tests for FinancialClaimVerifier: baseline inference, uncertainty detection,
and missing evidence handling.
"""

import unittest
from src.verifier import FinancialClaimVerifier

class TestFinancialClaimVerifier(unittest.TestCase):
    def setUp(self):
        self.verifier = FinancialClaimVerifier(
            baseline_model_path="outputs/best_baseline_model.pkl",
            use_transformer=False
        )

    def test_missing_evidence_triggers_uncertainty(self):
        claim = "Did inflation drop in 2023?"
        # Empty evidence
        res = self.verifier.verify_claim(claim, evidence_text="")
        self.assertEqual(res["predicted_label"], "NEI", "Missing evidence must yield NEI verdict.")
        self.assertTrue(res["is_uncertain"], "Missing evidence must trigger is_uncertain flag.")
        self.assertIsNotNone(res["uncertainty_reason"])

    def test_valid_evidence_prediction_and_normalization(self):
        claim = "Did Biden call for an end to shareholder capitalism?"
        evidence = "Although those details were not available, it's clear that Biden was not calling for the abolition of shareholding itself."
        res = self.verifier.verify_claim(claim, evidence_text=evidence)
        
        self.assertIn(res["predicted_label"], ["true", "false", "NEI"])
        self.assertIn("class_probabilities", res)
        probs = res["class_probabilities"]
        prob_sum = probs["false"] + probs["true"] + probs["NEI"]
        self.assertAlmostEqual(prob_sum, 1.0, places=2, msg="Probabilities must sum to 1.0.")
        self.assertFalse(res["is_uncertain"])
        self.assertIn("TF-IDF", res["model_engine"])

    def test_nli_aggregate_probabilities_remain_valid_for_independent_maxima(self):
        self.verifier.has_transformer = True
        scores = iter([
            {"entailment": 0.8741, "contradiction": 0.0154, "neutral": 0.1105},
            {"entailment": 0.2000, "contradiction": 0.3288, "neutral": 0.4712},
        ])
        self.verifier.predict_nli_single_pair = lambda claim, sentence: next(scores)

        result = self.verifier.verify_claim(
            claim="Company revenue increased by 10 percent in 2024.",
            evidence_docs=[
                {"text": "Revenue increased by ten percent in 2024.", "source_urls": [], "doc_id": 1},
                {"text": "The company reported additional financial results.", "source_urls": [], "doc_id": 2},
            ],
            engine_choice="transformer_nli",
        )

        probabilities = result["class_probabilities"]
        self.assertTrue(all(0.0 <= value <= 1.0 for value in probabilities.values()))
        self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=3)
        self.assertEqual(result["verdict"], "true")

    def test_evidence_ranking_api(self):
        from src.transformer_model import ClaimVerifier
        cv = ClaimVerifier()
        claim = "national debt increase"
        candidates = [
            "the treasury department issued monthly figures",
            "national debt increased substantially over four years",
            "unemployment fell to historical lows"
        ]
        ranked = cv.rank_evidence_sentences(claim, candidates)
        self.assertEqual(len(ranked), 3)
        # Candidate 1 contains both 'national' and 'debt'
        self.assertIn("national debt", ranked[0][0])
        self.assertGreater(ranked[0][1], ranked[2][1])

if __name__ == "__main__":
    unittest.main()
