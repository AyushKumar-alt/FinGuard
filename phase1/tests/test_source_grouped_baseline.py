"""
Unit tests for the reproducible, isolated source-grouped baseline experiments.
Verifies file generation, schema compliance, and metric boundaries.
"""

import os
import json
import unittest
import joblib

EXPERIMENT_DIR = os.path.join("outputs", "experiments", "source_grouped_baseline")

class TestSourceGroupedBaseline(unittest.TestCase):
    def test_artifacts_exist(self):
        self.assertTrue(os.path.exists(EXPERIMENT_DIR), "Experiment directory must exist.")
        self.assertTrue(
            os.path.exists(os.path.join(EXPERIMENT_DIR, "results.json")),
            "results.json must exist."
        )
        self.assertTrue(
            os.path.exists(os.path.join(EXPERIMENT_DIR, "experiment.log")),
            "experiment.log must exist."
        )
        self.assertTrue(
            os.path.exists(os.path.join(EXPERIMENT_DIR, "experiment_a_raw_model.joblib")),
            "Raw model checkpoint must exist."
        )
        self.assertTrue(
            os.path.exists(os.path.join(EXPERIMENT_DIR, "experiment_b_cleaned_model.joblib")),
            "Cleaned model checkpoint must exist."
        )

    def test_metrics_schema_and_counts(self):
        results_path = os.path.join(EXPERIMENT_DIR, "results.json")
        with open(results_path, "r", encoding="utf-8") as f:
            res = json.load(f)
            
        self.assertIn("split_counts", res)
        self.assertEqual(res["split_counts"]["train"], 2357)
        self.assertEqual(res["split_counts"]["val"], 337)
        self.assertEqual(res["split_counts"]["test"], 675)
        self.assertEqual(res["split_counts"]["total"], 3369)
        
        for exp_key in ["experiment_a_raw", "experiment_b_cleaned"]:
            self.assertIn(exp_key, res)
            test_m = res[exp_key]["test_metrics"]
            self.assertGreater(test_m["accuracy"], 0.50)
            self.assertGreater(test_m["macro_f1"], 0.50)
            self.assertEqual(
                test_m["per_class"]["false"]["support"] +
                test_m["per_class"]["true"]["support"] +
                test_m["per_class"]["NEI"]["support"],
                675
            )

    def test_model_inference_pipeline(self):
        model_a = joblib.load(os.path.join(EXPERIMENT_DIR, "experiment_a_raw_model.joblib"))
        sample_text = ["Claim: Apple reported $90B revenue. Evidence: Apple Inc. declared $90B in sales."]
        preds = model_a.predict(sample_text)
        self.assertEqual(len(preds), 1)
        self.assertIn(preds[0], [0, 1, 2])

if __name__ == "__main__":
    unittest.main()

