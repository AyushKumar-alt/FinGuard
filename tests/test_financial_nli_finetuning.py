"""
Unit and Integration Tests for FinGuard Phase 2 Financial Domain NLI Fine-Tuning.

Tests:
1. Pair construction schema & provenance integrity
2. Strict split non-leakage & claim clumping
3. Explicit label mapping (0=contradiction, 1=entailment, 2=neutral)
4. Model weights & configuration loading
5. FinancialClaimVerifier inference compatibility
6. FinGuardPipeline backward compatibility
"""

import os
import sys
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

class TestFinancialNLIFineTuning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = ROOT / "phase2" / "data" / "financial_nli"
        cls.model_dir = ROOT / "models" / "finetuned_nli"
        cls.results_dir = ROOT / "phase2" / "results" / "financial_nli"

    def test_split_integrity_and_zero_leakage(self):
        """Verifies zero claim overlap across train, validation, and test splits."""
        stats_file = self.data_dir / "dataset_statistics.json"
        self.assertTrue(stats_file.exists(), "dataset_statistics.json must exist")
        with open(stats_file, "r", encoding="utf-8") as f:
            stats = json.load(f)
        
        self.assertEqual(stats["integrity_checks"]["train_val_claim_overlap"], 0)
        self.assertEqual(stats["integrity_checks"]["train_test_claim_overlap"], 0)
        self.assertFalse(stats["integrity_checks"]["test_claims_used_in_training"])

    def test_pair_construction_schema(self):
        """Verifies premise, claim, label, and provenance schema in generated JSONL files."""
        for split in ["train.jsonl", "validation.jsonl"]:
            split_path = self.data_dir / split
            self.assertTrue(split_path.exists(), f"{split} must exist")
            with open(split_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                self.assertTrue(bool(first_line), f"{split} must not be empty")
                record = json.loads(first_line)
                
                self.assertIn("premise", record)
                self.assertIn("claim", record)
                self.assertIn("label", record)
                self.assertIn("label_name", record)
                self.assertIn("provenance_rule", record)
                self.assertIn("confidence", record)
                self.assertIn(record["label"], [0, 1, 2])
                self.assertIn(record["label_name"], ["contradiction", "entailment", "neutral"])

    def test_provenance_registry(self):
        """Verifies labeling provenance definitions."""
        prov_file = self.data_dir / "labeling_provenance.json"
        self.assertTrue(prov_file.exists(), "labeling_provenance.json must exist")
        with open(prov_file, "r", encoding="utf-8") as f:
            prov = json.load(f)
        
        self.assertIn("R_ENTAIL_HIGH_OVERLAP", prov)
        self.assertIn("R_CONTRA_EXPLICIT_REFUTATION", prov)
        self.assertIn("R_NEUTRAL_ARTICLE_CONTEXT", prov)

    def test_finetuned_model_artifacts_exist(self):
        """Verifies fine-tuned model checkpoint weights and configuration exist on disk."""
        self.assertTrue((self.model_dir / "config.json").exists())
        self.assertTrue((self.model_dir / "model.safetensors").exists())
        self.assertTrue((self.model_dir / "tokenizer.json").exists())
        self.assertTrue((self.model_dir / "training_history.json").exists())

    def test_finetuned_config_label_mapping(self):
        """Verifies checkpoint label IDs match expected NLI semantics."""
        config_path = self.model_dir / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            
        id2label = config.get("id2label", {})
        # JSON keys are strings
        self.assertEqual(id2label.get("0", id2label.get(0)), "contradiction")
        self.assertEqual(id2label.get("1", id2label.get(1)), "entailment")
        self.assertEqual(id2label.get("2", id2label.get(2)), "neutral")

    def test_verifier_alias_loading_and_inference(self):
        """Verifies FinancialClaimVerifier loads fine-tuned model and produces valid probabilities."""
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        
        from src.verifier import FinancialClaimVerifier
        verifier = FinancialClaimVerifier(
            nli_model_name="financial_finetuned",
            use_transformer=True
        )
        self.assertTrue(verifier.has_transformer)
        self.assertEqual(verifier.id2label_nli[0], "contradiction")
        self.assertEqual(verifier.id2label_nli[1], "entailment")
        self.assertEqual(verifier.id2label_nli[2], "neutral")

        # Test single pair inference
        claim = "Tesla quarterly automotive revenue reached $20 billion."
        evidence = "In Q3, Tesla reported automotive revenues exceeding 21 billion dollars, surpassing estimates."
        scores = verifier.predict_nli_single_pair(claim, evidence)
        
        self.assertIn("entailment", scores)
        self.assertIn("contradiction", scores)
        self.assertIn("neutral", scores)
        self.assertAlmostEqual(sum(scores.values()), 1.0, places=2)
        # Entailment should be positive
        self.assertGreater(scores["entailment"], 0.3)

    def test_pipeline_backward_compatibility(self):
        """Verifies FinGuardPipeline default behavior remains intact."""
        from src.pipeline import FinGuardPipeline
        # Test initialization with light models to verify default wiring
        pipeline = FinGuardPipeline(json_path="finfact.json", load_heavy_models=False)
        self.assertEqual(pipeline.verifier.nli_model_name, "cross-encoder/nli-distilroberta-base")
        
        # Test pipeline execution on empty claim
        res = pipeline.run("")
        self.assertEqual(res["verdict"], "NEI")

if __name__ == "__main__":
    unittest.main()

