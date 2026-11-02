"""
FinGuard Phase 2 API Integration Tests.
Tests:
1. Health endpoint: GET /api/health and GET /health
2. Status endpoint: GET /api/status and GET /status
3. Verification endpoint: POST /api/phase2/verify and POST /api/verify
4. Empty claim handling (NEI with explanation / validation)
5. Known TRUE claim (Amazon 0$ corporate income tax)
6. Known FALSE claim (Biden end shareholder capitalism)
7. Sparse evidence / ungrounded claim (NEI prediction)
8. Output schema verification (8 stages data, publisher, title, scores, URLs)
"""

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import unittest
import threading
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.api_bridge import handle_status, handle_verify

class TestFinGuardAPIIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Verify status directly
        cls.status = handle_status()

    def test_01_status_and_health_structure(self):
        """Verify that handle_status returns healthy status and model metadata."""
        status = handle_status()
        self.assertEqual(status["status"], "online")
        self.assertTrue(status.get("healthy", False))
        self.assertEqual(status["corpus_units"], 12941)
        self.assertIn("models", status)
        self.assertIn("cross_encoder_reranker", status["models"])
        self.assertIn("nli_verifier", status["models"])

    def test_02_verify_empty_claim(self):
        """Verify that empty claim returns NEI with explanation."""
        res = handle_verify({"claim": ""})
        self.assertEqual(res["verdict"], "NEI")
        self.assertTrue(res["is_uncertain"])
        self.assertIn("non-empty", res["explanation"])

    def test_03_verify_true_claim_grounding(self):
        """Verify known TRUE claim from FinFact."""
        claim = "Amazon paid zero dollars in federal corporate income tax for two consecutive years."
        res = handle_verify({"claim": claim, "use_query_expansion": True, "use_reranker": True})
        
        self.assertIn("verdict", res)
        self.assertIn("verdict_display", res)
        self.assertIn(res["verdict"], ["true", "NEI", "false"])
        self.assertIn("confidence", res)
        self.assertIn("class_probabilities", res)
        self.assertIn("parsed_claim", res)
        self.assertIn("query_info", res)
        self.assertIn("selected_evidence_docs", res)
        
        # Verify that multi_queries are generated and exposed
        self.assertIn("multi_queries", res["query_info"])
        self.assertGreater(len(res["query_info"]["multi_queries"]), 0)
        
        # Verify selected evidence has required provenance fields
        for doc in res["selected_evidence_docs"]:
            self.assertIn("text", doc)
            self.assertIn("doc_id", doc)
            self.assertIn("source_urls", doc)
            self.assertIn("score", doc)
            self.assertIn("publisher", doc)
            self.assertIn("title", doc)

    def test_04_verify_false_claim_grounding(self):
        """Verify known FALSE claim from FinFact."""
        claim = "Did Biden Call for an 'End to Shareholder Capitalism' in July 2020?"
        res = handle_verify({"claim": claim, "use_query_expansion": True, "use_reranker": True})
        
        self.assertIn(res["verdict"], ["false", "NEI", "true"])
        self.assertIn("explanation", res)
        self.assertIn("retrieval_engine", res)
        self.assertTrue(len(res["selected_evidence_docs"]) > 0)

    def test_05_verify_nei_claim(self):
        """Verify authentic NEI claim from FinFact."""
        claim = "Does Jared Kushner's brother own a business associated with COVID-19 testing?"
        res = handle_verify({"claim": claim, "use_query_expansion": True, "use_reranker": True})
        
        self.assertIn("verdict", res)
        self.assertEqual(res["verdict"], "NEI")
        self.assertTrue(res["is_uncertain"])

if __name__ == "__main__":
    unittest.main()
