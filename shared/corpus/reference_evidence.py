"""
Reference-Evidence Extraction & Canonical Corpus Manifest Builder for FinGuard.

Extracts reference evidence from finfact.json with provenance, performs
deterministic matching against candidate evidence units, and generates
reproducible, versioned manifests for claims and corpus documents.
"""

import os
import sys
import json
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gold_benchmark_schema import (
    format_claim_id,
    format_evidence_id,
    normalize_text_for_matching,
    compute_content_sha256,
    BENCHMARK_VERSION
)

class ReferenceEvidenceExtractor:
    def __init__(self, json_path: str = "finfact.json"):
        self.json_path = json_path
        self.claims: Dict[str, Dict[str, Any]] = {}
        self.corpus: Dict[str, Dict[str, Any]] = {}
        self.reference_mappings: Dict[str, List[Dict[str, Any]]] = {}
        self._load_and_extract()

    def _load_and_extract(self):
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"Dataset not found at {self.json_path}")

        with open(self.json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        text_to_ev_id: Dict[str, str] = {}
        claim_ref_ids: Dict[str, List[str]] = defaultdict(list)
        claim_ref_details: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        doc_counter = 0

        # Step 1: Scan all records to build canonical candidate evidence corpus
        for idx, item in enumerate(raw_data):
            cid = format_claim_id(idx)
            claim_text = item.get("claim", "").strip()
            ev_list = item.get("evidence", []) or []

            self.claims[cid] = {
                "claim_id": cid,
                "record_index": idx,
                "claim_text": claim_text,
                "label": item.get("label", ""),
                "url": item.get("url", ""),
                "author": item.get("author", "Unknown"),
                "posted": item.get("posted", ""),
                "issues": item.get("issues", []) or [],
                "raw_evidence_count": len(ev_list),
                "has_reference_evidence": False,
                "reference_evidence_ids": []
            }

            for ev in ev_list:
                if not isinstance(ev, dict):
                    continue
                sent = ev.get("sentence", "").strip()
                if not sent or len(sent.split()) < 4:
                    continue

                hrefs = ev.get("hrefs", []) or []
                norm_text = normalize_text_for_matching(sent)

                if norm_text not in text_to_ev_id:
                    eid = format_evidence_id(doc_counter)
                    doc_counter += 1
                    text_to_ev_id[norm_text] = eid
                    self.corpus[eid] = {
                        "evidence_id": eid,
                        "doc_index": parse_evidence_doc_id(eid),
                        "text": sent,
                        "normalized_text": norm_text,
                        "sha256": compute_content_sha256(sent),
                        "source_urls": list(set(hrefs)),
                        "parent_claim_ids": [cid],
                        "doc_type": "evidence_sentence"
                    }
                else:
                    eid = text_to_ev_id[norm_text]
                    if cid not in self.corpus[eid]["parent_claim_ids"]:
                        self.corpus[eid]["parent_claim_ids"].append(cid)
                    for h in hrefs:
                        if h not in self.corpus[eid]["source_urls"]:
                            self.corpus[eid]["source_urls"].append(h)

                if eid not in claim_ref_ids[cid]:
                    claim_ref_ids[cid].append(eid)
                    claim_ref_details[cid].append({
                        "evidence_id": eid,
                        "text": sent,
                        "source_urls": list(set(hrefs)),
                        "match_type": "exact_or_normalized_sentence",
                        "annotation_source": "finfact_reference",
                        "initial_support_relation": self._infer_initial_relation(item.get("label", "")),
                        "silver_relevance_grade": 3  # Reference evidence explicitly associated with claim
                    })

        # Update claim references
        for cid, eids in claim_ref_ids.items():
            self.claims[cid]["reference_evidence_ids"] = eids
            self.claims[cid]["has_reference_evidence"] = len(eids) > 0
            self.reference_mappings[cid] = claim_ref_details[cid]

        print(f"Loaded {len(self.claims):,} claims from {self.json_path}.", flush=True)
        print(f"Extracted {len(self.corpus):,} canonical candidate evidence sentences.", flush=True)
        claims_with_ref = sum(1 for c in self.claims.values() if c["has_reference_evidence"])
        print(f"Claims with reference evidence: {claims_with_ref:,} / {len(self.claims):,}", flush=True)

    def _infer_initial_relation(self, claim_label: str) -> str:
        """Heuristic initial relation for reference evidence based on fact-checker verdict."""
        l = claim_label.lower().strip()
        if l == "true":
            return "supports"
        elif l == "false":
            return "refutes"
        elif l == "nei":
            return "context_only"
        return "unknown"

    def match_evidence_sentence(self, candidate_text: str, target_claim_id: str) -> Tuple[bool, str, Optional[str]]:
        """
        Deterministic matcher checking if a candidate text matches known reference evidence.
        Returns: (is_match, match_type, matched_evidence_id)
        """
        norm_cand = normalize_text_for_matching(candidate_text)
        known_refs = self.reference_mappings.get(target_claim_id, [])

        for ref in known_refs:
            norm_ref = normalize_text_for_matching(ref["text"])
            # 1. Exact match
            if candidate_text.strip() == ref["text"].strip():
                return True, "exact", ref["evidence_id"]
            # 2. Normalized match
            if norm_cand == norm_ref:
                return True, "normalized", ref["evidence_id"]
            # 3. Sentence within passage
            if len(norm_cand) > 30 and (norm_cand in norm_ref or norm_ref in norm_cand):
                return True, "passage_substring", ref["evidence_id"]

        return False, "none", None

    def export_canonical_manifests(self, output_dir: str = "data/benchmark/v1"):
        os.makedirs(output_dir, exist_ok=True)

        claims_path = os.path.join(output_dir, "claims_manifest.json")
        with open(claims_path, "w", encoding="utf-8") as f:
            json.dump(self.claims, f, indent=2)

        corpus_path = os.path.join(output_dir, "corpus_manifest.json")
        with open(corpus_path, "w", encoding="utf-8") as f:
            json.dump(self.corpus, f, indent=2)

        ref_path = os.path.join(output_dir, "reference_evidence.json")
        with open(ref_path, "w", encoding="utf-8") as f:
            json.dump(self.reference_mappings, f, indent=2)

        print(f"Exported canonical manifests to {output_dir}:", flush=True)
        print(f"  - {claims_path} ({len(self.claims):,} claims)")
        print(f"  - {corpus_path} ({len(self.corpus):,} evidence units)")
        print(f"  - {ref_path} ({len(self.reference_mappings):,} claim reference sets)")

def parse_evidence_doc_id(eid: str) -> int:
    return int(eid.replace("ev_", ""))

if __name__ == "__main__":
    extractor = ReferenceEvidenceExtractor("finfact.json")
    extractor.export_canonical_manifests("data/benchmark/v1")
