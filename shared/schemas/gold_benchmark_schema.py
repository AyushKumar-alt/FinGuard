"""
FinGuard Gold-Evidence Relevance Benchmark Schema & Validation Tools.

Defines the formal data contracts, identifier formatting, relevance grades,
support relations, and validation routines for the gold relevance benchmark.
"""

import hashlib
import json
import re
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List

BENCHMARK_VERSION = "1.0.0"

# Relevance Grades Definition
# 3: Directly decisive evidence
# 2: Substantially relevant evidence
# 1: Contextually relevant evidence
# 0: Judged not relevant
# None / "unjudged": Unknown relevance
VALID_RELEVANCE_GRADES = {0, 1, 2, 3, None}

GRADE_DESCRIPTIONS = {
    3: "Directly decisive evidence: Strongly supports or refutes the factual assertion, including the relevant entity, metric, time period, and numerical details.",
    2: "Substantially relevant evidence: Provides meaningful factual support or contradiction but is incomplete or requires contextual interpretation.",
    1: "Contextually relevant evidence: Helps understand the claim but does not independently establish its truth or falsity.",
    0: "Judged not relevant: Reviewed evidence does not materially help establish or assess the claim.",
    None: "Unjudged: Unknown relevance. No defensible judgment exists."
}

VALID_SUPPORT_RELATIONS = {"supports", "refutes", "mixed", "context_only", "unknown"}
VALID_JUDGMENT_STATUSES = {"unjudged", "silver_reference", "reviewed", "adjudicated"}
VALID_ANNOTATION_SOURCES = {"finfact_reference", "manual_annotation", "model_assisted", "unjudged_pool"}
VALID_ADJUDICATION_STATUSES = {"none", "pending", "approved", "disputed"}

def normalize_text_for_matching(text: str) -> str:
    """
    Deterministic normalization for text comparison:
    - Normalizes unicode quotes and dashes
    - Collapses multiple whitespace characters
    - Lowercases text
    - Strips leading/trailing punctuation and whitespace
    """
    if not isinstance(text, str):
        return ""
    t = text.strip()
    t = t.replace('\u2014', ' - ').replace('\u2013', ' - ')
    t = t.replace('\u201c', '"').replace('\u201d', '"')
    t = t.replace('\u2018', "'").replace('\u2019', "'")
    t = re.sub(r'\s+', ' ', t)
    return t.lower().strip()

def compute_content_sha256(text: str) -> str:
    """Computes SHA-256 hex digest of normalized text."""
    norm = normalize_text_for_matching(text)
    return hashlib.sha256(norm.encode('utf-8')).hexdigest()

def format_claim_id(record_index: int) -> str:
    """Generates stable canonical claim ID."""
    return f"claim_{int(record_index):04d}"

def parse_claim_id(claim_id: str) -> int:
    """Parses integer index from canonical claim ID."""
    m = re.match(r'^claim_(\d+)$', claim_id)
    if m:
        return int(m.group(1))
    raise ValueError(f"Invalid claim_id format: '{claim_id}'. Expected 'claim_XXXX'.")

def format_evidence_id(doc_index: int) -> str:
    """Generates stable canonical evidence unit ID."""
    return f"ev_{int(doc_index):05d}"

def parse_evidence_id(evidence_id: str) -> int:
    """Parses integer index from canonical evidence ID."""
    m = re.match(r'^ev_(\d+)$', evidence_id)
    if m:
        return int(m.group(1))
    raise ValueError(f"Invalid evidence_id format: '{evidence_id}'. Expected 'ev_XXXXX'.")

@dataclass
class EvidenceAnnotationRecord:
    claim_id: str
    evidence_id: str
    relevance_grade: Optional[int]  # 0, 1, 2, 3, or None for Unjudged
    judgment_status: str           # "unjudged", "silver_reference", "reviewed", "adjudicated"
    support_relation: str          # "supports", "refutes", "mixed", "context_only", "unknown"
    annotation_source: str         # "finfact_reference", "manual_annotation", "model_assisted", "unjudged_pool"
    annotation_method: str         # "exact_reference_match", "normalized_sentence_match", "manual_review", "unjudged"
    annotator_or_reviewer: str
    adjudication_status: str       # "none", "pending", "approved", "disputed"
    rationale: str
    source_url: Optional[str] = None
    dataset_version: str = BENCHMARK_VERSION
    claim_text: Optional[str] = None
    evidence_text: Optional[str] = None
    retrieved_by_systems: List[str] = field(default_factory=list)
    system_ranks: Dict[str, int] = field(default_factory=dict)

    def validate(self) -> None:
        """Validates record fields against schema invariants."""
        if not re.match(r'^claim_\d+$', self.claim_id):
            raise ValueError(f"Invalid claim_id: '{self.claim_id}'")
        if not re.match(r'^ev_\d+$', self.evidence_id):
            raise ValueError(f"Invalid evidence_id: '{self.evidence_id}'")
        
        if self.relevance_grade not in VALID_RELEVANCE_GRADES:
            raise ValueError(f"Invalid relevance_grade: {self.relevance_grade}. Must be 0, 1, 2, 3, or None.")
            
        if self.judgment_status not in VALID_JUDGMENT_STATUSES:
            raise ValueError(f"Invalid judgment_status: '{self.judgment_status}'. Must be one of {VALID_JUDGMENT_STATUSES}.")
            
        if self.support_relation not in VALID_SUPPORT_RELATIONS:
            raise ValueError(f"Invalid support_relation: '{self.support_relation}'. Must be one of {VALID_SUPPORT_RELATIONS}.")
            
        if self.annotation_source not in VALID_ANNOTATION_SOURCES:
            raise ValueError(f"Invalid annotation_source: '{self.annotation_source}'. Must be one of {VALID_ANNOTATION_SOURCES}.")
            
        if self.adjudication_status not in VALID_ADJUDICATION_STATUSES:
            raise ValueError(f"Invalid adjudication_status: '{self.adjudication_status}'. Must be one of {VALID_ADJUDICATION_STATUSES}.")

        # Invariant: Unjudged records MUST have relevance_grade=None and judgment_status="unjudged"
        if self.judgment_status == "unjudged" and self.relevance_grade is not None:
            raise ValueError(f"Unjudged record cannot have a numeric relevance_grade ({self.relevance_grade}). Must be None.")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceAnnotationRecord":
        grade = data.get("relevance_grade")
        if grade is not None:
            grade = int(grade)
        
        record = cls(
            claim_id=str(data["claim_id"]),
            evidence_id=str(data["evidence_id"]),
            relevance_grade=grade,
            judgment_status=str(data.get("judgment_status", "unjudged")),
            support_relation=str(data.get("support_relation", "unknown")),
            annotation_source=str(data.get("annotation_source", "unjudged_pool")),
            annotation_method=str(data.get("annotation_method", "unjudged")),
            annotator_or_reviewer=str(data.get("annotator_or_reviewer", "unassigned")),
            adjudication_status=str(data.get("adjudication_status", "none")),
            rationale=str(data.get("rationale", "")),
            source_url=data.get("source_url"),
            dataset_version=str(data.get("dataset_version", BENCHMARK_VERSION)),
            claim_text=data.get("claim_text"),
            evidence_text=data.get("evidence_text"),
            retrieved_by_systems=data.get("retrieved_by_systems", []),
            system_ranks=data.get("system_ranks", {})
        )
        record.validate()
        return record
