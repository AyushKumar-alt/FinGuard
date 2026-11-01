"""
Query Construction & Financial Query Expansion Module for FinGuard.
Builds targeted retrieval queries by combining:
1. Original claim query
2. Entity + Metric focused query
3. Exact Monetary Value + Period query
4. Finance-aware synonym expansion (configurable dictionary)
5. Company alias expansion (verified tickers/abbreviations)
6. Direction-aware query expansion (supporting vs contradicting branches)
7. Multi-query generation for Weighted Reciprocal Rank Fusion (Weighted RRF)

Strict Safeguards:
- Preserves the original query as primary anchor.
- Separates supporting-direction and contradicting-direction branches.
- Never combines increase and decrease into a single synonym set.
- Never negates positive assertions or removes negation qualifiers.
- Deterministic, deduplicated, and capped to prevent query drift.
"""

import os
import sys
import re
import yaml
from typing import Dict, List, Any, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    from src.claim_understanding import extract_financial_entities
except ImportError:
    from claim_understanding import extract_financial_entities

DEFAULT_SYNONYMS = {
    "net profit": ["net income", "profit after tax"],
    "revenue": ["sales", "operating revenue"],
    "decreased": ["declined", "fell", "dropped", "reduced"],
    "increased": ["rose", "grew", "climbed", "improved"],
    "total assets": ["assets", "total asset base"],
    "fy2024": ["fiscal year 2024", "financial year 2024"],
    "profit": ["net income", "earnings"],
    "debt": ["liabilities", "federal debt"],
    "national debt": ["gross federal debt", "public debt"],
    "income tax": ["corporate tax", "federal income tax"],
    "interest rates": ["benchmark rate", "federal funds rate"],
    "inflation": ["consumer price index", "cpi"],
    "shareholder capitalism": ["shareholder primacy", "Friedman doctrine"]
}

DEFAULT_COMPANY_ALIASES = {
    "Apple": ["Apple Inc.", "AAPL"],
    "Apple Inc.": ["Apple", "AAPL"],
    "Amazon": ["Amazon.com", "AMZN"],
    "Amazon.com": ["Amazon", "AMZN"],
    "Microsoft": ["Microsoft Corp", "MSFT"],
    "Microsoft Corp": ["Microsoft", "MSFT"],
    "Google": ["Alphabet", "GOOGL"],
    "Alphabet": ["Google", "GOOGL"],
    "Tesla": ["Tesla Inc.", "TSLA"],
    "Tesla Inc.": ["Tesla", "TSLA"],
    "Meta": ["Facebook", "META"],
    "Facebook": ["Meta", "META"]
}

DEFAULT_DIRECTIONAL_GROUPS = {
    "decrease": {
        "supporting": ["decreased", "declined", "fell", "dropped", "reduced"],
        "contradicting": ["increased", "rose", "grew", "climbed", "improved"]
    },
    "increase": {
        "supporting": ["increased", "rose", "grew", "climbed", "improved"],
        "contradicting": ["decreased", "declined", "fell", "dropped", "reduced"]
    }
}

DEFAULT_RRF_WEIGHTS = {
    "original_claim": 1.2,
    "entity_metric_period": 1.0,
    "synonym_expanded": 0.9,
    "numerical_query": 1.0,
    "supporting_direction": 1.0,
    "contradicting_direction": 1.0
}

def load_expansion_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Loads configuration from yaml file, falling back to built-in defaults."""
    candidates = []
    if config_path:
        candidates.append(config_path)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidates.append(os.path.join(repo_root, "phase2", "config", "financial_query_expansion.yaml"))
    candidates.append(os.path.join("phase2", "config", "financial_query_expansion.yaml"))

    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                if isinstance(cfg, dict):
                    return cfg
            except Exception:
                pass

    return {
        "enabled": True,
        "max_queries": 6,
        "rrf_k": 60,
        "per_query_top_k": 15,
        "final_candidate_top_k": 15,
        "synonyms": DEFAULT_SYNONYMS,
        "company_aliases": DEFAULT_COMPANY_ALIASES,
        "directional_groups": DEFAULT_DIRECTIONAL_GROUPS,
        "rrf_weights": DEFAULT_RRF_WEIGHTS
    }

class FinancialQueryBuilder:
    def __init__(self, config_path: Optional[str] = None, max_expansions: int = 4):
        self.config = load_expansion_config(config_path)
        self.max_expansions = max_expansions
        self.enabled = self.config.get("enabled", True)
        self.synonyms = self.config.get("synonyms", DEFAULT_SYNONYMS)
        self.company_aliases = self.config.get("company_aliases", DEFAULT_COMPANY_ALIASES)
        self.directional_groups = self.config.get("directional_groups", DEFAULT_DIRECTIONAL_GROUPS)
        self.rrf_weights = self.config.get("rrf_weights", DEFAULT_RRF_WEIGHTS)
        self.max_queries = self.config.get("max_queries", 6)

    def generate_multi_queries(self, claim: str, parsed_claim: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Generates structured multi-query objects for Weighted RRF:
        Q0: Original claim (Weight 1.2)
        Q1: Entity + Metric + Period (Weight 1.0)
        Q2: Synonym-expanded query (Weight 0.9)
        Q3: Numerical query with amounts/percentages (Weight 1.0)
        Q4: Supporting direction query (Weight 1.0)
        Q5: Contradicting direction query (Weight 1.0)
        """
        if parsed_claim is None:
            parsed_claim = extract_financial_entities(claim)

        original_query = claim.strip()
        candidates: List[Dict[str, Any]] = []

        # Q0: Original claim (always first)
        candidates.append({
            "query": original_query,
            "group": "original_claim",
            "weight": float(self.rrf_weights.get("original_claim", 1.2)),
            "purpose": "Original claim"
        })

        if not self.enabled:
            return candidates

        orgs = [o for o in parsed_claim.get("organizations", []) if o]
        metrics = [m for m in parsed_claim.get("financial_metrics", []) if m]
        amounts = [a for a in parsed_claim.get("monetary_amounts", []) if a]
        pcts = [p for p in parsed_claim.get("percentages", []) if p]
        periods = [pr for pr in parsed_claim.get("periods", []) if pr]
        directional_terms = parsed_claim.get("directional_terms", [])
        negations = [n for n in parsed_claim.get("negations", []) if n]
        numbers = amounts + pcts

        # Company alias check
        primary_org = orgs[0] if orgs else ""
        alias_org = None
        if primary_org and primary_org in self.company_aliases and self.company_aliases[primary_org]:
            alias_org = self.company_aliases[primary_org][0]

        # Q1: Entity + Metric + Period (High precision anchor)
        q1_parts = []
        if orgs:
            q1_parts.extend(orgs)
        if metrics:
            q1_parts.extend(metrics)
        if periods:
            q1_parts.extend(periods)
        if q1_parts:
            q1_text = " ".join(q1_parts).strip()
            candidates.append({
                "query": q1_text,
                "group": "entity_metric_period",
                "weight": float(self.rrf_weights.get("entity_metric_period", 1.0)),
                "purpose": "Entity + metric + period"
            })

        # Q2: Synonym-expanded Query
        synonym_metric = None
        for m in metrics:
            m_key = m.lower()
            if m_key in self.synonyms and self.synonyms[m_key]:
                synonym_metric = self.synonyms[m_key][0]
                break

        synonym_period = None
        for p in periods:
            p_key = p.lower()
            if p_key in self.synonyms and self.synonyms[p_key]:
                synonym_period = self.synonyms[p_key][0]
                break

        q2_parts = []
        if alias_org and not synonym_metric:
            q2_parts.append(alias_org)
        elif orgs:
            q2_parts.extend(orgs)

        if synonym_metric:
            q2_parts.append(synonym_metric)
        elif metrics:
            q2_parts.extend(metrics)

        if synonym_period:
            q2_parts.append(synonym_period)
        elif periods:
            q2_parts.extend(periods)

        if q2_parts and (synonym_metric or synonym_period or alias_org):
            q2_text = " ".join(q2_parts).strip()
            candidates.append({
                "query": q2_text,
                "group": "synonym_expanded",
                "weight": float(self.rrf_weights.get("synonym_expanded", 0.9)),
                "purpose": "Synonym-expanded query"
            })

        # Q3: Numerical Query
        if numbers:
            q3_parts = []
            if orgs:
                q3_parts.extend(orgs)
            q3_parts.extend(numbers)
            if metrics:
                q3_parts.extend(metrics)
            if periods:
                q3_parts.extend(periods)
            q3_text = " ".join(q3_parts).strip()
            candidates.append({
                "query": q3_text,
                "group": "numerical_query",
                "weight": float(self.rrf_weights.get("numerical_query", 1.0)),
                "purpose": "Numerical query"
            })

        # Q4 & Q5: Direction-aware Queries (Supporting vs Contradicting)
        identified_direction = None
        claim_lower = original_query.lower()

        for dt in directional_terms:
            d_type = dt.get("direction")
            if d_type in ("decrease", "increase"):
                identified_direction = d_type
                break

        if not identified_direction:
            for term in self.directional_groups.get("decrease", {}).get("supporting", []):
                if re.search(r'\b' + re.escape(term) + r'\b', claim_lower):
                    identified_direction = "decrease"
                    break
            if not identified_direction:
                for term in self.directional_groups.get("increase", {}).get("supporting", []):
                    if re.search(r'\b' + re.escape(term) + r'\b', claim_lower):
                        identified_direction = "increase"
                        break

        if identified_direction in self.directional_groups:
            dir_cfg = self.directional_groups[identified_direction]
            supp_terms = dir_cfg.get("supporting", [])
            contra_terms = dir_cfg.get("contradicting", [])

            # Supporting direction query
            if supp_terms:
                supp_word = supp_terms[1] if len(supp_terms) > 1 and supp_terms[0] in claim_lower else supp_terms[0]
                q4_parts = []
                if orgs:
                    q4_parts.extend(orgs)
                if negations:
                    q4_parts.extend(negations)
                q4_parts.append(supp_word)
                if metrics:
                    q4_parts.extend(metrics)
                if periods:
                    q4_parts.extend(periods)
                q4_text = " ".join(q4_parts).strip()
                candidates.append({
                    "query": q4_text,
                    "group": "supporting_direction",
                    "weight": float(self.rrf_weights.get("supporting_direction", 1.0)),
                    "purpose": "Supporting direction"
                })

            # Contradicting direction query
            if contra_terms:
                contra_word = contra_terms[0]
                q5_parts = []
                if orgs:
                    q5_parts.extend(orgs)
                if negations:
                    q5_parts.extend(negations)
                q5_parts.append(contra_word)
                if metrics:
                    q5_parts.extend(metrics)
                if periods:
                    q5_parts.extend(periods)
                q5_text = " ".join(q5_parts).strip()
                candidates.append({
                    "query": q5_text,
                    "group": "contradicting_direction",
                    "weight": float(self.rrf_weights.get("contradicting_direction", 1.0)),
                    "purpose": "Contradicting direction"
                })

        # Negation-specific query if negation exists and not yet captured
        if negations and (orgs or metrics):
            q_neg_parts = []
            if orgs:
                q_neg_parts.extend(orgs)
            q_neg_parts.extend(negations)
            if metrics:
                q_neg_parts.extend(metrics)
            q_neg_text = " ".join(q_neg_parts).strip()
            candidates.append({
                "query": q_neg_text,
                "group": "negation_preserved",
                "weight": 1.0,
                "purpose": "Negation preserved"
            })

        # Normalize and deduplicate queries while preserving order
        seen = set()
        final_queries = []
        for q_obj in candidates:
            norm = " ".join(q_obj["query"].strip().lower().split())
            if not norm:
                continue
            if norm not in seen:
                seen.add(norm)
                final_queries.append(q_obj)
                if len(final_queries) >= self.max_queries:
                    break

        return final_queries

    def build_queries(self, claim: str, parsed_claim: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generates deterministic, direction-preserving query variations.
        Maintains complete backward compatibility with existing tests and callers,
        while returning multi_queries for Weighted RRF.
        """
        if parsed_claim is None:
            parsed_claim = extract_financial_entities(claim)

        original_query = claim.strip()
        multi_queries = self.generate_multi_queries(claim, parsed_claim)

        # Legacy expanded_queries: exclude contradicting direction and original query
        # to strictly prevent direction-flipping in legacy string concatenation
        expanded_queries = []
        for q_obj in multi_queries:
            if q_obj["group"] not in ("original_claim", "contradicting_direction"):
                q_text = q_obj["query"].strip()
                if q_text and q_text.lower() != original_query.lower():
                    expanded_queries.append(q_text)

        # Cap expanded_queries to max_expansions
        expanded_queries = expanded_queries[:self.max_expansions]

        if expanded_queries:
            combined_query = f"{original_query} " + " ".join(expanded_queries)
        else:
            combined_query = original_query

        return {
            "original_query": original_query,
            "expanded_queries": expanded_queries,
            "combined_query": combined_query,
            "multi_queries": multi_queries,
            "query_strings": [q["query"] for q in multi_queries],
            "key_entities": parsed_claim.get("organizations", []),
            "key_metrics": parsed_claim.get("financial_metrics", []),
            "monetary_values": parsed_claim.get("monetary_amounts", []),
            "time_periods": parsed_claim.get("periods", [])
        }
