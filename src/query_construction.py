"""
Query Construction & Financial Query Expansion Module for FinGuard.
Builds targeted retrieval queries by combining:
1. Original claim query
2. Entity + Metric focused query
3. Exact Monetary Value + Period query
4. Direction-preserving Financial Synonym expansion

Strict Safeguards:
- Preserves the original query as primary anchor.
- Strictly prevents expansions that reverse direction (e.g., never expands 'increase' to 'decrease').
- Never negates positive assertions or removes negation qualifiers.
- Deterministic and capped to prevent query drift.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, List, Any

try:
    from src.claim_understanding import extract_financial_entities
except ImportError:
    from claim_understanding import extract_financial_entities

# Controlled Financial Metric Synonyms (Direction-Neutral)
METRIC_SYNONYMS = {
    "profit": ["net income", "earnings"],
    "net profit": ["net income", "after-tax earnings"],
    "revenue": ["sales", "turnover"],
    "debt": ["federal debt", "liabilities"],
    "national debt": ["gross federal debt", "public debt"],
    "income tax": ["corporate tax", "federal income tax"],
    "interest rates": ["benchmark rate", "federal funds rate"],
    "inflation": ["consumer price index", "cpi"],
    "shareholder capitalism": ["shareholder primacy", "Friedman doctrine"]
}

class FinancialQueryBuilder:
    def __init__(self, max_expansions: int = 4):
        self.max_expansions = max_expansions

    def build_queries(self, claim: str, parsed_claim: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generates deterministic, direction-preserving query variations.
        """
        if parsed_claim is None:
            parsed_claim = extract_financial_entities(claim)
            
        original_query = claim.strip()
        orgs = parsed_claim.get("organizations", [])
        metrics = parsed_claim.get("financial_metrics", [])
        amounts = parsed_claim.get("monetary_amounts", [])
        pcts = parsed_claim.get("percentages", [])
        periods = parsed_claim.get("periods", [])
        directions = [d["term"] for d in parsed_claim.get("directional_terms", [])]
        negations = parsed_claim.get("negations", [])
        
        expanded_queries = []
        
        # 1. Entity + Metric Query (High precision anchor)
        if orgs and metrics:
            entity_metric = f"{' '.join(orgs)} {' '.join(metrics)}"
            if periods:
                entity_metric += f" {' '.join(periods)}"
            expanded_queries.append(entity_metric.strip())
            
        # 2. Metric + Value + Direction Query
        value_metric_parts = []
        if orgs:
            value_metric_parts.extend(orgs)
        if metrics:
            value_metric_parts.extend(metrics)
        if amounts:
            value_metric_parts.extend(amounts)
        if pcts:
            value_metric_parts.extend(pcts)
        if directions:
            value_metric_parts.append(directions[0]) # retain primary direction
        if value_metric_parts:
            expanded_queries.append(" ".join(value_metric_parts).strip())
            
        # 3. Synonym Expansion (Safeguarded against direction flip)
        synonym_parts = []
        if orgs:
            synonym_parts.extend(orgs)
        for m in metrics:
            syns = METRIC_SYNONYMS.get(m.lower(), [])
            if syns:
                synonym_parts.append(syns[0]) # pick primary approved synonym
        if synonym_parts and synonym_parts != orgs:
            if amounts:
                synonym_parts.extend(amounts)
            if periods:
                synonym_parts.extend(periods)
            expanded_queries.append(" ".join(synonym_parts).strip())
            
        # 4. Negation-Preserving Query if claim has negation
        if negations and (orgs or metrics):
            neg_query = f"{' '.join(orgs)} {' '.join(negations)} {' '.join(metrics)}"
            expanded_queries.append(neg_query.strip())
            
        # Deduplicate while preserving order, excluding original query
        seen = {original_query.lower()}
        unique_expansions = []
        for q in expanded_queries:
            q_clean = q.strip()
            if q_clean and q_clean.lower() not in seen:
                seen.add(q_clean.lower())
                unique_expansions.append(q_clean)
                
        # Limit to max_expansions
        final_expansions = unique_expansions[:self.max_expansions]
        
        # Combined query formulation for BM25/Dense matching
        if final_expansions:
            combined_query = f"{original_query} " + " ".join(final_expansions)
        else:
            combined_query = original_query

        return {
            "original_query": original_query,
            "expanded_queries": final_expansions,
            "combined_query": combined_query,
            "key_entities": orgs,
            "key_metrics": metrics,
            "monetary_values": amounts,
            "time_periods": periods
        }

if __name__ == "__main__":
    builder = FinancialQueryBuilder()
    examples = [
        "ABC Bank's net profit increased by 30% to ₹4,500 crore in FY2025.",
        "Did Biden Call for an 'End to Shareholder Capitalism' in July 2020?",
        "Amazon paid $0 in federal corporate income tax despite billions in profit."
    ]
    for ex in examples:
        res = builder.build_queries(ex)
        print("\nOriginal:", res["original_query"])
        print("Expansions:", res["expanded_queries"])
        print("Combined Query:", res["combined_query"])
