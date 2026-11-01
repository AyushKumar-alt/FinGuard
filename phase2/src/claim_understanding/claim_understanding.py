"""
Financial Claim Understanding & Entity/Metric Extraction Module for FinGuard.
Extracts structured financial semantics:
- Organizations and companies
- Financial metrics (profit, debt, revenue, taxes, etc.)
- Monetary amounts and currencies (preserving $11B, ₹5,000, $7 trillion, etc.)
- Percentages and ratios (10.5%, 30%, 50 bps)
- Fiscal years and quarters (FY2025, Q4 2025, 2020-2024)
- Special tokens (401(k), S&P 500)
- Directional assertions (increased, fell, exceeded, doubled)
- Negation and comparison operators (not, zero, did not, less than)

Does not fabricate missing fields; reports extraction uncertainty appropriately.
"""

import re
import sys
from typing import Dict, List, Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Financial Metrics Vocabulary
FINANCIAL_METRICS = {
    "revenue", "profit", "net profit", "gross profit", "net income", "operating income",
    "debt", "national debt", "gross debt", "federal debt", "deficit", "budget deficit",
    "eps", "earnings per share", "market capitalization", "market cap", "cash flow",
    "operating cash flow", "free cash flow", "taxes", "federal income tax", "corporate tax",
    "income tax", "capital gains", "interest rates", "interest rate", "inflation", "cpi",
    "dividend", "dividends", "shareholder capitalism", "pension", "pensions", "retirement",
    "valuation", "sales", "gross sales", "wages", "minimum wage", "gdp", "economic growth",
    "spending", "budget", "state budget", "operating margin", "profit margin"
}

# Directional Terminology
DIRECTIONAL_TERMS = {
    "increase": ["increase", "increased", "increasing", "increases", "rise", "rose", "rising", "surge", "surged", "grow", "grew", "growth", "jumped", "exceeded", "doubled", "tripled", "quadrupled", "climb", "climbed", "record high"],
    "decrease": ["decrease", "decreased", "decreasing", "decreases", "fall", "fell", "falling", "drop", "dropped", "dropping", "plunge", "plunged", "decline", "declined", "shrink", "shrank", "cut", "cuts", "below", "record low", "under"],
    "neutral_change": ["flat", "unchanged", "stagnant", "stabilized"]
}

# Negation and Comparison Patterns
NEGATION_PATTERNS = [
    r'\bnot\b', r'\bno\b', r'\bnever\b', r'\bzero\b', r'\bnone\b',
    r'\bdid\s+not\b', r'\bdoes\s+not\b', r'\bwas\s+not\b', r'\bwere\s+not\b',
    r'\bwithout\b', r'\bfailed\s+to\b', r'\brefused\s+to\b', r'\babolish\b',
    r'\bended\b', r'\bend\s+to\b', r'\bstop\b', r'\bneither\b', r'\bnor\b'
]

COMPARISON_PATTERNS = [
    r'\bmore\s+than\b', r'\bless\s+than\b', r'\bhigher\s+than\b', r'\blower\s+than\b',
    r'\bgreater\s+than\b', r'\bexceeded\b', r'\bsurpassed\b', r'\bat\s+least\b',
    r'\bat\s+most\b', r'\bover\b', r'\bunder\b', r'\bequal\s+to\b', r'\bcompared\s+to\b'
]

# Known Financial Organizations / Entities
KNOWN_ORGANIZATIONS = [
    "Amazon", "Apple", "Tesla", "Microsoft", "Google", "Alphabet", "Meta", "Facebook",
    "Twitter", "Federal Reserve", "Fed", "Treasury", "US Treasury", "Treasury Department",
    "IRS", "Internal Revenue Service", "Securities and Exchange Commission", "SEC",
    "Business Roundtable", "Oscar", "Oscar Health", "Thrive Capital", "Raw Story",
    "Snopes", "Politifact", "Brookings Institute", "White House", "Congress", "Senate",
    "House of Representatives", "World Bank", "IMF", "CDC", "WHO"
]

def finance_aware_tokenize(text: str) -> List[str]:
    """
    Tokenizes financial text while strictly preserving:
    - Currency expressions ($11B, ₹5,000, €100M, £2.5B)
    - Percentages (10.5%, 30%)
    - Fiscal periods (FY2025, Q4 2025, FY25)
    - Special tokens (401(k), S&P 500)
    """
    if not text:
        return []
        
    pattern = re.compile(
        r'(?:[\$₹€£]\s*\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:trillion|billion|million|thousand|crore|lakh|[TBMBK]))?)'  # Currency amounts
        r'|(?:\d+(?:\.\d+)?%)'                                  # Percentages
        r'|(?:\b(?:FY|Q[1-4])\s*\d{2,4}\b)'                    # Fiscal periods (FY2025, Q4 2025)
        r'|(?:401\(k\)|\b401k\b)'                               # 401(k) retirement
        r'|(?:\bS&P\s*500\b)'                                    # S&P 500
        r'|(?:\b\w+(?:-\w+)+\b)'                                # Hyphenated terms (fact-check, shareholder-capitalism)
        r'|(?:\b\w+\b)',                                        # Regular words
        re.IGNORECASE
    )
    tokens = pattern.findall(text)
    return [t.strip() for t in tokens if t.strip()]

def extract_financial_entities(claim: str) -> Dict[str, Any]:
    """
    Analyzes a financial claim and returns structured semantic metadata.
    """
    if not isinstance(claim, str) or not claim.strip():
        return {
            "original_claim": "",
            "cleaned_claim": "",
            "tokens": [],
            "organizations": [],
            "financial_metrics": [],
            "monetary_amounts": [],
            "percentages": [],
            "periods": [],
            "directional_terms": [],
            "negations": [],
            "comparisons": [],
            "is_uncertain": True,
            "extraction_notes": ["Empty or invalid claim text provided."]
        }
        
    raw_claim = claim.strip()
    tokens = finance_aware_tokenize(raw_claim)
    claim_lower = raw_claim.lower()
    
    # 1. Extract Organizations
    orgs_found = []
    for org in KNOWN_ORGANIZATIONS:
        if re.search(r'\b' + re.escape(org) + r'\b', raw_claim, re.IGNORECASE):
            orgs_found.append(org)
            
    # Corporate suffix pattern (e.g. ABC Bank, Thrive Capital, Business Roundtable)
    corp_suffix_pat = re.compile(r'\b[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*\s+(?:Bank|Corp|Corporation|Inc|LLC|Group|Holdings|Fund|Capital|Trust|Roundtable)\b')
    for m in corp_suffix_pat.findall(raw_claim):
        if m not in orgs_found:
            orgs_found.append(m)
            
    # Also detect capitalized multi-word proper nouns that look like entities (exclude sentence start false positives)
    words = raw_claim.split()
    if len(words) > 1:
        # Check from word 1 onward to avoid first word false positives like "Did Biden"
        tail_text = " ".join(words[1:])
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', tail_text)
        for pn in proper_nouns:
            if pn not in orgs_found and pn.lower() not in ["united states", "federal reserve", "shareholder capitalism"]:
                orgs_found.append(pn)
            
    # 2. Extract Financial Metrics
    metrics_found = []
    for m in sorted(FINANCIAL_METRICS, key=len, reverse=True):
        if re.search(r'\b' + re.escape(m) + r'\b', claim_lower):
            if not any(m in existing for existing in metrics_found):
                metrics_found.append(m)

    # 3. Extract Monetary Amounts
    monetary_pattern = re.compile(
        r'(?:[\$₹€£]\s*\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:trillion|billion|million|thousand|crore|lakh|[TBMBK]))?'
        r'|\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:trillion|billion|million|dollars?|rupees?|euros?|pounds?|crore|lakh)\b)',
        re.IGNORECASE
    )
    monetary_found = [m.strip() for m in monetary_pattern.findall(raw_claim) if m.strip()]
    
    # 4. Extract Percentages
    pct_pattern = re.compile(r'\b\d+(?:\.\d+)?\s*(?:%|percent\b|percentage\b|basis\s+points\b|bps\b)', re.IGNORECASE)
    percentages_found = [p.strip() for p in pct_pattern.findall(raw_claim) if p.strip()]
    
    # 5. Extract Fiscal Years and Periods
    period_pattern = re.compile(
        r'(?:\bFY\s*\d{2,4}\b|\bQ[1-4]\s*(?:20\d{2}|19\d{2})?\b|\b(?:19|20)\d{2}(?:\s*-\s*(?:19|20)?\d{2})?\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+(?:19|20)\d{2}\b)',
        re.IGNORECASE
    )
    periods_found = [p.strip() for p in period_pattern.findall(raw_claim) if p.strip()]
    
    # 6. Extract Directional Assertions
    direction_found = []
    for dir_type, terms in DIRECTIONAL_TERMS.items():
        for t in terms:
            if re.search(r'\b' + re.escape(t) + r'\b', claim_lower):
                direction_found.append({"term": t, "direction": dir_type})
                
    # 7. Extract Negations
    negations_found = []
    for pat in NEGATION_PATTERNS:
        matches = re.findall(pat, claim_lower)
        if matches:
            negations_found.extend(matches)
            
    # 8. Extract Comparison Operators
    comparisons_found = []
    for pat in COMPARISON_PATTERNS:
        matches = re.findall(pat, claim_lower)
        if matches:
            comparisons_found.extend(matches)

    # Uncertainty assessment
    notes = []
    is_uncertain = False
    if not metrics_found and not monetary_found and not percentages_found:
        notes.append("No explicit numerical metric or monetary figure identified in claim.")
        is_uncertain = True
    if not orgs_found:
        notes.append("No named corporate entity or government body explicitly identified.")

    return {
        "original_claim": raw_claim,
        "cleaned_claim": " ".join(tokens),
        "tokens": tokens,
        "organizations": list(dict.fromkeys(orgs_found)),
        "financial_metrics": list(dict.fromkeys(metrics_found)),
        "monetary_amounts": list(dict.fromkeys(monetary_found)),
        "percentages": list(dict.fromkeys(percentages_found)),
        "periods": list(dict.fromkeys(periods_found)),
        "directional_terms": direction_found,
        "negations": list(dict.fromkeys(negations_found)),
        "comparisons": list(dict.fromkeys(comparisons_found)),
        "is_uncertain": is_uncertain,
        "extraction_notes": notes
    }

if __name__ == "__main__":
    test_claims = [
        "Did Biden Call for an 'End to Shareholder Capitalism' in July 2020?",
        "Amazon paid $0 in federal corporate income tax despite record profit in FY2018.",
        "The US national debt exceeded $7 trillion under Donald Trump, rising by 30%.",
        "ABC Bank's net profit fell by 10.5% to ₹5,000 crore in Q4 2025."
    ]
    for c in test_claims:
        res = extract_financial_entities(c)
        print("\nClaim:", c)
        print("  Orgs:", res["organizations"])
        print("  Metrics:", res["financial_metrics"])
        print("  Amounts:", res["monetary_amounts"])
        print("  Percentages:", res["percentages"])
        print("  Periods:", res["periods"])
        print("  Directions:", [d["term"] for d in res["directional_terms"]])
        print("  Negations:", res["negations"])
