"""
Pipeline 1: Domain-Aware Stop-Word Filter for Financial Text.
Scoped exclusively to Pipeline 1 — does not affect any existing module.
"""
import nltk

# Base general stop words from NLTK (already downloaded)
try:
    from nltk.corpus import stopwords as _sw
    _NLTK_STOPS = set(_sw.words("english"))
except Exception:
    _NLTK_STOPS = set()

# Financial terms that MUST NOT be treated as stop words
# (override any general stop-word list)
FINANCIAL_PROTECTED_TERMS = {
    # Negation and directionality (reversing meaning if removed)
    "not", "no", "never", "without", "nor", "neither", "failed",
    "increased", "decreased", "declined", "rose", "fell", "grew",
    "improved", "worsened", "gained", "lost", "surged", "dropped",
    # Financial acronyms
    "gdp", "rbi", "eps", "npa", "cpi", "sec", "irs", "imf", "fed",
    "bse", "nse", "ipo", "etf", "roi", "ebitda", "nps",
    # Financial quantifiers
    "zero", "nil", "negative", "positive",
    # Units (critical for financial amounts)
    "billion", "million", "trillion", "crore", "lakh", "thousand",
    "percent", "per", "cent", "bps",
    # Fiscal markers
    "fy", "q1", "q2", "q3", "q4",
    # Financial entities often used as function words but meaningful
    "over", "under", "above", "below",  # comparative amounts
}

# Additional low-value financial-domain words (add only clearly valueless terms)
FINANCIAL_DOMAIN_STOPS = {
    "said", "says", "according", "reported", "report", "stated", "claim",
    "claims", "noted", "added", "told", "announced", "confirmed",
    "however", "although", "despite", "yet", "though", "also",
    "would", "could", "should", "may", "might", "will",  # modal verbs
}

# Final stop word set: NLTK stops + financial domain stops, minus protected terms
STOP_WORDS = (_NLTK_STOPS | FINANCIAL_DOMAIN_STOPS) - FINANCIAL_PROTECTED_TERMS

# Configuration metadata (stored in index)
STOPWORD_CONFIG = {
    "base_list": "nltk_english",
    "domain_additions": sorted(FINANCIAL_DOMAIN_STOPS),
    "protected_terms_excluded": sorted(FINANCIAL_PROTECTED_TERMS),
    "total_stop_words": len(STOP_WORDS)
}

def is_stop_word(token: str) -> bool:
    return token.lower() in STOP_WORDS

def filter_tokens(tokens: list) -> list:
    """Remove stop words from token list, preserving order."""
    return [t for t in tokens if not is_stop_word(t)]
