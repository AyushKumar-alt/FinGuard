"""
Pipeline 1: POS-Guided Financial Lemmatizer.
Uses NLTK averaged_perceptron_tagger_eng + WordNetLemmatizer.
Protected financial tokens (currency amounts, percentages, fiscal periods, acronyms)
are passed through WITHOUT lemmatization to preserve financial meaning.

POS tag mapping (Penn Treebank -> WordNet):
  NN/NNS/NNP/NNPS -> 'n' (noun)
  VB/VBD/VBG/VBN/VBP/VBZ -> 'v' (verb)
  JJ/JJR/JJS -> 'a' (adjective)
  RB/RBR/RBS -> 'r' (adverb)
  All others -> 'n' (conservative fallback: noun lemmatization)

Known limitation: WordNet cannot lemmatize all irregular verb forms.
Example: 'fell' -> 'fell' (not 'fall') because WordNet has no entry for 'fell' as VBD.
This limitation is documented and unavoidable without a larger lexicon.
"""

import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Protected token patterns — these bypass lemmatization entirely
_PROTECTED_PATTERNS = [
    re.compile(r'^[\$₹€£]\s*[\d,]+(?:\.\d+)?(?:\s*(?:trillion|billion|million|thousand|crore|lakh|[TBMK]))?$', re.IGNORECASE),  # currency amounts
    re.compile(r'^\d+(?:\.\d+)?%$'),                         # percentages
    re.compile(r'^(?:FY|Q[1-4])\s*\d{2,4}(?:[-/]\d{2,4})?$', re.IGNORECASE), # fiscal periods like FY2025, FY2025-26
    re.compile(r'^[A-Z]{2,6}$'),                              # acronyms (GDP, RBI, EPS, NPA)
    re.compile(r'^401\(?k\)?$', re.IGNORECASE),               # 401(k)
    re.compile(r'^\d+(?:[,\.]\d+)*$'),                        # pure numbers
    re.compile(r'^[\$₹€£]$'),                                  # standalone currency symbols
]

def _is_protected(token: str) -> bool:
    """Returns True if token should bypass lemmatization."""
    for pat in _PROTECTED_PATTERNS:
        if pat.match(token):
            return True
    return False

_PENN_TO_WN = {
    'NN': 'n', 'NNS': 'n', 'NNP': 'n', 'NNPS': 'n',
    'VB': 'v', 'VBD': 'v', 'VBG': 'v', 'VBN': 'v', 'VBP': 'v', 'VBZ': 'v',
    'JJ': 'a', 'JJR': 'a', 'JJS': 'a',
    'RB': 'r', 'RBR': 'r', 'RBS': 'r',
}

class POSGuidedLemmatizer:
    """
    Applies POS-guided lemmatization using NLTK WordNet.
    Protected tokens (currency, percentages, fiscal periods, acronyms) are never modified.
    """
    def __init__(self):
        self._lemmatizer = WordNetLemmatizer()
        self._available = True
        self._library = "nltk.stem.WordNetLemmatizer + nltk.tag.averaged_perceptron_tagger_eng"
        # Verify resources on init
        try:
            _ = wordnet.synsets("test")
            _ = nltk.pos_tag(["test"])
        except Exception as e:
            self._available = False
            self._library = f"unavailable: {e}"

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def library_info(self) -> str:
        return self._library

    def lemmatize_tokens(self, tokens: list) -> list:
        """
        Lemmatize a list of tokens using POS-guided WordNet lemmatization.
        Protected tokens pass through unchanged.
        Returns list of lemmatized tokens in same order.
        """
        if not tokens or not self._available:
            return tokens

        # Separate protected tokens (pass through) from regular tokens (lemmatize)
        protected_mask = [_is_protected(t) for t in tokens]
        
        # Only POS-tag non-protected tokens (reduces noise from financial tokens)
        regular_tokens = [(i, t) for i, t in enumerate(tokens) if not protected_mask[i]]
        
        result = list(tokens)  # copy
        
        if regular_tokens:
            indices, words = zip(*regular_tokens)
            try:
                tagged = nltk.pos_tag(list(words))
            except Exception:
                # Fallback: use noun lemmatization for all
                tagged = [(w, 'NN') for w in words]

            for (orig_idx, _), (word, tag) in zip(regular_tokens, tagged):
                wn_pos = _PENN_TO_WN.get(tag, 'n')  # default to noun
                try:
                    lemma = self._lemmatizer.lemmatize(word.lower(), wn_pos)
                    result[orig_idx] = lemma
                except Exception:
                    result[orig_idx] = word.lower()  # conservative fallback

        return result

    def lemmatize_text(self, text: str, tokenizer_fn) -> list:
        """Convenience: tokenize then lemmatize. Returns list of lemmas."""
        tokens = tokenizer_fn(text)
        return self.lemmatize_tokens(tokens)

