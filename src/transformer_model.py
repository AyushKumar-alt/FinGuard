"""
Claim Verifier and Evidence Ranking Interface.
Provides backwards compatibility for ClaimVerifier while routing to
FinancialClaimVerifier in src/verifier.py.
"""

from src.verifier import FinancialClaimVerifier

class ClaimVerifier:
    """
    Wrapper providing backwards compatibility for the ClaimVerifier API.
    Routes to the verified FinancialClaimVerifier engine.
    """
    def __init__(self, baseline_model_path="outputs/best_baseline_model.pkl", use_transformer=False):
        self.verifier = FinancialClaimVerifier(
            baseline_model_path=baseline_model_path,
            use_transformer=use_transformer
        )
        self.baseline_model = self.verifier.baseline_model

    def rank_evidence_sentences(self, claim: str, evidence_sentences: list) -> list:
        """
        Ranks candidate evidence sentences by lexical keyword relevance to the claim.
        Returns a sorted list of tuples (sentence, score).
        """
        if not evidence_sentences:
            return []
            
        claim_tokens = set(claim.lower().split())
        scored = []
        for sent in evidence_sentences:
            if not sent or not sent.strip():
                continue
            sent_tokens = set(sent.lower().split())
            overlap = len(claim_tokens.intersection(sent_tokens))
            score = overlap / (len(claim_tokens) + 1e-5)
            scored.append((sent, round(score, 3)))
            
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def predict(self, claim: str, evidence_text: str = "", engine_choice: str = None) -> dict:
        """Runs claim verification with uncertainty analysis."""
        return self.verifier.verify_claim(claim, evidence_text=evidence_text, engine_choice=engine_choice)

if __name__ == "__main__":
    cv = ClaimVerifier()
    res = cv.predict("Did Biden Call for an 'End to Shareholder Capitalism'?", "Biden did not propose dissolving the stock market.")
    print("ClaimVerifier API result:", res)
