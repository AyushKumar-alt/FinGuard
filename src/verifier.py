"""
Evidence-Grounded Financial Claim Verifier & Multi-Sentence NLI Aggregation.
Combines:
1. Calibrated Baseline Classifier (TF-IDF + Balanced Logistic Regression)
2. Pretrained Cross-Encoder NLI ('cross-encoder/nli-distilroberta-base')
   - Dynamically reads model config id2label
   - Maps entailment -> true, contradiction -> false, neutral -> NEI
3. Multi-Evidence Sentence Aggregation Strategy:
   - Evaluates each candidate sentence independently
   - Explicitly separates supporting vs contradicting evidence
   - Detects conflicting evidence across sources
   - Gates missing or insufficient evidence to NEI
"""

import os
import sys
import json
import joblib
import numpy as np
import torch
from typing import List, Dict, Any, Tuple

DEFAULT_NLI_MODEL = "cross-encoder/nli-distilroberta-base"
FINETUNED_NLI_MODEL = "models/finetuned_nli"
LABEL_MAPPING = {0: "false", 1: "true", 2: "NEI"}

class FinancialClaimVerifier:
    def __init__(self, baseline_model_path: str = "outputs/best_baseline_model.pkl",
                 nli_model_name: str = DEFAULT_NLI_MODEL,
                 use_transformer: bool = True,
                 device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.baseline_model = None
        self.nli_tokenizer = None
        self.nli_model = None
        
        # Support alias names
        if nli_model_name == "financial_finetuned":
            self.nli_model_name = FINETUNED_NLI_MODEL
        elif nli_model_name == "pretrained_zero_shot":
            self.nli_model_name = DEFAULT_NLI_MODEL
        else:
            self.nli_model_name = nli_model_name
            
        self.id2label_nli = {}
        self.has_transformer = False
        
        # 1. Load baseline model
        if os.path.exists(baseline_model_path):
            try:
                self.baseline_model = joblib.load(baseline_model_path)
                print(f"Loaded calibrated baseline classifier from {baseline_model_path}", flush=True)
            except Exception as e:
                print(f"Warning: Could not load baseline model: {e}", flush=True)
                
        # 2. Optionally load pretrained/fine-tuned NLI transformer
        if use_transformer:
            self._init_nli_model()

    def _init_nli_model(self):
        """Initializes NLI cross-encoder model and reads actual id2label mapping."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            print(f"Loading NLI cross-encoder: {self.nli_model_name} on {self.device}...", flush=True)
            local_only = os.path.exists(self.nli_model_name)
            self.nli_tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name, local_files_only=local_only)
            self.nli_model = AutoModelForSequenceClassification.from_pretrained(self.nli_model_name, local_files_only=local_only)
            self.nli_model.to(self.device)
            self.nli_model.eval()
            
            # Read actual model config id2label
            raw_id2label = getattr(self.nli_model.config, "id2label", {0: 'contradiction', 1: 'entailment', 2: 'neutral'})
            self.id2label_nli = {int(k): v.lower() for k, v in raw_id2label.items()}
            self.has_transformer = True
            print(f"NLI Model initialized successfully! Dynamic id2label: {self.id2label_nli}", flush=True)
        except Exception as e:
            print(f"Notice: Pretrained NLI model could not be loaded ({e}). Verifier will use calibrated baseline.", flush=True)
            self.has_transformer = False

    def predict_nli_single_pair(self, claim: str, evidence_sentence: str) -> Dict[str, float]:
        """
        Runs cross-encoder NLI inference on a single (premise=evidence, hypothesis=claim) pair.
        Returns normalized probabilities: {'entailment': p, 'contradiction': p, 'neutral': p}
        """
        if not self.has_transformer or self.nli_model is None:
            return {"entailment": 0.33, "contradiction": 0.33, "neutral": 0.34}
            
        # NLI format: premise = evidence, hypothesis = claim
        encoded = self.nli_tokenizer(
            evidence_sentence, claim,
            padding=True, truncation=True, max_length=160, return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            logits = self.nli_model(**encoded).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            
        nli_probs = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
        for class_idx, prob in enumerate(probs):
            class_name = self.id2label_nli.get(class_idx, "neutral").lower()
            if "entail" in class_name or "true" in class_name:
                nli_probs["entailment"] += float(prob)
            elif "contra" in class_name or "false" in class_name:
                nli_probs["contradiction"] += float(prob)
            else:
                nli_probs["neutral"] += float(prob)
                
        # Normalize
        total = sum(nli_probs.values()) + 1e-9
        return {k: round(v / total, 4) for k, v in nli_probs.items()}

    def predict_baseline(self, claim: str, combined_evidence: str) -> Tuple[str, float, Dict[str, float]]:
        """Inference using calibrated Logistic Regression on TF-IDF features."""
        text = f"Claim: {claim} Evidence: {combined_evidence}"
        if self.baseline_model is None:
            return "NEI", 0.33, {"false": 0.33, "true": 0.33, "NEI": 0.34}
            
        if hasattr(self.baseline_model, "predict_proba"):
            probs = self.baseline_model.predict_proba([text])[0]
            pred_id = int(np.argmax(probs))
            confidence = float(probs[pred_id])
            class_probs = {
                "false": round(float(probs[0]), 4),
                "true": round(float(probs[1]), 4),
                "NEI": round(float(probs[2]), 4)
            }
        else:
            pred_id = int(self.baseline_model.predict([text])[0])
            confidence = 0.85
            class_probs = {"false": 0.0, "true": 0.0, "NEI": 0.0}
            class_probs[LABEL_MAPPING[pred_id]] = 1.0
            
        return LABEL_MAPPING[pred_id], round(confidence, 4), class_probs

    def verify_claim(self, claim: str, evidence_docs: List[Dict[str, Any]] = None,
                     raw_evidence_text: str = None, evidence_text: str = None,
                     engine_choice: str = "auto") -> Dict[str, Any]:
        """
        Comprehensive Evidence-Grounded Verification with multi-evidence conflict aggregation.
        """
        if raw_evidence_text is None and evidence_text is not None:
            raw_evidence_text = evidence_text

        # Parse candidate evidence sentences
        candidate_sentences = []
        if evidence_docs:
            for doc in evidence_docs:
                txt = doc.get("text", "").strip()
                if txt and len(txt.split()) >= 3:
                    candidate_sentences.append({
                        "text": txt,
                        "source_urls": doc.get("source_urls", []),
                        "doc_id": doc.get("doc_id", None),
                        "retrieval_score": doc.get("score", doc.get("rerank_score", 0.0))
                    })
        elif raw_evidence_text and raw_evidence_text.strip():
            # Split raw text into sentences
            sents = [s.strip() for s in raw_evidence_text.split(".") if len(s.strip().split()) >= 3]
            for s in sents:
                candidate_sentences.append({
                    "text": s,
                    "source_urls": ["Manual Evidence Input"],
                    "doc_id": None,
                    "retrieval_score": 1.0
                })

        # 1. Uncertainty Gating: Missing or insufficient evidence
        if not candidate_sentences:
            return {
                "claim": claim,
                "verdict": "NEI",
                "predicted_label": "NEI",
                "verdict_display": "NEI (Not Enough Information)",
                "confidence": 0.50,
                "class_probabilities": {"false": 0.25, "true": 0.25, "NEI": 0.50},
                "is_uncertain": True,
                "is_conflicting": False,
                "uncertainty_reason": "No adequate evidence provided or retrieved to substantiate or refute this claim.",
                "explanation": "Insufficient factual evidence available. Verification cannot proceed without supporting sources.",
                "supporting_evidence": [],
                "contradicting_evidence": [],
                "neutral_evidence": [],
                "model_engine": "Rule-Gated Uncertainty Engine",
                "active_engine": "Rule-Gated Uncertainty Engine"
            }

        # 2. Select Engine
        use_nli = (engine_choice in ("transformer_nli", "auto", "financial_finetuned", "pretrained_zero_shot")) and self.has_transformer
        
        supporting_evidence = []
        contradicting_evidence = []
        neutral_evidence = []
        
        if use_nli:
            all_records = []
            for sent_info in candidate_sentences:
                stext = sent_info["text"]
                nli_scores = self.predict_nli_single_pair(claim, stext)
                item_record = {
                    "text": stext,
                    "source_urls": sent_info["source_urls"],
                    "doc_id": sent_info["doc_id"],
                    "entailment_score": nli_scores["entailment"],
                    "contradiction_score": nli_scores["contradiction"],
                    "neutral_score": nli_scores["neutral"]
                }
                all_records.append(item_record)
                
                if nli_scores["contradiction"] >= 0.50:
                    contradicting_evidence.append(item_record)
                elif nli_scores["entailment"] >= 0.50:
                    supporting_evidence.append(item_record)
                else:
                    neutral_evidence.append(item_record)

            # Documented Multi-Sentence Aggregation Logic
            has_conflict = len(supporting_evidence) > 0 and len(contradicting_evidence) > 0
            max_contra = max([e["contradiction_score"] for e in all_records] or [0.0])
            max_entail = max([e["entailment_score"] for e in all_records] or [0.0])
            max_neutral = max([e["neutral_score"] for e in all_records] or [0.0])
            
            if has_conflict:
                verdict = "NEI"
                confidence = 0.55
                class_probs = {"false": 0.45, "true": 0.45, "NEI": 0.10}
                is_uncertain = True
                uncertainty_reason = "Conflicting evidence detected across retrieved sources: some sources corroborate while others refute."
                explanation = "Retrieved sources present conflicting accounts. Further authoritative verification is required."
            elif contradicting_evidence and max_contra >= 0.60:
                verdict = "false"
                confidence = max_contra
                class_probs = {"false": max_contra, "true": max_entail, "NEI": max_neutral}
                is_uncertain = False
                uncertainty_reason = None
                explanation = "Direct evidence contradicts or refutes the key assertions made in the claim."
            elif supporting_evidence and max_entail >= 0.60:
                verdict = "true"
                confidence = max_entail
                class_probs = {"false": max_contra, "true": max_entail, "NEI": max_neutral}
                is_uncertain = False
                uncertainty_reason = None
                explanation = "Direct evidence corroborates and confirms the factual premise of the claim."
            else:
                verdict = "NEI"
                confidence = 0.50
                class_probs = {"false": 0.25, "true": 0.25, "NEI": 0.50}
                is_uncertain = True
                uncertainty_reason = "Retrieved evidence does not establish a conclusive relationship (neutral dominates)."
                explanation = "Available evidence is partial, tangential, or lacks direct substantiation."
                
            engine_name = f"Pretrained Cross-Encoder NLI ({self.nli_model_name})"
        else:
            # Calibrated Baseline Classifier
            comb_text = " ".join([s["text"] for s in candidate_sentences])
            verdict, confidence, class_probs = self.predict_baseline(claim, comb_text)
            is_uncertain = (verdict == "NEI") or (confidence < 0.45)
            has_conflict = False
            uncertainty_reason = "Low classification confidence on candidate evidence text." if is_uncertain else None
            
            if verdict == "false":
                explanation = "The evidence context contradicts key elements of the financial claim."
                contradicting_evidence = candidate_sentences
            elif verdict == "true":
                explanation = "The evidence context corroborates the financial assertion."
                supporting_evidence = candidate_sentences
            else:
                explanation = "Insufficient or ambiguous evidence context to reach a definitive verdict."
                neutral_evidence = candidate_sentences
                
            engine_name = "TF-IDF + Calibrated Balanced Logistic Regression"

        # Format probability distribution
        total = sum(class_probs.values()) + 1e-9
        class_probs = {k: round(v / total, 4) for k, v in class_probs.items()}

        return {
            "claim": claim,
            "verdict": verdict,
            "predicted_label": verdict,
            "verdict_display": verdict.upper(),
            "confidence": round(confidence, 4),
            "class_probabilities": class_probs,
            "is_uncertain": is_uncertain,
            "is_conflicting": has_conflict,
            "uncertainty_reason": uncertainty_reason,
            "explanation": explanation,
            "supporting_evidence": supporting_evidence,
            "contradicting_evidence": contradicting_evidence,
            "neutral_evidence": neutral_evidence,
            "model_engine": engine_name,
            "active_engine": engine_name
        }

def candidate_sentences_scores(claim: str, candidate_sentences: List[Dict[str, Any]], verifier: FinancialClaimVerifier) -> List[Dict[str, float]]:
    scores = []
    for s in candidate_sentences:
        res = verifier.predict_nli_single_pair(claim, s["text"])
        scores.append(res)
    return scores

if __name__ == "__main__":
    verifier = FinancialClaimVerifier(use_transformer=True)
    claim = "Did Biden call for an end to shareholder capitalism?"
    evidence = [
        {"text": "Biden did not propose dissolving the stock market or ending shareholding itself.", "source_urls": ["url1"]},
        {"text": "Biden advocated transitioning from shareholder primacy to stakeholder-driven corporate governance.", "source_urls": ["url2"]}
    ]
    res = verifier.verify_claim(claim, evidence_docs=evidence)
    print("\nNLI Verification Output:")
    print("Verdict:", res["verdict"])
    print("Confidence:", res["confidence"])
    print("Engine:", res["active_engine"])
    print("Contradicting Sents:", len(res["contradicting_evidence"]))
    print("Supporting Sents:", len(res["supporting_evidence"]))
