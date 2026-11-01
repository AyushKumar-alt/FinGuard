# Financial NLI Human Annotation & Review Guidelines

## Overview
This document specifies the scientific protocol for adjudicating ambiguous claim-evidence pairs in FinGuard Phase 2.

## Label Taxonomy
1. **Contradiction (Label 0)**
   - The premise evidence directly conflicts with, refutes, or negates the key factual assertions of the financial claim.
   - Examples: Direct negation ("did not happen"), conflicting monetary amount ("$10B instead of $20B"), incorrect fiscal year, or debunking of a fabricated rumor.

2. **Entailment (Label 1)**
   - The premise evidence directly substantiates, corroborates, or proves the factual premise of the financial claim.
   - Both the entity, financial metric, direction, and magnitude must align.

3. **Neutral (Label 2)**
   - The premise evidence discusses relevant context, entity history, regulatory background, or general definitions, but neither establishes truth nor refutes the claim.
   - Also assigned when evidence is tangential, incomplete, or refers to a different fiscal period without explicit conflict.

## Ambiguity Adjudication Protocol
- When an evidence passage quotes a false claim without debunking language, annotate as **Neutral** unless accompanied by explicit refutation.
- When an evidence passage mentions the same company but an unrelated topic, annotate as **Neutral**.
