# FinGuard Gold-Evidence Benchmark: Annotation & Adjudication Guidelines

## 1. Overview & Core Principles

The **FinGuard Gold-Evidence Relevance Benchmark** measures the ability of retrieval systems to discover evidence that substantiates or refutes financial assertions.

### Golden Rule: Evidence Relevance $\ne$ Claim Veracity
- An evidence sentence that **directly refutes** a False claim is **Grade 3 (Directly Decisive)**, with `support_relation="refutes"`.
- An evidence sentence that **directly confirms** a True claim is **Grade 3 (Directly Decisive)**, with `support_relation="supports"`.
- Do not penalize an evidence passage simply because the claim it discusses is factually incorrect. Relevance measures whether the evidence provides factual grounding to evaluate the assertion.

---

## 2. Four-Tier Relevance Grading Scale

| Grade | Label | Operational Definition | Criteria & Financial Indicators |
| :---: | :--- | :--- | :--- |
| **3** | **Directly Decisive** | Resolves the factual assertion definitively. | Matches the target entity, metric, monetary amount, percentage, and fiscal timeframe. Explicitly confirms or contradicts the assertion. |
| **2** | **Substantially Relevant** | Meaningful factual evidence requiring minor context or inference. | Identifies the key entity and metric, but may lack exact temporal boundaries, or presents partial numbers that require minor arithmetic or contextual interpretation. |
| **1** | **Contextually Relevant** | Background, editorial, or procedural context. | Mentions the target company or financial topic in general terms (e.g. broader industry trends, regulatory framework, editorial commentary) without resolving the specific assertion. |
| **0** | **Judged Not Relevant** | Confirmed unhelpful or off-topic. | Does not discuss the target entity, metric, or event in any material way. |
| **None** | **Unjudged** | Unknown relevance. | Default status for candidate sentences pooled from retrieval systems that have not yet been evaluated by a human reviewer. **Never silently assign 0.** |

---

## 3. Support Relations

Track support relation independently from the relevance grade:
- **`supports`:** The evidence passage corroborates, affirms, or substantiates the claim.
- **`refutes`:** The evidence passage contradicts, disproves, or denies the claim.
- **`mixed`:** The passage presents conflicting accounts (e.g., reports varying estimates from different analysts).
- **`context_only`:** The passage provides factual background (Grade 1 or 2) without taking a definitive stance.
- **`unknown`:** Default for unjudged candidates.

---

## 4. Financial Domain Edge Cases & Concrete Examples

### Example 1: Numerical Values & Currencies
- **Claim:** *"Did the US national debt increase by over $7 trillion under Donald Trump?"*
- **Candidate A:** *"Treasury Department data shows the total public debt outstanding rose from $19.9 trillion in January 2017 to $27.8 trillion in January 2021, an increase of roughly $7.8 trillion."*
  - **Grade:** `3` | **Relation:** `supports` | **Rationale:** Direct official figures verifying the >$7T assertion.
- **Candidate B:** *"The national debt surpassed $20 trillion in late 2017 amid rising government borrowing."*
  - **Grade:** `2` | **Relation:** `context_only` | **Rationale:** Substantially relevant milestone, but incomplete over the full presidential term.
- **Candidate C:** *"The Treasury Department conducts regular weekly auctions of short-term bills."*
  - **Grade:** `0` | **Relation:** `context_only` | **Rationale:** Irrelevant procedural detail.

### Example 2: Directional Language & Refutation
- **Claim:** *"Did Biden call for an 'End to Shareholder Capitalism' in July 2020?"*
- **Candidate A:** *"Although Facebook memes claimed Biden proposed abolishing 401(k) savings, Biden explicitly stated he was proposing a shift from shareholder primacy to stakeholder accountability, not an end to shareholding."*
  - **Grade:** `3` | **Relation:** `refutes` | **Rationale:** Directly clarifies Biden's actual words and refutes the literal claim.
- **Candidate B:** *"Shareholder capitalism has been the prevailing corporate governance doctrine since Milton Friedman's 1970 essay."*
  - **Grade:** `1` | **Relation:** `context_only` | **Rationale:** Academic historical background.

### Example 3: Fiscal Periods & Quarters
- **Claim:** *"Amazon paid zero dollars in federal income tax in FY2018."*
- **Candidate A:** *"According to regulatory SEC Form 10-K filings, Amazon reported a current federal tax rebate of $129 million for the 2018 fiscal year, resulting in an effective federal income tax rate of 0%."*
  - **Grade:** `3` | **Relation:** `supports` | **Rationale:** Direct confirmation of zero federal tax liability for the exact fiscal year.
- **Candidate B:** *"Amazon paid $162 million in federal income taxes in 2019."*
  - **Grade:** `1` | **Relation:** `context_only` | **Rationale:** Different fiscal year; useful for comparative context only.

### Example 4: Incomplete & Ambiguous Claims
- When an assertion cannot be resolved because key parameters are omitted (e.g., missing currency or unclear baseline):
  - Mark incomplete evidence as **Grade 2** with `support_relation="context_only"`.
  - Provide a clear rationale explaining the ambiguity.

---

## 5. Adjudication Workflow

1. **Reviewer Assignment:**
   Annotators inspect candidate pairs in `data/benchmark/v1/pooled_candidates_annotation.csv`.
2. **Independent Verification:**
   Annotator verifies that the evidence text is genuinely traceable to an authoritative source URL and does not hallucinate facts.
3. **Approval:**
   Record is updated with `relevance_grade`, `support_relation`, `annotator_or_reviewer="human_reviewer_id"`, and `adjudication_status="approved"`.
4. **Dispute Resolution:**
   If two annotators disagree on Grade 2 vs Grade 3, a lead adjudicator reviews the rationale and assigns final approved status.
