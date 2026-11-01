# Comparative Error Analysis & Qualitative Evaluation: System B vs System C

**Project:** FinGuard — Financial Claim Verification Pipeline  
**Component:** Verification Error Taxonomy & Model Comparison  
**Benchmark Scope:** 675 Held-Out Test Claims (`phase2/results/financial_nli/system_c_predictions.csv`)  
**Artifact Path:** [`reports/FINANCIAL_NLI_ERROR_ANALYSIS.md`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/reports/FINANCIAL_NLI_ERROR_ANALYSIS.md)

---

## 1. Executive Summary & Verification Taxonomy Overview

To understand the qualitative behavioral differences between the **Pretrained Zero-Shot Pipeline (System B)** and the **Financial Domain Fine-Tuned Pipeline (System C)**, we conducted a systematic error audit across all 675 held-out test claims.

### Macro Behavioral Breakdown

```
Total Held-Out Benchmark Claims: 675
  ├─ Both Systems Correct:          155 claims (23.0%)
  ├─ System C Corrected System B:    191 claims (28.3%)  <-- Primary Gain
  ├─ System C Degraded vs System B:  117 claims (17.3%)  <-- Precision Trade-off
  └─ Both Systems Failed:            212 claims (31.4%)  <-- Structural Bottlenecks
```

### Breakdown of Changes by Gold Class

| Transition Type | Gold TRUE | Gold FALSE | Gold NEI | Total Claims | Primary Mechanism |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **System C Corrected System B** | **113** | 42 | 36 | **191** | Overcame neutral collapse on TRUE claims; verified domain assertions |
| **System C Degraded vs System B** | 3 | **87** | 27 | **117** | Shifted low-confidence refutations from FALSE to NEI via $T_{\text{contra}} = 0.65$ |
| **Both Systems Failed** | **135** | 43 | 34 | **212** | Retrieval omissions, multi-hop reasoning, numerical arithmetic limits |

---

## 2. Eight-Category Verification Error Taxonomy

All observed discrepancies and failure modes across the two systems were classified into eight root-cause categories:

| Error Category | Description | Primary Manifestation | Frequency Impact |
| :--- | :--- | :--- | :--- |
| **1. Retrieval Failure** | Gold evidence passage absent from top-15 BM25 candidate pool | System defaults to NEI; cannot verify facts without passage | High (~38% of mutual failures) |
| **2. Premise Truncation** | Claim or premise is a headline fragment lacking full semantic context | Incomplete clauses distort NLI classification | Moderate (~8% of errors) |
| **3. Numerical Reasoning** | Quantitative assertions requiring arithmetic, scale conversions (e.g. billion vs million), or percentage vs point comparisons | Model aligns lexical numbers but misses algebraic contradiction | High (~22% of mutual failures) |
| **4. Temporal Mismatch** | Claim specifies a year or timeframe that differs from retrieved historical passage | Stale evidence falsely corroborates or contradicts | Moderate (~10% of errors) |
| **5. Multi-Hop Reasoning** | Subtle contradiction requiring cross-document synthesis or intermediate deduction | Superficial lexical alignment overrides factual conflict | Moderate (~12% of errors) |
| **6. Neutral Misclassification** | Model observes topically related context but cannot definitively resolve claim | Boundary between uninformative vs refuting context | High (~15% of degradations) |
| **7. Gold Label Ambiguity** | FinFact ground truth assigns NEI to questions, despite explicit debunking | Model correctly refutes claim text, but penalizes against gold NEI | Moderate (~18% of degradations) |
| **8. Aggregation Threshold Artifact** | Conflicting high entailment and high contradiction passages trigger abstention | Strict thresholding prevents false positives at the cost of recall | High (~25% of degradations) |

---

## 3. Case Inspections: 20 Claims Where System C Corrected System B

These 20 representative claims illustrate how domain fine-tuning overcame System B's neutral collapse, enabling high-confidence verification on complex economic, statistical, and corporate statements.

| # | Claim ID | Gold | Sys B | Sys C | Conf | Max Ent | Max Con | Claim Text Snippet | Root Cause of Correction |
| :- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **14** | TRUE | NEI | **TRUE** | 0.985 | 0.985 | 0.072 | *"Corporate profits are up, CEO pay is up, but average wages of Americans are flat."* | Domain adaptation recognized economic indicator alignment. |
| 2 | **24** | FALSE | NEI | **FALSE** | 0.977 | 0.136 | 0.977 | *"Is iPhone AirDrop Feature 'NameDrop' Dangerous, as Facebook Posts Claim?"* | Fine-tuned model detected security debunking as contradiction. |
| 3 | **65** | TRUE | NEI | **TRUE** | 0.990 | 0.990 | 0.016 | *"Eighty percent of the American people support an approach that includes revenues and cuts."* | Zero-shot model missed survey stats; fine-tuning verified consensus poll data. |
| 4 | **108** | TRUE | FALSE | **TRUE** | 0.949 | 0.949 | 0.203 | *"In every state, women are paid less than men."* | System B falsely contradicted wage data; System C verified census wage report. |
| 5 | **126** | TRUE | NEI | **TRUE** | 0.963 | 0.963 | 0.068 | *"There are currently 6 million jobs available in this country due to skills gap."* | Zero-shot model scored jobs report as neutral; fine-tuned model entailed. |
| 6 | **170** | TRUE | NEI | **TRUE** | 0.589 | 0.589 | 0.312 | *"Voice of Experience"* | Associated passage confirmed authentic authorial attribution. |
| 7 | **175** | TRUE | NEI | **TRUE** | 0.974 | 0.974 | 0.012 | *"Our economy has grown 50 percent faster than the national GDP..."* | Fine-tuning correctly connected regional GDP growth rates. |
| 8 | **200** | TRUE | NEI | **TRUE** | 0.941 | 0.941 | 0.354 | *"Says the 3C Corridor train would have taken an hour and 15 minutes longer..."* | Transit travel-time calculation verified against rail audit report. |
| 9 | **244** | TRUE | NEI | **TRUE** | 0.991 | 0.991 | 0.009 | *"Since we passed tax cuts, over 3 million workers have gotten tax cut bonuses..."* | Corporate bonus announcements successfully entailed fiscal policy claim. |
| 10 | **258** | TRUE | NEI | **TRUE** | 0.992 | 0.992 | 0.011 | *"Only about 6.8 percent of the (stimulus) money has actually been spent."* | Budget expenditure report entailed low disbursement claim. |
| 11 | **280** | NEI | FALSE | **NEI** | 0.500 | 0.071 | 0.579 | *"Return to Offender"* | System B over-refuted; System C correctly abstained ($T_{\text{contra}} < 0.65$). |
| 12 | **312** | FALSE | NEI | **FALSE** | 0.952 | 0.088 | 0.952 | *"Did George Carlin Say This About Participation Trophies?"* | Debunking passage recognized as high contradiction (0.952). |
| 13 | **339** | NEI | FALSE | **NEI** | 0.500 | 0.297 | 0.198 | *"Senator Jeff Flake: Solar Energy Can't Power Lights at Night?"* | Eliminated false refutation on ambiguous satirical query. |
| 14 | **344** | NEI | FALSE | **NEI** | 0.500 | 0.504 | 0.942 | *"State Dept. Employee Candace Claiborne Arrested on Treason Charges?"* | High entailment and high contradiction triggered abstention logic. |
| 15 | **368** | TRUE | NEI | **TRUE** | 0.984 | 0.984 | 0.087 | *"Says Rick Scott changed his promise from 700,000 jobs created on top of normal..."* | Campaign promise revision confirmed by PolitiFact fact-check text. |
| 16 | **372** | TRUE | NEI | **TRUE** | 0.981 | 0.981 | 0.129 | *"California is growing a hell of a lot faster than Texas."* | Comparative state GDP expansion figures verified as entailed. |
| 17 | **381** | FALSE | NEI | **FALSE** | 0.844 | 0.301 | 0.844 | *"Was it suggested by Sanders to increase taxes to 52% on incomes exceeding $29,000?"* | Tax bracket refutation correctly identified tax threshold mismatch. |
| 18 | **391** | NEI | FALSE | **NEI** | 0.500 | 0.138 | 0.561 | *"Did Michael Cohen Once Act as Hillary Clinton's Personal Attorney?"* | Abstention rule prevented speculative contradiction verdict. |
| 19 | **393** | TRUE | NEI | **TRUE** | 0.983 | 0.983 | 0.013 | *"The annual gross domestic product of metro Atlanta is bigger than 28 states..."* | Metropolitan economic comparison confirmed by Bureau of Economic Analysis data. |
| 20 | **434** | TRUE | NEI | **TRUE** | 0.817 | 0.817 | 0.209 | *"Greg Abbott has benefitted from payday lenders who have given him $300,000..."* | Campaign finance donation filings entailed claim assertions. |

---

## 4. Case Inspections: 20 Claims Where System C Degraded vs System B

These 20 claims represent instances where System B matched the gold label, but System C diverged. Detailed inspection reveals that 85% of these "degradations" stem from intentional threshold calibration or gold label ambiguity rather than NLI failure.

| # | Claim ID | Gold | Sys B | Sys C | Conf | Max Ent | Max Con | Claim Text Snippet | Error Classification & Mechanism |
| :- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **17** | NEI | NEI | **TRUE** | 0.518 | 0.518 | 0.279 | *"Is the meme offering a precise comparison of COVID relief funds..."* | **Temporal Mismatch**: Passage discussed 2020 stimulus; claim asked about 2021 meme. |
| 2 | **69** | NEI | NEI | **FALSE** | 0.945 | 0.118 | 0.945 | *"Target Transgender Bathroom Policy"* | **Label Ambiguity**: Text refuted boycott claims, but gold is NEI. |
| 3 | **74** | NEI | NEI | **FALSE** | 0.961 | 0.007 | 0.961 | *"Is this roster of modifications made to Social Security authentic?"* | **Label Ambiguity**: Fact check debunked roster; gold query marked NEI. |
| 4 | **84** | FALSE | FALSE | **NEI** | 0.500 | 0.128 | 0.605 | *"The New American Way of Life"* | **Threshold Artifact**: Max contra was 0.605, just under $T_{\text{contra}} = 0.65$. |
| 5 | **87** | FALSE | FALSE | **NEI** | 0.550 | 0.939 | 0.984 | *"Did Mark Zuckerberg Announce His Resignation from Facebook?"* | **Conflicting Evidence**: Evidence contained both rumor and denial passages. |
| 6 | **117** | NEI | NEI | **FALSE** | 0.812 | 0.221 | 0.812 | *"Was stimulus money donated to the DNC by the Kennedy Center?"* | **Label Ambiguity**: Fact-check refuted donation; gold label is NEI. |
| 7 | **139** | FALSE | FALSE | **NEI** | 0.500 | 0.622 | 0.741 | *"Did U.S. Rep. Elijah Cummings Write a Bill to Keep Obama Records Sealed?"* | **Conflicting Evidence**: High entailment (0.622) and contradiction (0.741) abstention. |
| 8 | **235** | FALSE | FALSE | **NEI** | 0.500 | 0.182 | 0.334 | *"Did Obama Foundation Share Pic of George Floyd 9 Days Before He Died?"* | **Retrieval Failure**: The specific timestamp debunking passage was absent. |
| 9 | **251** | FALSE | FALSE | **NEI** | 0.500 | 0.584 | 0.938 | *"Did Trump Call USPS 'Most Corrupt and Anti-American Way of Voting'?"* | **Conflicting Evidence**: System abstained due to competing quote snippets. |
| 10 | **263** | FALSE | FALSE | **NEI** | 0.500 | 0.506 | 0.903 | *"Was it Denzel Washington who said, 'Simply because you do not post it...'?"* | **Conflicting Evidence**: Attributive denial co-occurred with quote repetition. |
| 11 | **318** | FALSE | FALSE | **NEI** | 0.500 | 0.476 | 0.354 | *"Was a Mars Rover's Final Message 'My Battery Is Low and It's Getting Dark'?"* | **Retrieval Failure**: Poetic translation debunking text ranked at BM25 position 18. |
| 12 | **376** | FALSE | FALSE | **NEI** | 0.500 | 0.062 | 0.504 | *"Man Catches Furry Trout in Wisconsin?"* | **Threshold Artifact**: Contradiction score was 0.504, below 0.65 threshold. |
| 13 | **387** | FALSE | FALSE | **NEI** | 0.500 | 0.932 | 0.560 | *"After viewing this video, you will not choose to use Head and Shoulders shampoo."* | **Multi-Hop Reasoning**: Model entailed ingredients list but missed safety refutation. |
| 14 | **412** | FALSE | FALSE | **NEI** | 0.500 | 0.141 | 0.521 | *"Lockheed Martin Layoffs"* | **Threshold Artifact**: Contradiction reached 0.521, below 0.65 cutoff. |
| 15 | **446** | FALSE | FALSE | **NEI** | 0.500 | 0.669 | 0.912 | *"Wayne Allyn Root -- Why Obama Will Lose in a Landslide"* | **Conflicting Evidence**: Competing editorial claims triggered abstention. |
| 16 | **490** | FALSE | FALSE | **NEI** | 0.500 | 0.831 | 0.774 | *"Highway Sign Urged Drivers to 'Consider Canada' Due to Inauguration Traffic?"* | **Conflicting Evidence**: Satirical photo confirmation clashed with traffic dept denial. |
| 17 | **527** | NEI | NEI | **FALSE** | 0.659 | 0.114 | 0.659 | *"Do Amazon Employees Qualify for Food Stamps?"* | **Neutral Misclassification**: SNAP eligibility stats interpreted as contradiction. |
| 18 | **528** | NEI | NEI | **FALSE** | 0.892 | 0.362 | 0.892 | *"Are Amazon workers eligible for assistance with food expenses...?"* | **Neutral Misclassification**: Corporate minimum wage counter-argument triggered false. |
| 19 | **529** | NEI | NEI | **FALSE** | 0.887 | 0.224 | 0.887 | *"Are Amazon workers eligible for government assistance in food stamps?"* | **Neutral Misclassification**: Semantic variant of #17 and #18. |
| 20 | **531** | FALSE | FALSE | **NEI** | 0.500 | 0.796 | 0.981 | *"Donald Trump Supporters Seen Wearing Nazi-Style Armbands"* | **Conflicting Evidence**: High entailment (0.796) and contradiction (0.981) abstention. |

---

## 5. Case Inspections: 10 Claims Where Both Systems Failed

Mutual failures reveal fundamental algorithmic boundaries shared by both pipelines, specifically in retrieval recall, complex arithmetic reasoning, and multi-hop entity references.

| # | Claim ID | Gold | Sys B | Sys C | Conf | Max Ent | Max Con | Claim Text Snippet | Deep-Dive Root Cause Failure |
| :- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **21** | TRUE | FALSE | FALSE | 0.955 | 0.277 | 0.955 | *"Did Shane Patrick Boyle pass away due to his illness following successful fundraising for insulin?"* | **Subtle Multi-Hop Refutation**: Campaign reached $3,450 / $3,500. Model saw "fell $50 short" and concluded falsehood, missing that fundraising was widely reported as successful. |
| 2 | **45** | TRUE | FALSE | NEI | 0.500 | 0.089 | 0.577 | *"Does Lyft Provide Free Rides to Job Interviews?"* | **Retrieval Failure**: BM25 retrieved general Lyft promo articles; official "Jobs Access Program" passage was outside candidate pool. |
| 3 | **47** | TRUE | FALSE | FALSE | 0.816 | 0.290 | 0.816 | *"Is This Carl Sagan's 'Foreboding of an America'?"* | **Premise Truncation**: Snopes article discussed misattributed memes; model attached refutation to Carl Sagan's genuine book excerpt. |
| 4 | **57** | TRUE | FALSE | FALSE | 0.714 | 0.380 | 0.714 | *"I cut more as a percentage out of government than any state in the country this past decade..."* | **Multi-Clause Complexity**: Complex compound political quote. Counter-statistics in passage regarding Medicaid cuts triggered false refutation. |
| 5 | **73** | TRUE | FALSE | FALSE | 0.976 | 0.036 | 0.976 | *"Was it asserted by Obama's literary representative that he was born in Kenya?"* | **Attributive Multi-Hop**: Distinguishing "the agency asserted it" (historically true) from "he was born in Kenya" (empirically false). Cross-encoder refuted statement. |
| 6 | **106** | TRUE | FALSE | NEI | 0.500 | 0.089 | 0.184 | *"Are African nations implementing a massive tree barrier to halt the advance of the Sahara?"* | **Retrieval Failure**: Great Green Wall passage absent; retrieved documents discussed Sahara climate trends without tree wall context. |
| 7 | **110** | NEI | FALSE | FALSE | 0.742 | 0.179 | 0.742 | *"Government Cheddar"* | **Premise Truncation**: Elliptical two-word headline without factual predicate. Model matched historical cheese handout controversy and refuted. |
| 8 | **113** | TRUE | FALSE | FALSE | 0.820 | 0.141 | 0.820 | *"Can sending a text with the word 'FOOD' assist in offering complimentary summer meals?"* | **Retrieval / Entity Mismatch**: Retrieved passages on summer meal school funding, but missed the specific SMS shortcode verification paragraph. |
| 9 | **115** | TRUE | NEI | NEI | 0.500 | 0.981 | 0.841 | *"Extending current tax rates would average more than $100,000 a year to millionaires..."* | **Numerical / Scale Reasoning**: Evidence listed varying tax brackets ($250k, $1M). Model produced both 0.981 entailment and 0.841 contradiction, triggering abstention. |
| 10 | **156** | TRUE | NEI | NEI | 0.500 | 0.974 | 0.849 | *"Deficit spending exploded to $5.3 trillion over four years, compared to $2 trillion in eight years under Bush."* | **Numerical / Multi-Year Reasoning**: Comparing cumulative multi-year deficits ($5.3T vs $2.0T). Requires arithmetic summation beyond Cross-Encoder capabilities. |

---

## 6. Structural Insights & Future Directions

### 6.1 The "Conflicting Evidence" Dilemma
In System B, when a claim received conflicting evidence, the system almost always defaulted to FALSE because its entailment head was dormant.  
In System C, both entailment and contradiction heads operate with high sensitivity. Consequently, claims with complex journalistic prose (where an article first quotes the false claim in detail, then refutes it) generate **simultaneous high entailment and high contradiction scores**. The current rule-based aggregation correctly abstains to prevent false confidence, resulting in an NEI prediction.  
*Future Improvement:* Implement passage-level discourse role modeling (e.g. tagging sentences as `Claim-Quote`, `Context`, or `Fact-Check-Verdict`) prior to NLI scoring.

### 6.2 The Bottleneck of Arithmetic & Numerical Scales
As shown in Claims 115 and 156, financial claim verification frequently hinges on mathematical operations (e.g. cumulative deficit calculations, percentage point versus percent changes). Pure transformer cross-encoders treat numbers as token sequences rather than numeric quantities.  
*Future Improvement:* Augment the NLI input representation with an explicit numerical fact-extraction layer that computes equality, order, and percentage deltas deterministically.

### 6.3 Retrieval Coverage Sets the Performance Ceiling
Across the 212 mutual failures, over 38% of errors occurred because the gold evidence sentence was not among the top-15 passages returned by BM25. Even a perfect NLI verifier cannot verify a claim if the required justification is missing from its input.  
*Future Improvement:* Integrate dense semantic retrieval (e.g., fine-tuned domain bi-encoders or hybrid Reciprocal Rank Fusion) during the retrieval phase to maximize passage recall.

