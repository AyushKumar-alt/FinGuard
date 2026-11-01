# FinGuard Human-Adjudicated Evidence Relevance and Stance Protocol

## 1. Objective and scope

This protocol specifies a blinded human assessment of the candidate passages in the FinGuard root-cause investigation. It measures whether retrieved passages are relevant to a claim, what evidential stance each passage takes, and whether claim-specific financial details match.

This is annotation preparation, not a result report. The two reviewer CSVs contain blank annotation fields; no human judgments or evaluation metrics have been created. The original source file is `outputs/experiments/root_cause_investigation/human_relevance_review.csv` and must remain unchanged.

The sample contains 100 claims and 500 claim-passage pairs. These are reranker ranks 1–5 from the saved 15-candidate trace for each sampled claim. They are **not** necessarily the four passages selected for NLI in the default pipeline. The sample is stratified by claim-level gold class in its private mapping (34 false, 33 true, 33 NEI); reviewers must not be shown that class or the sampling strata.

## 2. Unit of annotation and blinding

The annotation unit is one **claim + one passage**. Reviewers assess the displayed passage on its own merits in relation to the displayed claim. The five pairs for a claim share an anonymous claim-group ID, allowing reviewers to see them as a set; passage order is randomized and reranker rank is hidden.

Reviewer files are:

- `data/human_evaluation/reviewer_1.csv`
- `data/human_evaluation/reviewer_2.csv`

They have the same cases, randomized order, IDs, columns, and blank labels. Give each file only to its assigned reviewer. Do not share either completed file with the other reviewer before both independent submissions are locked. Neither reviewer file includes original claim IDs, gold labels, model predictions, retrieval ranks, scores, heuristic match fields, or source URLs. Claim and passage text remain visible because they are the objects being judged; text may itself contain a fact-check phrase, which should not be treated as proof unless the passage supplies evidence for the proposition.

`data/human_evaluation/private_evaluation_mapping.csv` is restricted to the evaluation custodian. It maps anonymous IDs to original claim IDs, source passage IDs, ranks, URLs, scores, gold labels, and the original heuristic/judgment fields. Do not distribute it to reviewers or store it beside reviewer copies with shared access. The original artifact contains no model-predicted verdict column; none is added to the reviewer files.

## 3. Annotation fields and response options

### Topical relevance

Choose exactly one:

- **Relevant** — The passage directly addresses the central proposition or a material component of the claim and could bear on its truth.
- **Partially relevant** — The passage concerns the same broader subject/entity or addresses only one component, but leaves a material part of the claim unresolved.
- **Irrelevant** — The passage has no meaningful bearing on the proposition, even if it shares a company, person, event, or keywords.

Relevance is about topical/propositional connection, not whether the passage supports the claim or whether the claim is true.

### Evidence stance

Choose exactly one:

- **Supports claim** — The passage asserts or provides evidence that would make the claim more likely to be true.
- **Contradicts claim** — The passage asserts or provides evidence that would make the claim more likely to be false.
- **Neither supports nor contradicts** — The passage is understandable but does not establish either direction. This includes background information or a related fact that does not answer the claim.
- **Insufficient information** — The passage is relevant, but it lacks necessary detail to decide its direction (for example, it gives a total without the comparison period or denominator required by the claim).
- **Cannot determine** — The text is too ambiguous, corrupted, internally inconsistent, or otherwise unclear for a reliable stance judgment.

Do not equate topical relevance with support. A passage can be relevant and neutral, or partially relevant and insufficient.

### Factual specificity

For each criterion, choose **Yes**, **No**, **Unclear**, or **Not applicable**:

1. `specificity_entity` — Does the passage identify the same entity as the claim (company, government, person, or other subject)?
2. `specificity_financial_metric` — Does it address the same financial/economic measure or proposition component?
3. `specificity_amount_percentage` — Where a numeric amount/percentage is claimed, does the passage give the matching value or a clearly equivalent value?
4. `specificity_currency_unit` — Where a currency, scale, unit, denominator, or comparable basis is material, does the passage use the matching one?
5. `specificity_reporting_period` — Where a date, quarter, fiscal year, or comparison period is material, does the passage match it?
6. `specificity_direction_polarity` — Where direction or polarity is material (increase/decrease, positive/negative, negation), does the passage match it?

Use **Not applicable** only when the criterion is not stated or needed by the claim. Use **Unclear** when it is relevant but the supplied passage does not allow a decision. A literal value mismatch is **No**, even if the passage is topically relevant. Do not infer missing values from outside knowledge.

### Optional explanation

`reviewer_explanation` is optional free text. Use it for concise reasoning, ambiguity, units/period interpretation, or why a passage is related but not probative. Avoid adding external claim verdicts.

## 4. Examples

**Same company, no claim evidence:** Claim: “Company A’s quarterly revenue rose 12%.” Passage: “Company A announced a new sustainability initiative.” The passage names the same company, but says nothing about revenue. Depending on the study’s relevance threshold, mark it **Partially relevant** only if the shared context meaningfully bears on the claim; if it is merely an unrelated company announcement, mark **Irrelevant**. In either case it does not support or contradict the revenue claim. This illustrates that relevance and stance are separate judgments.

**Same metric, wrong period:** Claim: “Company A’s FY2024 operating income fell.” Passage: “Operating income rose in FY2023.” This is topically relevant, but contradicts the direction only for a different period; mark period specificity **No** and do not treat it as direct evidence about FY2024.

**Matching subject, no amount:** Claim: “The bank’s deposits increased by 8% in Q2.” Passage: “The bank reported higher deposits.” This can be relevant and supportive of the direction, but the amount/percentage and possibly the reporting period are not established; use **Unclear** for those applicable specificity fields.

**Related topic without a claim answer:** Claim: “The bill allocated $X to small businesses.” Passage: “The bill included several pandemic relief programs.” The passage is related but does not provide the allocation or establish the claim; usually partially relevant, with stance **Neither supports nor contradicts** or **Insufficient information** depending on whether the text gives any probative detail.

## 5. Ambiguous cases and consistency rules

- Judge the exact proposition and scope in the claim; do not silently broaden it.
- Treat entity, metric, value, unit, period, and direction as separate checks. A match on one does not make the entire claim supported.
- A passage can be topically relevant but factually wrong, incomplete, or stance-neutral.
- If an article excerpt quotes a claim without independently corroborating it, do not treat the quotation alone as proof. Annotate the stance of the passage as presented and explain the limitation.
- A passage that explicitly reports a fact-check verdict is still judged for whether its presented reasoning/evidence bears on the claim; the verdict word alone is not independent supporting evidence.
- If multiple propositions occur in one claim, annotate the passage against the claim as written and note which component it addresses.
- Do not browse for the gold label or search the source URL. Source URLs are omitted from reviewer files to preserve blinding and consistent passage-only judgments.
- If no confident single stance is possible because text is ambiguous, choose **Cannot determine**, not a guessed direction.

## 6. Independent review, disagreement, and adjudication

Reviewers complete their files independently and do not communicate case-level judgments before submission. The evaluation custodian checks only structural validity (known IDs, permitted response values, and completion status) before merging the two submissions.

After both initial files are locked:

1. Preserve and analyze the original independent labels before any discussion.
2. Flag every pair with disagreement on topical relevance, stance, or an applicable specificity field.
3. A third adjudicator, if available, reviews the claim and passage without seeing gold labels or model outputs and records the final decision plus a short rationale.
4. If no third adjudicator is available, the two reviewers discuss flagged cases only after initial labels are locked. Record the consensus separately; if consensus is not reached, retain the disagreement and mark the adjudicated value unresolved rather than forcing a label.
5. Never overwrite the original independent labels with adjudicated labels.

### Single-reviewer pilot

If only one reviewer is available, use `reviewer_1.csv` as a pilot and leave `reviewer_2.csv` unused. Record that the study is single-reviewed. Do not report inter-annotator agreement or claim adjudication. A second reviewer should independently annotate at least the full sample (preferred) or a prespecified subset before reliability or final adjudicated results are reported. Single-reviewer labels are provisional and susceptible to individual interpretation bias.

## 7. Inter-annotator agreement

Calculate agreement only after both reviewers have completed and locked their independent files, and before adjudication:

- Report raw exact agreement and a confusion matrix for each field.
- Use weighted Cohen’s kappa for ordinal topical relevance (irrelevant < partially relevant < relevant).
- Use nominal Cohen’s kappa for the five-category stance field.
- For the six specificity fields, report agreement on Yes/No among applicable, determinate judgments; report **Unclear** and **Not applicable** rates separately. A multi-category agreement statistic may additionally be reported if those categories are retained.
- Because category prevalence can make kappa unstable, include category marginals and raw agreement; optionally report Gwet’s AC1 as a prevalence-robust supplement, not a replacement selected after seeing results.
- Use claim-clustered bootstrap intervals if confidence intervals are reported, since five passages from the same claim are not independent.

Do not interpret high agreement as evidence that the retrieval system is good; reliability measures consistency, not validity.

## 8. Planned evaluation after annotation

No human evaluation metrics are calculated during preparation. Once annotation is complete and the labels are validated:

- **Reranked-passage relevance:** Map relevant/partially relevant/irrelevant to gains 2/1/0 for graded NDCG@5; report per-claim macro NDCG@5 and P@5 using a clearly stated threshold (fully relevant as the primary strict threshold, with relevant-or-partial as a secondary threshold). Report the proportion of claims with at least one fully relevant passage and rank of the first fully relevant passage. Include per-rank relevance proportions.
- **Stance profile:** Report support/contradiction/neutral-or-insufficient distributions by topical relevance and claim-level gold class, with claim-level grouping. A passage stance is not itself a claim verdict; do not count neutral or missing evidence as evidence against a claim.
- **Specificity:** Report per-field Yes/No proportions among applicable determinate judgments, alongside Unclear and Not applicable rates. Break down by claim class and topical relevance where useful.
- **Claim-level evidence coverage:** Describe how often the top five contain human-judged supporting evidence for gold-true claims or contradicting evidence for gold-false claims. Report NEI separately. Do not infer a final system prediction from these labels unless a claim aggregation rule is specified in advance and evaluated on a separate, appropriate protocol.
- **Limits:** This sample contains only ranks 1–5 from the reranked top 15 for 100 stratified claims. It cannot establish recall over the full corpus, candidate-generation recall, or comparative BM25-vs-dense effectiveness. The sample’s class balance is stratified and is not the population prevalence; report class-stratified results and avoid unweighted population claims unless reweighted using a justified target distribution.

Keep retrieval metrics, passage relevance/stance judgments, and claim-verification metrics separate. Do not merge these annotation results with prior silver or automatically generated judgments as if they were human gold labels.

## 9. Provenance and completion checks

The source file has 500 rows, 100 unique numeric claim IDs, five distinct reranker ranks (1–5) per claim, and 469 unique passage texts. All source rows have claim, passage, source URL, BM25 score, reranker score, heuristic fields, and a gold label; the three original human-judgment fields are blank. The sample contains 34 false, 33 true, and 33 NEI claims. It has no exact duplicate rows or repeated passage within a claim; some passage texts recur across different claims.

Each source row was matched against `retrieval_error_analysis.csv` by claim ID and reranker rank: all 500 passages, source URLs, reranker scores, and BM25 scores match the saved trace. Each sampled claim has 15 trace candidates. Thus ranks 1–5 are verified as the first five by reranker rank, not merely as five retrieved passages. The numeric claim IDs also map to the corresponding positional records in `finfact.json`; claim text (after whitespace normalization) and gold labels match. The trace provides a corpus `doc_id`, included only in the private mapping. The rank and score fields are intentionally withheld from reviewers.

The two reviewer files are annotation templates, not completed data. Before analysis, validate that both contain exactly the same anonymous IDs, every annotation field is populated with an allowed value or explicitly marked unresolved, and no reviewer file contains private mapping columns.
