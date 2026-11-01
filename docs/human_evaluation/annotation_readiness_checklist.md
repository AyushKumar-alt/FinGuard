# FinGuard annotation readiness checklist

Use this checklist before distributing files and again before analyzing any labels. Mark each item only after the named check is completed. This is a protocol checklist, not an evaluation result.

## File integrity and blinding

- [ ] Confirm `reviewer_1.csv` and `reviewer_2.csv` each have 500 data rows, identical pair IDs and anonymous claim-group IDs, and exactly five rows per group.
- [ ] Confirm both reviewer files have the same schema and ordering, with all annotation response cells blank before distribution.
- [ ] Confirm reviewer files contain no original claim IDs, gold labels, predictions, retrieval ranks, scores, heuristic fields, or URLs.
- [ ] Confirm the private mapping has one row for every pair ID and can join each pair to its original claim ID, source row, passage/document ID, gold label, rank, source URL, and scores.
- [ ] Verify the original `outputs/experiments/root_cause_investigation/human_relevance_review.csv` is preserved. Its prepared-file SHA-256 is `dab1cf90039d285d5ee6c63ac68dd2d710f74e06f9babcff13518a072f8013d4`; compare before release.
- [ ] Keep the private mapping under evaluation-custodian access; distribute each reviewer file only to its named reviewer. Do not place the mapping in a shared reviewer folder.
- [ ] Record the calibration worksheet's sample source and confirm its claim IDs do not occur in the final evaluation mapping. Keep calibration cases and labels out of the evaluation dataset.

## Reviewer independence and calibration

- [ ] Send reviewers separate instructions and separate reviewer CSVs; do not expose one reviewer's responses to the other.
- [ ] Have reviewers complete the calibration worksheet independently and lock/return those responses before discussion.
- [ ] Discuss calibration disagreements without gold labels, predictions, URLs, or scores. Calibration labels are not evaluation data.
- [ ] Resolve the protocol's boundary cases: Partial vs. Irrelevant; Neither vs. Insufficient information; general direction vs. exact amount; wrong period vs. contradiction; quoted assertion vs. corroborated evidence.
- [ ] If ambiguity changes the decision rule, document the resolution in a dated, versioned protocol addendum and distribute the same locked protocol to both reviewers before formal annotation.
- [ ] Record reviewer identities separately from annotation IDs and establish the submission deadline and secure return path.

## Completion validation and analysis gate

- [ ] After both submissions arrive, preserve read-only copies and record hashes before merging.
- [ ] Validate that each file contains exactly the expected pair IDs once, no unexpected rows, and only allowed values for each field; check required fields are complete and explanations remain optional.
- [ ] Check that claim/passage text and anonymous IDs were not changed and that no private fields or gold labels were added.
- [ ] Merge by pair ID only after both initial files are locked. Preserve each reviewer's labels in separate columns; never overwrite raw labels with adjudicated values.
- [ ] Calculate and report inter-annotator agreement on independent labels before adjudication: exact agreement/confusion matrices; weighted kappa for ordinal relevance; nominal kappa for stance; specificity agreement with Unclear/Not applicable rates; include marginals and claim-clustered uncertainty if used.
- [ ] Flag disagreements by field. Use a blinded third adjudicator where available; otherwise discuss only after independent submission and record consensus separately. Retain unresolved disagreement rather than forcing consensus.
- [ ] Do not calculate relevance, stance, specificity, or claim-level human evaluation metrics until actual human annotation is complete, validated, and the analysis plan is locked.
- [ ] Report sample limits: 100 stratified claims, five reranked passages per claim, calibration disjoint from evaluation; no claim about corpus-wide recall or candidate-generation recall from this sample.

## Current preparation status

Verified before distribution: both prepared reviewer files contain 500 identical blank cases, 100 anonymous claim groups with five passages each, and omit the listed sensitive columns; the private mapping joins all 500 source records. A separate pool of 575 claims with saved reranker traces exists outside the final evaluation sample, sufficient for a disjoint 25-example calibration worksheet. The worksheet uses different claim IDs, but comes from the same root-cause experiment and corpus; it is not an independent external dataset. The custodian should still check the release hashes and control actual file access when distributing.
