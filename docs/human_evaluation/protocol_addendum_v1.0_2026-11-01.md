# FinGuard Human Evidence Annotation Protocol Addendum

**Version:** 1.0  
**Date:** 2026-11-01  
**Applies to:** `docs/human_evidence_annotation_protocol.md` and the reviewer briefing  
**Status:** Custodian approval required before formal annotation

## Purpose and agreement status

This addendum formalizes the decision rules already described in `docs/human_evaluation/reviewer_briefing.md` so that both reviewers receive the same versioned guidance. It does not replace or silently edit the original protocol.

**Important:** This is a custodian-prepared clarification, not evidence that reviewers have already discussed or mutually agreed to it. No calibration discussion or reviewer sign-off is recorded as of this version. The custodian must present the rules during calibration, record any agreed revision/version, and lock the protocol before distributing the formal 500-pair annotation files. Do not describe this addendum as reviewer-consensus-approved until that occurs.

## Rule classification

“Clarification” below makes an existing distinction more operational without changing its intended construct. “More specific decision rule” below resolves an ambiguity by prescribing a particular permitted label for a recurring case. If the calibration discussion changes a decision rule, issue a new version rather than editing this file in place.

### A. Clarifications of existing definitions

1. **Topical relevance is distinct from stance.** As in protocol sections 3–5, a passage can be relevant yet neutral, or partially relevant yet insufficient. Shared entity/topic alone does not make a passage relevant: it must meaningfully bear on a material claim component. Use *Partially relevant* for meaningful but incomplete connection; use *Irrelevant* for incidental overlap with no bearing on the proposition.
2. **Neither vs. insufficient information.** *Neither supports nor contradicts* applies when an understandable passage contributes no probative information either way. *Insufficient information* applies when a passage bears on the proposition but lacks a necessary detail. *Cannot determine* is reserved for text too ambiguous, corrupted, or internally inconsistent to judge.
3. **Specificity fields are separate from overall stance.** Assess direction, amount/percentage, unit/currency, period, metric, and entity independently. A passage can match one specificity dimension while leaving another unresolved. Do not infer missing details from outside knowledge.
4. **Wrong-period evidence is scoped to its period.** A trend in a different reporting period is not evidence about the claim's stated period merely because entity and metric match. This follows the protocol's instruction to judge the exact proposition and scope and not treat a different-period passage as direct evidence for the claim.
5. **A quoted assertion establishes attribution, not necessarily truth.** When the claim asks whether a person made a statement, a passage containing that statement can bear on attribution. When the claim asks whether the underlying financial proposition is true, a quotation alone is not independent corroboration. A reported fact-check verdict alone is not proof; assess the presented reasoning/evidence.

### B. More specific decision rules for consistent application

1. **Exact amount claim; passage gives direction only:** If the claim asserts an exact amount/percentage and the passage establishes only a direction (for example, “increased”), label stance **Insufficient information** for the full claim. Mark direction specificity **Yes** and amount/percentage specificity **Unclear** (or **No** only if a stated amount conflicts). If the claim asserts direction only, that same passage may be **Supports claim**.
2. **Same entity/metric but a different reporting period:** Mark reporting-period specificity **No**. Ordinarily label stance **Insufficient information** for the claim's period, not **Contradicts claim**. Choose **Contradicts claim** only when evidence opposes the proposition within the period and scope stated in the claim. If the only connection is incidental word overlap, use **Irrelevant** for relevance and **Neither supports nor contradicts** for stance.
3. **Passage merely quotes a claim about its truth:** Ordinarily label stance **Neither supports nor contradicts** for the underlying financial proposition; the quote proves at most that the assertion was made. If the evaluated claim is explicitly about whether that assertion was made, the quote may support that attribution proposition. Assess any additional factual evidence in the passage separately.

These are annotation rules, not model outputs or suggested labels for any calibration/evaluation example. The examples in the briefing remain illustrative only.

## Consistency with the original protocol

This addendum narrows no response vocabulary and does not change the unit, blinding rules, fields, adjudication process, or planned metrics. It is consistent with:

- Section 3: topical relevance and stance are separate; an incomplete passage may be insufficient.
- Section 4: direction can be supported while amount/period remain unspecified; wrong-period material does not directly establish the target claim.
- Section 5: judge the exact claim scope; quotations and verdict phrases are not independent proof.

Where the original protocol permits judgment but this addendum specifies a default for a recurring boundary case, use this addendum only after custodian approval and protocol lock. If the two documents appear inconsistent in a case not covered here, pause and ask the custodian; do not improvise a new rule.

## Approval and change control

Before formal annotation, the custodian records:

- whether both reviewers received this exact version;
- calibration discussion date and whether reviewers accepted these rules;
- any remaining ambiguity and its disposition;
- the locked protocol/addendum version used for annotation.

Preserve this version. Any substantive change requires a new dated version and distribution to both reviewers before formal work starts.
