# FinGuard reviewer briefing

## Before you begin

You will receive one reviewer CSV and, separately, a copy of the calibration worksheet. Do not open or request the private evaluation mapping. The evaluation custodian retains it. Your file contains anonymous case and pair IDs, claim text, passage text, and blank fields only. It intentionally omits original IDs, gold labels, model verdicts, ranks, scores, heuristics, and URLs.

Complete the calibration worksheet independently first. Lock or return your initial calibration responses before discussing the examples together. The worksheet uses claims excluded from the final 500-pair evaluation sample; calibration responses are not evaluation judgments and must never be copied into the evaluation CSV. Use calibration discussion to resolve interpretation questions. Before formal annotation, the custodian should record and version any agreed clarifications, then freeze the protocol. If a disagreement remains material, pause and clarify the rule rather than guessing.

For the evaluation file, annotate every pair independently. Keep your completed file private until both reviewers have submitted and the custodian confirms that initial labels are locked. Do not discuss case-level labels, browse for the source, or search for the claim verdict during independent annotation. Add concise explanations only when helpful; do not put external verdicts or private information in the file.

## Common decision points

- **Relevance is not stance.** A shared company or topic alone is not evidence that the passage bears on the proposition. Use Partially relevant only when the passage provides meaningful context for, or evidence about, at least one material component of the claim. Use Irrelevant for incidental keyword/entity overlap with no bearing on the proposition. A passage can be relevant yet neutral.
- **Neither vs. insufficient information.** Use Neither supports nor contradicts when the passage is understandable but contributes no probative information in either direction. Use Insufficient information when it bears on the proposition but omits a detail needed to resolve it. If the text itself is ambiguous or corrupted, use Cannot determine.
- **Direction vs. exact amount.** Judge stance against the full claim, not an isolated component. If a claim asserts an exact amount/percentage but the passage establishes only the direction, choose Insufficient information for the full claim; separately mark direction specificity Yes and amount/percentage specificity Unclear. If the claim asserts direction only, the same passage may support it. Do not treat direction alone as confirmation of an exact numerical claim.
- **Wrong period is not automatically contradiction.** A passage about the same entity/metric but a different reporting period cannot, by itself, establish or refute the claim’s stated period. Mark period specificity No and ordinarily use Insufficient information for the full claim; reserve Contradicts for evidence that opposes the proposition within the claimed period/scope. If the passage has no meaningful connection beyond a shared term, use Irrelevant/Neither as appropriate.
- **Quoted assertions are not independent corroboration.** A passage that only reports someone making a claim establishes that the statement was made, not that its financial proposition is true; ordinarily judge its stance as Neither supports nor contradicts. If the claim itself is about whether the person made that statement, the quote can bear on that attribution claim. Do not treat a fact-check verdict word by itself as supporting evidence; assess the reasoning or facts presented.
- **Do not fill gaps from memory.** Judge the supplied passage and claim. Use Unclear, Not applicable, Insufficient information, or Cannot determine as defined rather than importing outside facts.

## During and after annotation

Use the exact response values in the protocol and preserve all anonymous IDs. Do not add columns, sort away the IDs, or alter claim/passage text. If a case cannot be resolved under the frozen rules, select the closest permitted response and explain the ambiguity; flag it to the custodian without consulting the other reviewer.

The custodian validates completion and allowed values after both files are locked. Independent labels are retained unchanged. Disagreements are analyzed before adjudication; a third adjudicator is preferred, and consensus discussion occurs only after independent labels are secured. No reviewer should calculate or discuss system performance from the gold mapping.
