# FinGuard human annotation release manifest

**Manifest date:** 2026-11-01  
**Release status:** **NOT READY**

## Verified artifacts

SHA-256 values are for the files at the paths below. CSV row counts exclude the header. “Reviewer-facing” means only that named file may be individually delivered; do not share the containing data directory wholesale.

| File | Rows | Purpose | Access | SHA-256 |
|---|---:|---|---|---|
| `outputs/experiments/root_cause_investigation/human_relevance_review.csv` | 500 | Original source annotation sample | Custodian only; not for reviewers | `dab1cf90039d285d5ee6c63ac68dd2d710f74e06f9babcff13518a072f8013d4` |
| `outputs/experiments/root_cause_investigation/human_relevance_review.release-backup-2026-11-01.csv` | 500 | Separate byte-for-byte backup; Windows read-only attribute set | Custodian only | `dab1cf90039d285d5ee6c63ac68dd2d710f74e06f9babcff13518a072f8013d4` |
| `data/human_evaluation/reviewer_1.csv` | 500 | Reviewer 1's blank annotation template | Reviewer 1 only; share this file individually | `8b46dcf092a1c8708a25c7be1dfad0ac5c0688a0d56fb9c3d7a5d1ebcf0e6918` |
| `data/human_evaluation/reviewer_2.csv` | 500 | Reviewer 2's identical blank annotation template | Reviewer 2 only; share this file individually | `8b46dcf092a1c8708a25c7be1dfad0ac5c0688a0d56fb9c3d7a5d1ebcf0e6918` |
| `data/restricted_evaluation/private_evaluation_mapping.csv` | 500 | Restricted anonymous-pair join to original IDs, gold labels, ranks, URLs, scores, and source fields | Evaluation custodian only; never distribute or put in a reviewer-shared folder | `d491b1aba09960d257412df9c352beddf9dd50b9338ddaf4d939d3509db275fa` |
| `docs/human_evaluation/calibration_worksheet.csv` | 25 | Disjoint calibration examples and blank responses | Calibration reviewers; no gold or model metadata | `7a18b5379ab653c0f6be1108408cba4c0d517ad2ab3f82a188ecd8e69d31303d` |
| `docs/human_evaluation/reviewer_briefing.md` | N/A | Reviewer workflow and decision guidance | Both reviewers | `1bf113f07367244ffbb029e24ce5f2800a18d4dbe98acb59a1a65475b0a26267` |
| `docs/human_evidence_annotation_protocol.md` | N/A | Original annotation protocol; preserved without edits | Both reviewers | `43c53a853c013af59ee347cf96306ce15bb7d8970a7e64490ff5eb143e310e06` |
| `docs/human_evaluation/protocol_addendum_v1.0_2026-11-01.md` | N/A | Dated, versioned decision-rule clarification | Both reviewers after custodian approval | `b36f6ed338a808c3452740dab92668507b6da282253eba66f5448911781b02fd` |
| `docs/human_evaluation/annotation_readiness_checklist.md` | N/A | Release and post-annotation process checklist | Custodian; share with reviewers if useful | `75f88027fb78da3b8acc0c05975d08665bf659347bf3229b7d41af9fb0e255f4` |

This manifest's SHA-256 is not embedded in itself. Compute it after saving if an external release record requires a manifest digest.

## Checks completed

- Both reviewer files have 500 rows, identical pair/group IDs and contents, 100 anonymous claim groups with exactly five pairs each, and blank annotation fields.
- Reviewer schemas contain no gold labels, original claim IDs, predictions, ranks, scores, heuristic columns, or URLs.
- The restricted mapping copy has exactly 500 records and the same SHA-256 as the former mapping at `data/human_evaluation/private_evaluation_mapping.csv`; all 500 anonymous pair IDs join correctly to both reviewer templates. The former mapping was removed only after these copy, digest, row-count, and join checks passed.
- The reviewer distribution directory `data/human_evaluation/` now contains only `reviewer_1.csv` and `reviewer_2.csv`; the private mapping is at `data/restricted_evaluation/private_evaluation_mapping.csv`.
- Actual operating-system/cloud access permissions have **not** been independently tested or verified. Repository directory separation is not proof of access-control enforcement.
- The original source and its read-only backup have identical bytes and SHA-256.
- Current source SHA-256 was computed as listed above. A historical hash-to-hash comparison is unavailable: no earlier digest was recorded. Matching source and backup hashes prove only that the backup matches the current source at backup time; they do **not** prove that the source is unchanged from the earlier preparation state.
- Calibration worksheet has exactly 25 distinct claims, none matching the 100 claims in the private final-evaluation mapping. All response fields are blank. Its schema has no labels, predictions, ranks, scores, or URL/source fields, and each passage links to a saved trace outside the final sample.
- The addendum was reviewed against the original protocol and briefing. It does not change the original protocol in place; it classifies clarifications separately from more-specific decision rules.
- No human judgments or evaluation metrics were generated.

## Custodian actions still required

1. Review and approve the addendum. No reviewer calibration discussion or mutual acceptance has yet been documented; do not represent the rules as reviewer-agreed until recorded.
2. Conduct calibration using the disjoint worksheet, resolve remaining questions, and record the locked protocol/addendum version before formal annotation.
3. Independently configure and verify actual operating-system/cloud access controls for `data/restricted_evaluation/`, then distribute only `reviewer_1.csv` to Reviewer 1 and `reviewer_2.csv` to Reviewer 2. Directory separation is in place, but access-control enforcement and file-level delivery separation have not been verified.
4. Record release-time hashes, reviewer identities separately from annotation IDs, and a secure return path. These operational controls are not verified by this repository audit.

## Final release decision: NOT READY

The files and data checks are ready, but reviewer distribution is gated on custodian approval and protocol lock after calibration. Actual operating-system/cloud access permissions remain unverified. The private mapping is now directory-separated from the reviewer templates, but access-control enforcement is not established by that separation. Distribute individual reviewer files only after these gates are satisfied. The source backup is read-only at the filesystem attribute level, not immutable storage.
