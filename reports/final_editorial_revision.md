# Final editorial revision — main_GPTreview (2026-10-30)

Three narrowly scoped corrections; no number, figure, table, or finding
changed. Companion reports: `repository_availability_check.md`,
`final_submission_checklist.md`.

| Issue | Manuscript location | Original wording | Revised wording | Justification |
|---|---|---|---|---|
| 1. Categorical grammaticality conclusion | §8.2 | "A pure stimulus-grammaticality artifact is therefore ruled out for the retained constructions." | "The persistence of the effect after the exclusion indicates that the observed reduction cannot be explained solely by that construction's grammaticality issue; it is evidence against one identified alternative explanation, not an elimination of every possible grammatical, pragmatic, or discourse-related confound in the retained sampled constructions." | Sampled review of 360 stimuli cannot establish that every retained construction is free of all linguistic confounds; persistence ≠ elimination. §7.8's parallel passage already used the qualified "not solely attributable" form (verified, unchanged). No other "rules out / eliminates / proves" claims found in sweep. |
| 2. Repository availability | Data and Code Availability (back matter) | "…maintained in a version-controlled repository (URL) that will be made publicly accessible upon publication." | Adds, from verified status: the repository is currently **private**; reviewer/editor access during peer review **will be arranged at submission** (Case C — not claimed as already available); public upon publication; and an explicit statement of what the repository tracks (code + retained run records, manifests, analysis tables, review artifacts) versus what is excluded with hash-verified rebuild instructions (bulk regenerable grid data, license-gated tokenizers). | `gh repo view` verified: exists, PRIVATE; reviewer access not yet arranged; artifact inventory verified against the tracked tree. Distinguishes code availability from experimental-artifact availability. |
| 3. Sample-scoped certification language | Abstract | "…finds unanimous semantic equivalence, excludes one construction on grammaticality…and records contested naturalness." | "…found unanimous semantic-equivalence judgments **within the reviewed sample**, identified one construction excluded on grammaticality grounds…and documented disagreement regarding naturalness." (P3 persistence at 40.8 pp retained.) | The review evidences the 360-item sample, not exhaustive certification of 16,618 prompts. |
| 3 (cont.) | Introduction §1 | "…certifying semantic equivalence unanimously, excluding one construction…" | "…found unanimous semantic-equivalence judgments within the reviewed sample, identified one construction requiring grammaticality-based exclusion…" | Same. |
| 3 (cont.) | §3.3 Methodology | "A review of this form was executed for the evaluated prompt population" | "…executed on a stratified sample of the evaluated prompt population" | Avoids implying exhaustive coverage. |
| 3 (cont.) | §7.8 | "…measured on stimuli that two independent reviewers unanimously certify as meaning-preserving…" | "…measured on a stimulus population whose 360-item reviewed sample both reviewers unanimously judged meaning-preserving…" | The sweep stimuli outnumber the reviewed sample; certification language replaced by judgment language scoped to the sample. |
| 3 (cont.) | §9 Limitations | "…certifies semantic equivalence unanimously and grammaticality outside one excluded construction, but it is sampled…" | "…found unanimous semantic-equivalence judgments and, outside one excluded construction, acceptable grammaticality within the reviewed sample…it certifies neither every generated realization nor that every transformation is natural in every discourse context." | Same, and strengthens the existing sampled-not-exhaustive caveat. |

Untouched by design: §7.8's result counts (360 / 144 P3 / 324 unanimous /
16 excluded), the 40.8-pp sensitivity result, naturalness disagreement, all
Table 10 rows, Δ_{P3−P0} = −0.381 [−0.418, −0.344] (4 occurrences verified in
the compiled PDF), the language-gap estimand explanation, R+D+U identities,
and the invariance/response-bias distinction.

Compile: 31 pages, 0 undefined-reference warnings, no internal revision
labels.
