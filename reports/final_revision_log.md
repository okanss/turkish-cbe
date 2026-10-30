# Final revision log — main_GPTreview (submission version)

Revision of 2026-10-30 applied to `paper/main_GPTreview.tex` +
`paper/sections_gptreview/` only; the pre-revision version
(`paper/main.tex` + `paper/sections/`) is untouched. No experiment was run, no
number changed, no gate loosened. Companion audits:
`reports/final_numerical_audit.md`, `reports/submission_readiness.md`.

| Priority | Section | Original issue | Correction applied | Verification |
|---|---|---|---|---|
| P1 | §7.3 (Table 8 text) | κ_CBE = −0.027 interpreted as "the model's answers … are statistically unrelated" | Replaced with the little-agreement-beyond-the-pooled-marginal-reference formulation + explicit cross-reference to §4's non-independence caveat | Phrase absent from compiled PDF; wording matches §4.2 semantics |
| P1 | Abstract | Effect reported as positive-magnitude cost with magnitude CI "[34.4, 41.8]" | Signed convention: "reduces … by 38.1 percentage points (paired difference −0.381; 95% cluster-bootstrap CI [−0.418, −0.344])" | CI matches `p1_paired_tests.csv` exactly |
| P1 | §7.7 | "costs … 38.1 points … (CI [−0.418, −0.344])" mixed magnitude with signed CI; "already credible at −7.9 points" mixed pp with proportions | Direction defined (Δ_{P3−P0}); signed difference −0.381 with signed CI, reduction stated separately in pp; 7B stated as −0.079 | Audit PASS (34/34 checks) |
| P1 | §8 Table 10 caption | Direction of paired effects implicit | Caption now defines signed differences (condition − canonical; TR − EN) and that negative = reduction | Caption text |
| P1 | §8.2, Conclusion | Same magnitude/sign mixtures | Signed differences with signed CIs in §8.2; conclusion uses "reduces … by X percentage points" consistently | grep sweep clean |
| P1 | §7.1 | "Execution followed the audited protocol without exception" — in tension with unissued gate receipts for the >2,000-call sweep | Qualified: registered pilots followed recorded procedures in full; the sweep is a separately labeled exploratory experiment without claiming completion of prospective confirmatory gates | Phrase absent from PDF |
| P1 | §7.7, §9 | "ordinary McNemar inference is undefined" | "Conventional row-level McNemar inference is not appropriate for these dependent observations; uncertainty … latent-order cluster bootstrapping" (both locations). §4.3's existing wording ("cluster-aware uncertainty rather than an unqualified McNemar p-value") already correct, kept | Phrase absent from PDF |
| P2 | §7.7 | "locates the transition between them" over-identified a scale threshold | "the evaluated scales suggest a transition … toward more discriminative performance at 7B and 14B", with explicit note that four sizes of one quantized family identify no precise or universal threshold | Text |
| P2 | §7.8 | (Review wording) | Verified adequate: equivalence / grammaticality / naturalness / discourse preference already distinguished; persistence after exclusion reported; no unrestricted-equivalence claim. No change needed | Read-through |
| P2 | Figure 6 | 14B P3 effect not highlighted in panel (b) | GPTreview version now uses `fig6_p1_quantized_scaling_annotated.pdf`: annotation Δ_{P3−P0} = −0.381 placed from the retained CSV value by the generator (no hard-coded data); caption states direction, zero-reference, CI, and exploratory/quantized status. Unannotated figure kept for the pre-revision version | Rendered and visually checked; no overlap |
| P2 | Availability statement | "is available at <URL> (public upon publication)" while the repository is private | "is maintained in a version-controlled repository (<URL>) that will be made publicly accessible upon publication" | Matches actual repo state (private) |
| P2 | Front matter | Internal label "Research manuscript draft (GPTreview revision)" | Removed; date only ("30 October 2026"). Title/authors/ORCIDs/keywords/back-matter checked, unchanged | Title page of PDF |
| P3 | §§5–7, App. C | Procedural repetition | Reviewed; the GPTreview revision had already de-duplicated (§7.9 condensed, synthesis moved to §8). No further cuts made — remaining protocol text is load-bearing for integrity claims | Read-through |
| P3 | Figures/tables | Placement, numbering, captions | All floats `[t]`-placed near first reference; numbering and cross-references resolve (0 undefined); Figures 3–4 retained in main text per instruction | Compile log |
| P3 | Language | Terminology sweep | GPT-4o prose vs `gpt-4o-2024-08-06` snapshot id; P0–P5; Q4_K_M; pp vs %; "prompt identities" vs "calls" — consistent | grep sweep |

Compile status: `main_GPTreview.pdf`, 30 pages, 0 undefined references, no
visible-layout warnings attributable to this revision (two pre-existing minor
overfull boxes in unrelated appendix/results paragraphs, ≤29 pt, predate the
revision and do not clip content).
