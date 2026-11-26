# Response to Reviewers – *Information* (Manuscript ID: information-4703657, Round 1)

**Manuscript Title:** *Beyond Translation: Evaluating Logical Reasoning Competence in Large Language Models under Turkish Morphosyntactic Variations*  
**Authors:** Müge Oluçoğlu and Okan Bursa

---

## Cover Note to the Academic Editor

Dear Academic Editor,

We thank you and both reviewers for the careful, constructive assessment of our manuscript. We have revised the manuscript to address every methodological, empirical, and editorial concern raised in Round 1. In addition to revising the text and tables throughout, we executed the targeted explicit stepwise-reasoning follow-up requested by Reviewer 2 on a matched 2,025-prompt subset (149 latent orders; `Qwen2.5-14B-Instruct`, identical `Q4_K_M` weights, 4,096-token budget) and audited all denominators, invalid-response rates, and estimands across the entire manuscript.

Where a reviewer raised an empirical limitation that requires new large-scale runs outside the scope of this exploratory study—specifically confirmatory frozen-test evaluation and multi-family / full-precision model sweeps—we did not overclaim resolution; instead, we narrowed the empirical claims consistently across the Abstract, Sections 7.9, 8, and 9, and stated explicitly what remains future work.

We enclose:
1. A **marked revision PDF** (`Beyond_Translation_R1_marked.pdf`) with additions in red, deletions in red strike-through, and moved passages noted in red brackets.
2. A **clean submission PDF** (`Beyond_Translation_R1_clean.pdf`) compiled from the same LaTeX sources (`beyond-translation-mdpi-R1.zip`).
3. Full code, manifests, run records, and analysis outputs in the now-public companion repository: https://github.com/okanss/turkish-cbe (including the new structured-reasoning follow-up on branch `revision-r1-followup`).

Point-by-point responses to both reviewers follow below.

Sincerely,  
Müge Oluçoğlu and Okan Bursa

---

## Responses to Reviewer 1

We thank Reviewer 1 for the positive evaluation of the benchmark design and for pinpointing four concrete issues regarding invalid responses, uncertainty bands, P3 wording, and sample-size reporting.

### Comment 1: Invalid and truncated responses, denominators, and valid-pair coverage across main contrasts
**Summary of comment:** Report condition-specific invalid rates, valid-pair coverage, and both all-planned and jointly valid estimates for each main contrast, and make denominators explicit throughout.

**Response & Revisions:**
- **Equations & definitions (Section 4.3):** Added an explicit definition of the jointly valid contrast estimator alongside the all-planned estimator so both denominators are formalized up front.
- **Paired contrasts (Section 7.6, Table 11):** Added **Table 11**, reporting for every main 14B and 7B contrast (TR−EN canonical native names; P3−P0 across all nine relations and with `temporal_completion` excluded): the planned pair count (N), arm-specific invalid rates (Arm 0 / Arm 1), valid-pair coverage (v̂), the all-planned cluster-bootstrap estimate (all-planned difference [95% CI]), and the jointly valid estimate (jointly valid difference [95% CI]).
  - At 14B, main contrasts have at most one invalid response per contrast (v̂ ≥ 0.999), so all-planned and jointly valid estimates agree to three decimal places (e.g., TR−EN: −0.075; P3−P0: −0.381 over 1,350 planned / 1,349 jointly valid pairs).
  - At 7B, invalid responses are concentrated in Turkish (1.74% of canonical prompts), causing modest differences across estimands (TR−EN: −0.189 all-planned vs. −0.176 jointly valid; P3−P0: −0.076 all-planned vs. −0.079 jointly valid).
- **Single-condition entity-count profile (Section 7.6, Table 12):** Added **Table 12** reporting single-condition (non-paired) response counts and accuracies for 14B across n ∈ {4, 8, 16, 32, 64, 128}, including explicit N_planned, N_valid, invalid %, all-planned accuracy, and valid-only accuracy for both Turkish and English. (Unlike Table 11, which reports query-matched paired contrasts, Table 12 reports pooled single-condition profiles.) This audit revealed that 245 of 14B's 247 invalid responses are output truncations on Turkish prompts at n = 128 (12.02% invalid, N_planned = 2,038, N_valid = 1,793), where valid-only accuracy is 57.8% versus 50.8% all-planned. We corrected all references to "n = 128 dropping to chance" in Sections 7.6, 8.5, 9.1, and Section 10 (Conclusions) accordingly.
- **Consistent estimand labeling (Sections 7.6, 7.7, 8.1, 8.3, 8.4, and Table 15):** Every 7B contrast mentioned in prose and in Table 15 is now explicitly labeled with its estimand (`jointly valid` alongside the `all-planned` counterpart). Table 15's *Validity* row now states accurately that the direction of the language and P3 contrasts is unchanged across estimands, with 14B agreeing to the reported precision and 7B differing modestly.

---

### Comment 2: Wilson intervals in Figure 6(a)
**Summary of comment:** Either replace the Wilson bands in Figure 6(a) with latent-order cluster-bootstrap intervals or clearly remove any inferential interpretation and label them as descriptive only.

**Response & Revisions:**
- We adopted the second option consistently in both the figure and text:
  - **Figure 6(a):** Added an explicit in-panel label (`Shaded: Wilson 95% (descriptive only)`) and expanded the caption to state that Wilson bands are descriptive binomial precision summaries that treat prompts sharing a latent order as independent, therefore understate uncertainty, and carry no inferential interpretation, directing readers to **Table 11** for latent-order cluster-bootstrap intervals of the language contrasts.
  - **Section 4.3:** Reinforced the distinction between descriptive Wilson summaries and dependence-aware cluster-bootstrap inference.

---

### Comment 3: Precision and caution in describing P3 as "truth-preserving"
**Summary of comment:** Align the Abstract and Conclusions with the Discussion and linguistic validation results, qualifying P3 rather than describing it categorically as truth-preserving without noting contested naturalness and the excluded P3 construction.

**Response & Revisions:**
- **Abstract, Table 1, Sections 7.7, 8.3, 8.6, and Section 10 (Conclusions):** Replaced categorical "truth-preserving" / "answer-preserving" phrasing with cautious wording throughout—specifically *"Turkish morphosyntactic variants designed to preserve the gold answer"* in the Abstract and *"designed to preserve truth conditions and judged meaning-preserving in the reviewed sample"* for P3.
- Explicitly noted in the Abstract, Section 7.7, Section 8.3, and Section 10 (Conclusions) that the single construction excluded on grammaticality grounds (`temporal_completion`) was itself a P3 realization, and that P3 naturalness ratings diverged sharply: for P3, the two reviewers' median naturalness ratings were 4 and 2, and only 6 of the 144 P3 items received a naturalness rating of at least 4 from both reviewers, so the P3 deficit may reflect sensitivity to discourse-atypical word order as much as logical inference failure.

---

### Comment 4: Sample-size reporting and the unit of independence in the Abstract
**Summary of comment:** Clarify in the Abstract that 16,618 is the number of prompt identities derived from dependent latent source orders rather than independent observations.

**Response & Revisions:**
- **Abstract and Section 7.6:** Replaced *"16,618 matched prompts per model"* with *"16,618 prompt identities per model, derived from the 300 latent source orders of the pilot split, which are the unit of independence."* Added the corresponding distinction for GPT-4o (*"1512 pilot prompts derived from 32 latent orders"*).

---

## Responses to Reviewer 2

We thank Reviewer 2 for the thorough critique, especially for emphasizing concrete P3 stimulus examples, explicit stepwise reasoning, reviewer transparency, and readability. Note that Reviewer 2's report contains two items numbered `5`; we address them separately below as **5a** (explicit reasoning follow-up) and **5b** (linguistic reviewer details).

### Comment 1: Show concrete P3 stimuli
**Summary of comment:** Readers cannot judge what models were asked because no P3 stimulus is shown.

**Response & Revisions:**
- **Section 3.2 (Table 2):** Added **Table 2**, displaying verbatim prompts drawn directly from the frozen manifest for the same latent order, relation, and gold answer across P0 (canonical), P1, P2, P3, P4, P5, and the excluded `temporal_completion` P3 realization, together with the full system/user prompt structure. Readers can inspect the exact canonical-versus-P3 contrast side by side (e.g., *"Onur Güneş, Meryem Çetin'in önündedir."* vs. *"Meryem Çetin'in önündedir Onur Güneş."*).

---

### Comment 2: All empirical results come from the pilot split (no confirmatory frozen-test run)
**Summary of comment:** Because the frozen-test split was not run, the paper's main empirical claims are not yet established.

**Response & Revisions:**
- **Scope narrowed; confirmatory limitation remains open:** We agree that pilot-split evaluations do not provide independent confirmation. We revised the **Abstract**, **Section 7.9**, **Section 8.7**, **Section 9.1**, and **Section 10 (Conclusions)** to restrict all empirical claims strictly to the evaluated pilot cohorts and to label every result exploratory.
- The preregistered frozen-test evaluation remains pending and is explicitly preserved as future work (Section 9.2). The new structured-reasoning follow-up (Section 7.8) tests prompt-format sensitivity on a matched pilot subset and is likewise reported as exploratory.

---

### Comment 3: Limited model-family diversity and quantization
**Summary of comment:** Only three evaluated configurations discriminate (August 2024 GPT-4o and quantized Qwen2.5-7B/14B); Llama tokenizers were measured without evaluating Llama models, and no full-precision open-weight or reasoning-oriented models are included.

**Response & Revisions:**
- **Scope narrowed; experimental model coverage not expanded:** We did not add new model families or full-precision sweeps in this revision. Instead, we stated this boundary explicitly in **Sections 6.3, 7.9, 8.4, 9.1, and 9.2**: only three evaluated configurations discriminate, Llama tokenizers were measured without evaluating Llama models, and the new Section 7.8 stepwise follow-up uses the same quantized `Qwen2.5-14B-Instruct` (`Q4_K_M`) weights. Generalization to other open-weight families, full-precision weights, and dedicated reasoning-trained models therefore remains untested and is explicitly listed as future work.

---

### Comment 4: Bare EVET/HAYIR (YES/NO) format and 32-token output cap
**Summary of comment:** Restricting output to a bare label within 32 tokens (with 247 unparseable responses at 14B) may measure rapid pattern matching rather than transitive inference.

**Response & Revisions:**
- **Direct-answer audit (Sections 5.2, 7.6, 8.5, 9.1, and Table 12):** We audited all 247 invalid 14B responses and showed that 245 occur on Turkish n = 128 prompts (244 emitting EVET/HAYIR followed by truncated explanation), separating response-budget truncation from accuracy decline (Table 12: 50.8% all-planned vs. 57.8% valid-only at n = 128). We explicitly acknowledge in Sections 5.2, 8.5, and 9.1 that the direct-answer protocol probes immediate verdicts rather than explicit multi-step derivations.
- **Explicit stepwise-reasoning test (Section 7.8, Table 14):** We tested whether the P3 and entity-count patterns persisted under structured prompting with a 4,096-token output budget. Because both the instruction and the budget changed, the experiment cannot isolate their individual effects (details under Comment 5a below).

---

### Comment 5a: Execute at least one explicit reasoning-prompting condition on a subset
**Summary of comment:** Run at least one reasoning-prompting condition on a subset to test whether the P3 and complexity effects persist under explicit reasoning.

**Response & Revisions:**
- **New experiment executed (Section 7.8, Table 14, Section 8.3):** We ran the framework's pre-specified `structured_reasoning` prompt (normalize premises, identify query direction, perform transitive chaining, and end with a bare `FINAL: EVET/HAYIR` or `FINAL: YES/NO` line) on a matched 2,025-prompt pilot subset (675 each of EN P0, TR P0, TR P3 across all nine relations, `native_shared` names, n ∈ {4, 16, 64}, 149 latent orders) using identical `Qwen2.5-14B-Instruct` `Q4_K_M` weights (digest `7cdf5a01`), greedy decoding (`temperature = 0`, `top_k = 1`, `seed = 42`), `num_ctx = 8192`, and `num_predict = 4096` (0 transport failures; code and records in `src/inference/ollama_reasoning_runner.py`, `scripts/analyze_followup.py`, and `results/followup/`).
- **Results and careful interpretation:**
  - **All-planned estimates (Table 14):** Compared to the direct arm (invalid 0.05%; EN P0 92.9%, TR P0 85.3%, TR P3 45.0%; P3−P0 difference = −0.403 [−0.452, −0.355]; TR−EN difference = −0.076 [−0.130, −0.014]), the structured arm yielded 25.43% invalid responses (515/2,025: 139 truncations at 4,096 tokens—123 at n = 64—plus 338 strict final-line format deviations such as `4. FINAL: YES` and 38 duplicate `FINAL:` markers), with all-planned accuracies of EN P0 73.2%, TR P0 60.4%, TR P3 40.0%, P3−P0 difference = −0.204 [−0.245, −0.164], and TR−EN difference = −0.127 [−0.175, −0.076].
  - **Correct interpretation of the narrower all-planned P3 gap:** We state explicitly in Sections 7.8 and 8.3 that the narrower all-planned gap (−0.204 vs. −0.403) does **not** reflect improved P3 accuracy (TR P3 fell from 45.0% to 40.0%); rather, canonical TR P0 fell more steeply (85.3% to 60.4%) under the elevated invalid rate. Among jointly valid pairs, the P3 deficit persisted at **−28.9 percentage points** (391/675 pairs, 57.9% pair coverage; valid-only accuracies: EN P0 97.4%, TR P0 82.8%, TR P3 52.9%), though incomplete pair coverage limits interpretation of its exact magnitude. The jointly valid TR−EN gap likewise persisted (−13.1 pp, 375/675 pairs, 55.6% coverage).
  - **Complexity (n) effect by denominator:** On valid responses alone, accuracy declines monotonically across n ∈ {4, 16, 64} in all three presentations (EN P0: 100.0%, 97.9%, 95.1%; TR P0: 91.7%, 82.6%, 75.2%; TR P3: 66.9%, 50.0%, 43.9%). Under the all-planned estimand, monotonic decline holds only for TR P3 (44.9%, 41.8%, 33.3%), because strict marker-format deviations peak at n = 4 (36.0%–40.4% invalid, zero truncations) before truncations drive a drop from n = 16 to n = 64 in EN P0 and TR P0.

---

### Comment 5b: Linguistic reviewer profile, independence, blinding, and item allocation
**Summary of comment:** Specify whether reviewers are native speakers, their background, their relationship to the authors, whether canonical forms were visible, and how the 360 items were allocated across strata.

**Response & Revisions:**
- **Section 7.7 and Table 13:** Added a dedicated *Reviewers and procedure* paragraph and **Table 13**:
  - **Identity & background:** Stated explicitly that the two reviewers are the two authors (M.O. and O.B.), both native Turkish speakers and computer/software engineering and AI researchers with **no formal academic training in linguistics and no prior professional linguistic-annotation credentials**.
  - **Blinding & procedure:** Stated transparently that while the two authors rated all 360 items independently without viewing each other's ratings, the review took place *after* the model runs (so neither reviewer was blind to the study hypotheses or pilot results), and the review sheet displayed each transformed prompt alongside its canonical counterpart, intended relation, and gold answer, grouped by relation and condition rather than randomized.
  - **Allocation:** Added **Table 13** detailing the exact 360-item stratified allocation (P3 oversampled at 144 items = 16 per relation; canonical, general equivalent, P1, P2, P4, P5 at 36 items each = 4 per relation; size and query-type breakdown reported in caption).
  - **Limitation & future work (Sections 9.1–9.2):** Explicitly acknowledged this lack of independence and added external blind review by native speakers outside the author team to the confirmatory agenda.

---

### Comment 6: Unidentified RQ4 association component and unevaluated RQ5
**Summary of comment:** RQ4's association component is formally unidentified and RQ5 has no evaluation; training runs (two optimizer steps) validate infrastructure only.

**Response & Revisions:**
- **Rescoped explicitly (Introduction, Table 3, Section 9.1, Appendix D):** Updated **Table 3** and the Introduction to distinguish **RQ1–RQ3** (evaluated exploratorily in this article) from framework-level questions: for **RQ4**, only the descriptive lexical/tokenization diagnostics are executed while cross-vocabulary associations remain formally unidentified on a single pilot relation family; for **RQ5**, the new Section 7.8 stepwise-prompting experiment provides an exploratory prompt-intervention probe, whereas the matched base-versus-adapter comparison remains unexecuted. The two-step training infrastructure check was moved out of the main text to **Appendix D**, which states explicitly that no performance claim is drawn from it.

---

### Comment 7: Methodological novelty of R/D/U and κ_CBE
**Summary of comment:** Note clearly that R/D/U corresponds to a paired 2×2 contingency table, κ_CBE is a pooled-marginal Cohen's κ, and constant-responder invariance is analytically expected.

**Response & Revisions:**
- **Sections 4.1, 7.2, and 8.2:** Made the relation between R/D/U and a paired 2×2 contingency table explicit in Section 4.1: R is the proportion of jointly valid pairs answered correctly in both presentations, D is the proportion answered incorrectly in both, and U is the sum of the two discordant cell proportions. Described κ_CBE transparently as a pooled-marginal variant of Cohen's κ, and explicitly framed the constant-responder demonstration as an analytically expected failure mode that illustrates why raw invariance cannot be reported without correctness. Our methodological contribution is framed accordingly as applying these established paired statistics systematically across controlled morphosyntactic transformations with explicit planned vs. jointly valid denominators.

---

### Comment 8: Process/provenance jargon reducing readability
**Summary of comment:** Move execution-gate receipts, run IDs, hash logs, and failed API attempts out of the main narrative and simplify internal project terminology.

**Response & Revisions:**
- **Sections 5.2, 7.1, 7.6, 7.9, Appendix C, and Appendix D:** Removed internal jargon such as *"execution-gate receipt"* from the main text, moved failed API attempts, detailed provenance logs, and a compact terminology glossary to **Appendix C**, and moved the two-step training backend check to **Appendix D**. Rewrote the opening of **Section 7.9** so it leads directly with the three substantive empirical findings and their interpretive boundaries rather than project-infrastructure language.
