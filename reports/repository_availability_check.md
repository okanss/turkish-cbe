# Repository availability check — 2026-10-30

- **Repository reference:** `https://github.com/okanss/turkish-cbe` (as cited
  in the manuscript's Data and Code Availability statement; address verified
  correct via `gh repo view`).
- **Verified access status:** repository EXISTS; visibility **PRIVATE**
  (verified via GitHub API this session). It is NOT publicly accessible, and
  no public claim is made in the manuscript.
- **Reviewer-access status:** not yet arranged. The manuscript now states
  (Case C) that access for editors and reviewers during peer review *will be
  arranged at submission* — identified as a submission-preparation action, not
  claimed as already available.
- **Available artifacts (verified in the tracked tree this session):**
  benchmark-generation sources (`src/`), evaluation and statistical analysis
  scripts (`src/analysis/`, `src/evaluation/`, `src/inference/`), frozen
  manifests (`data/manifests/`, `data/complete_grid/manifest.json`,
  `experiment_dataset_manifest.json`), model configuration (`configs/`),
  dependency lockfiles (`requirements-*.lock`, `pyproject.toml`), retained
  experimental outputs (raw/normalized pilot records, all four sweep JSONLs,
  `results/p1/analysis/`, training plans/summaries), review instruments with
  both reviewers' completed annotations and adjudication
  (`validation/`), figure-generation scripts and plot manifests, README and
  REPRODUCIBILITY documentation, and the manuscript sources. All 17
  Appendix-C paths verified present.
- **Exclusions (documented in README/.gitignore, disclosed in the
  manuscript):** bulk regenerable grid materializations (deterministically
  rebuilt and hash-verified against tracked binding manifests), license-gated
  `meta-llama` tokenizer assets (not redistributable; acquisition
  instructions provided), trained adapter weight files (plans/summaries
  tracked).
- **Final manuscript wording:** "…maintained in a version-controlled
  repository (URL). The repository is currently private; access for editors
  and reviewers during peer review will be arranged at submission, and the
  repository will be made publicly accessible upon publication. The
  repository tracks the source code together with the retained experimental
  outputs …; bulk regenerable grid materializations and license-gated
  tokenizer assets are excluded, with deterministic, hash-verified rebuild
  and acquisition instructions in its documentation."
- **Remaining actions requiring author authorization:**
  1. At submission: grant reviewer access (simplest: flip the repository to
     public, or add a reviewer-access mechanism the journal supports).
  2. Optionally mint an archival DOI (e.g., Zenodo) at acceptance.
  No repository visibility change was made in this revision — that is the
  author's decision.
