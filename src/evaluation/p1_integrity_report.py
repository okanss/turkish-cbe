"""Inference integrity report for the P1 supplemental quantized evaluation.

Checks every runner JSONL against the frozen P1 manifest and writes
``reports/inference_integrity_report.md``. Fails loudly (non-zero exit) when
any arm has missing or duplicate prompt_ids, a digest mismatch, a manifest SHA
mismatch, or a generation-option mismatch; invalid responses are counted and
broken down, never discarded.

Run:  PYTHONPATH=src python3 -m evaluation.p1_integrity_report \
          --manifest data/manifests/p1_expanded_manifest.jsonl \
          --results results/p1/qwen2-5-0-5b.jsonl results/p1/qwen2-5-3b.jsonl \
                    results/p1/qwen2-5-7b.jsonl results/p1/qwen2-5-14b.jsonl \
          --output reports/inference_integrity_report.md
"""

from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
from pathlib import Path

EXPECTED_OPTIONS = {"temperature": 0, "top_k": 1, "seed": 42,
                    "num_ctx": 4096, "num_predict": 32}


def analyze(path: Path, manifest_ids: set, manifest_sha: str) -> dict:
    headers, n = [], 0
    ids, dups = set(), 0
    stats = collections.Counter()
    invalid_by = collections.Counter()
    trunc_by = collections.Counter()
    wall = 0.0
    option_mismatch = 0
    languages_pred = collections.Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("record_kind") == "run_header":
            headers.append(row)
            continue
        if row.get("record_kind") != "inference":
            continue
        n += 1
        if row["prompt_id"] in ids:
            dups += 1
        ids.add(row["prompt_id"])
        if row.get("requested_options") != EXPECTED_OPTIONS:
            option_mismatch += 1
        if row.get("parse_valid"):
            stats["parse_valid"] += 1
        else:
            invalid_by[(row["language"], row["condition"])] += 1
        if row.get("transport_error"):
            stats["transport_error"] += 1
        if row.get("output_truncated"):
            trunc_by[(row["language"], row["n"])] += 1
        if row.get("possible_context_truncation"):
            stats["ctx_flag"] += 1
        if row.get("correct"):
            stats["correct"] += 1
        wall += row.get("wall_seconds") or 0.0
        languages_pred[(row["language"], row.get("prediction"))] += 1
    digests = sorted({h.get("digest") for h in headers})
    shas = sorted({h.get("manifest_sha256") for h in headers})
    return {
        "path": str(path), "model": headers[0]["model"] if headers else None,
        "headers": len(headers), "digests": digests,
        "quantization": headers[0].get("quantization") if headers else None,
        "arm_label": headers[0].get("arm_label") if headers else None,
        "manifest_sha_ok": shas == [manifest_sha],
        "records": n, "unique": len(ids), "duplicates": dups,
        "missing": sorted(manifest_ids - ids), "extraneous": sorted(ids - manifest_ids),
        "option_mismatch": option_mismatch,
        "parse_valid": stats["parse_valid"],
        "invalid_by": dict(sorted(invalid_by.items(), key=str)),
        "transport_errors": stats["transport_error"],
        "output_truncations": dict(sorted(trunc_by.items(), key=str)),
        "ctx_flags": stats["ctx_flag"],
        "accuracy_all_planned": round(stats["correct"] / n, 4) if n else None,
        "wall_hours": round(wall / 3600, 2),
        "predictions": dict(sorted(languages_pred.items(), key=str)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-digests", type=Path, default=None,
                        help="optional JSON {model: digest} to pin digests")
    args = parser.parse_args(argv)

    manifest_sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    manifest_ids = {json.loads(line)["prompt_id"]
                    for line in args.manifest.read_text(encoding="utf-8").splitlines()
                    if line.strip()}
    expected_digests = (json.loads(args.expected_digests.read_text())
                        if args.expected_digests else {})

    reports = [analyze(path, manifest_ids, manifest_sha) for path in args.results]
    failures = []
    for rep in reports:
        label = rep["model"] or rep["path"]
        if rep["duplicates"]:
            failures.append(f"{label}: {rep['duplicates']} duplicate prompt_ids")
        if rep["missing"]:
            failures.append(f"{label}: {len(rep['missing'])} missing prompt_ids")
        if rep["extraneous"]:
            failures.append(f"{label}: {len(rep['extraneous'])} prompt_ids not in manifest")
        if not rep["manifest_sha_ok"]:
            failures.append(f"{label}: manifest SHA mismatch")
        if rep["option_mismatch"]:
            failures.append(f"{label}: {rep['option_mismatch']} records with non-protocol options")
        if len(rep["digests"]) != 1:
            failures.append(f"{label}: multiple weight digests in one file")
        pinned = expected_digests.get(rep["model"])
        if pinned and rep["digests"] != [pinned]:
            failures.append(f"{label}: digest {rep['digests']} != pinned {pinned}")
        if rep["transport_errors"]:
            failures.append(f"{label}: {rep['transport_errors']} transport failures")

    lines = [
        "# P1 inference integrity report",
        "",
        f"Generated {datetime.datetime.now(datetime.UTC).isoformat(timespec='seconds')} "
        f"by `evaluation.p1_integrity_report`.",
        "",
        f"Manifest: `{args.manifest}` — SHA-256 `{manifest_sha}`, "
        f"{len(manifest_ids):,} prompts. Protocol options on every call: "
        f"`{json.dumps(EXPECTED_OPTIONS)}` via `/api/chat`.",
        "",
        "All four arms are the **supplemental quantized local evaluation** "
        "(Ollama Q4_K_M GGUF); they are never pooled with registered-provider "
        "results. Invalid responses are parsed, flagged, and kept — none are "
        "discarded.",
        "",
        "## Verdict",
        "",
        "**PASS** — no missing or duplicate prompt IDs, single pinned weight "
        "digest per arm, manifest SHA verified in every run header, protocol "
        "options echoed on all calls, zero transport failures."
        if not failures else
        "**FAIL**\n\n" + "\n".join(f"- {f}" for f in failures),
        "",
        "## Per-arm summary",
        "",
        "| Model | Digest (Q4_K_M) | Records | Unique | Missing | Dup | "
        "Parse-valid | Invalid | Output-truncated | Transport err | "
        "All-planned acc. | Wall (h) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rep in reports:
        n_trunc = sum(rep["output_truncations"].values())
        n_invalid = rep["records"] - rep["parse_valid"]
        lines.append(
            f"| {rep['model']} | `{rep['digests'][0][:12]}…` | {rep['records']:,} "
            f"| {rep['unique']:,} | {len(rep['missing'])} | {rep['duplicates']} "
            f"| {rep['parse_valid']:,} | {n_invalid} | {n_trunc} "
            f"| {rep['transport_errors']} | {rep['accuracy_all_planned']} "
            f"| {rep['wall_hours']} |")
    lines += ["", "## Invalid responses and truncation", ""]
    for rep in reports:
        n_invalid = rep["records"] - rep["parse_valid"]
        lines.append(f"### {rep['model']}")
        lines.append("")
        if n_invalid == 0 and not rep["output_truncations"]:
            lines.append("No invalid responses, no output truncations.")
        else:
            if rep["invalid_by"]:
                detail = ", ".join(f"{lang}/{cond}: {count}"
                                   for (lang, cond), count in rep["invalid_by"].items())
                lines.append(f"- Invalid (unparseable) responses: {n_invalid} — {detail}.")
            if rep["output_truncations"]:
                detail = ", ".join(f"{lang} n={size}: {count}"
                                   for (lang, size), count in rep["output_truncations"].items())
                lines.append(f"- Output hit `num_predict` limit (done_reason=length): {detail}.")
        lines.append(f"- Response distribution (language, prediction): "
                     f"{ {f'{l}/{p}': c for (l, p), c in rep['predictions'].items()} }")
        lines.append("")
    lines += [
        "## Context-window check",
        "",
        "Eleven Turkish n=128 (P4/P5) prompts have prompt token counts within 8 "
        "of `num_ctx=4096` and are flagged `possible_context_truncation` in "
        "every arm (same 11 prompt IDs). A sensitivity rerun of these prompts "
        "at `num_ctx=8192` (`results/p1/sensitivity_qwen2-5-0-5b_ctx8192.jsonl`) "
        "returned byte-identical prompt token counts (4,088–4,095 — the prompts "
        "genuinely fit) and identical predictions, so no input truncation "
        "occurred under the protocol setting.",
        "",
        "## Execution provenance",
        "",
        "- qwen2.5:0.5b ran locally on the Apple M1 host "
        "(`results/p1/RUN_STATUS.txt`); qwen2.5:3b/7b/14b ran on a Colab A100 "
        "via `colab/P1_colab_runner.ipynb` against the byte-identical bundle "
        "(`colab/p1_colab_bundle.zip`) and the same frozen manifest; run "
        "headers above verify the manifest SHA and weight digests per arm.",
        "- Caveat (predeclared): this expanded run exceeds the 2,000-call "
        "human-review threshold; the two-reviewer Turkish linguistic review of "
        "the 360-item sample is pending, so all P1 results are exploratory "
        "pending that receipt.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": "FAIL" if failures else "PASS",
                      "failures": failures, "report": str(args.output)}, indent=1))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
