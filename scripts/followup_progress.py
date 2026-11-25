"""Summarize live progress of the R1 structured-reasoning follow-up run.

Reads the follow-up manifest and the partial (or complete) structured JSONL output
and writes a concise Markdown status report to ``results/followup/PROGRESS.md``.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/followup_reasoning_manifest.jsonl"),
    )
    ap.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("results/followup/qwen2-5-14b_structured.jsonl"),
    )
    ap.add_argument(
        "--progress-md",
        type=Path,
        default=Path("results/followup/PROGRESS.md"),
    )
    ap.add_argument("--status", default="RUNNING")
    args = ap.parse_args()

    total = sum(1 for line in args.manifest.open(encoding="utf-8") if line.strip())
    headers: list[dict] = []
    recs: dict[str, dict] = {}
    if args.output_jsonl.exists():
        for line in args.output_jsonl.open(encoding="utf-8"):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = obj.get("record_kind")
            if kind == "run_header":
                headers.append(obj)
            elif kind == "inference":
                recs[obj["prompt_id"]] = obj

    done = len(recs)
    pct = (100.0 * done / total) if total else 0.0
    valid = sum(1 for r in recs.values() if r.get("parse_valid"))
    invalid = done - valid
    truncated = sum(1 for r in recs.values() if r.get("output_truncated"))
    transport_err = sum(1 for r in recs.values() if r.get("transport_error"))
    walls = [r["wall_seconds"] for r in recs.values() if isinstance(r.get("wall_seconds"), (int, float))]
    out_toks = [r["output_tokens"] for r in recs.values() if isinstance(r.get("output_tokens"), int)]
    avg_wall = (sum(walls) / len(walls)) if walls else 0.0
    avg_tok = (sum(out_toks) / len(out_toks)) if out_toks else 0.0
    remaining = max(0, total - done)

    # Effective throughput from recent record timestamps (accurate under parallel workers)
    ts_list = []
    for r in recs.values():
        ts = r.get("timestamp")
        if isinstance(ts, str):
            try:
                ts_list.append(datetime.datetime.fromisoformat(ts.replace("Z", "+00:00")))
            except ValueError:
                pass
    ts_list.sort()
    eff_sec_per_prompt = avg_wall
    if len(ts_list) >= 10:
        recent = ts_list[-min(100, len(ts_list)):]
        span = (recent[-1] - recent[0]).total_seconds()
        if span > 1.0:
            eff_sec_per_prompt = span / (len(recent) - 1)
    eta_hours = (remaining * eff_sec_per_prompt) / 3600.0 if eff_sec_per_prompt else 0.0

    digest = headers[-1].get("digest", "unknown")[:12] if headers else "unknown"
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "# Revision R1 Follow-Up Progress (Structured Reasoning)",
        "",
        f"- **Updated:** {now}",
        f"- **Status:** {args.status}",
        f"- **Model:** `qwen2.5:14b` (digest: `{digest}`)",
        f"- **Completed:** {done} / {total} ({pct:.1f}%)",
        f"- **Valid / Invalid:** {valid} / {invalid} ({(100.0 * invalid / done) if done else 0.0:.2f}% invalid)",
        f"- **Truncated / Transport errors:** {truncated} / {transport_err}",
        f"- **Avg latency / prompt:** {avg_wall:.2f} s",
        f"- **Avg output tokens / prompt:** {avg_tok:.1f}",
        f"- **Estimated remaining time:** {eta_hours:.2f} hours ({remaining} prompts left)",
        "",
    ]

    if done:
        lines.append("## Interim Accuracy by Condition (completed prompts only)")
        lines.append("")
        lines.append("| Condition | Completed | Valid | Correct | Accuracy (all done) |")
        lines.append("|---|---:|---:|---:|---:|")
        for label, lang, cond in (
            ("EN P0", "en", "canonical"),
            ("TR P0", "tr", "canonical"),
            ("TR P3", "tr", "P3"),
        ):
            sub = [r for r in recs.values() if r["language"] == lang and r["condition"] == cond]
            v = sum(1 for r in sub if r.get("parse_valid"))
            c = sum(1 for r in sub if r.get("correct"))
            acc = (100.0 * c / len(sub)) if sub else 0.0
            lines.append(f"| {label} | {len(sub)} | {v} | {c} | {acc:.1f}% |")
        lines.append("")

    args.progress_md.parent.mkdir(parents=True, exist_ok=True)
    args.progress_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[{now}] {args.status}: {done}/{total} ({pct:.1f}%) | avg {avg_wall:.2f}s/prompt | ETA {eta_hours:.2f}h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
