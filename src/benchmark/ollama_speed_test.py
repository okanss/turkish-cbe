"""Phase 6 — M1/Ollama speed benchmark over representative P1 prompts.

Runs ~100 manifest prompts per model (stratified over entity count, language
and condition so short, medium and long prompts are all represented), via the
same runner code path the real evaluation uses, then estimates the full-run
time from measured throughput. Memory is sampled before/after each model.

Run:  PYTHONPATH=src python3 -m benchmark.ollama_speed_test \
          --manifest data/manifests/p1_expanded_manifest.jsonl \
          --models qwen2.5:0.5b qwen2.5:3b qwen2.5:7b \
          --output reports/m1_ollama_benchmark.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import re
import statistics
import subprocess
import time
from pathlib import Path

from inference.ollama_runner import load_manifest, model_digest, run_one

SEED = 20261028
PER_MODEL = 100


def _memory() -> dict:
    raw = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", raw).group(1))
    values = {}
    for line in raw.splitlines()[1:]:
        if ":" in line:
            key, _, count = line.rpartition(":")
            values[key.strip()] = int(count.strip().rstrip(".")) * page
    swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    used = re.search(r"used = ([\d.]+)M", swap)
    return {"free_bytes": values.get("Pages free", 0),
            "swap_used_bytes": float(used.group(1)) * 2**20 if used else None}


def representative(rows: list[dict], count: int) -> list[dict]:
    rng = random.Random(SEED)
    by_cell = collections.defaultdict(list)
    for row in rows:
        by_cell[(row["n"], row["language"], row["condition"] == "canonical")].append(row)
    chosen: list[dict] = []
    cells = sorted(by_cell)
    while len(chosen) < count:
        for cell in cells:
            if by_cell[cell] and len(chosen) < count:
                chosen.append(by_cell[cell].pop(rng.randrange(len(by_cell[cell]))))
    return chosen


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-model", type=int, default=PER_MODEL)
    parser.add_argument("--num-ctx", type=int, default=4096)
    args = parser.parse_args(argv)
    rows = load_manifest(args.manifest)
    sample = representative(rows, args.per_model)
    options = {"temperature": 0, "top_k": 1, "seed": 42,
               "num_ctx": args.num_ctx, "num_predict": 32}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "digest", "quantization", "prompt_id", "n", "language",
              "condition", "prompt_tokens", "output_tokens",
              "prompt_tokens_per_second", "output_tokens_per_second",
              "total_seconds", "load_seconds", "parse_valid", "done_reason",
              "possible_context_truncation", "free_bytes_before",
              "free_bytes_after", "swap_used_after"]
    summary = {}
    with args.output.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=fields)
        writer.writeheader()
        for model in args.models:
            digest = model_digest(model)
            before = _memory()
            model_rows, load_seconds = [], None
            started = time.monotonic()
            for index, row in enumerate(sample):
                record = run_one(model, row, options, timeout=600, retries=1)
                if record.get("failure"):
                    writer.writerow({"model": model, "digest": digest["digest"],
                                     "prompt_id": row["prompt_id"],
                                     "done_reason": record.get("transport_error")})
                    continue
                if index == 0:
                    load_seconds = record["raw_response"].get("load_duration", 0) / 1e9
                after = _memory()
                pes = record["prompt_eval_seconds"] or 1e-9
                ges = record["generation_seconds"] or 1e-9
                entry = {
                    "model": model, "digest": digest["digest"],
                    "quantization": digest["quantization"],
                    "prompt_id": row["prompt_id"], "n": row["n"],
                    "language": row["language"], "condition": row["condition"],
                    "prompt_tokens": record["prompt_tokens"],
                    "output_tokens": record["output_tokens"],
                    "prompt_tokens_per_second": round((record["prompt_tokens"] or 0) / pes, 2),
                    "output_tokens_per_second": round((record["output_tokens"] or 0) / ges, 2),
                    "total_seconds": round(record["total_seconds"], 4),
                    "load_seconds": round(load_seconds or 0, 3),
                    "parse_valid": record["parse_valid"],
                    "done_reason": record["done_reason"],
                    "possible_context_truncation": record["possible_context_truncation"],
                    "free_bytes_before": before["free_bytes"],
                    "free_bytes_after": after["free_bytes"],
                    "swap_used_after": after["swap_used_bytes"],
                }
                writer.writerow(entry)
                model_rows.append(entry)
                sink.flush()
            wall = time.monotonic() - started
            seconds = [r["total_seconds"] for r in model_rows]
            if seconds:
                summary[model] = {
                    "measured_calls": len(model_rows),
                    "wall_seconds_for_sample": round(wall, 1),
                    "median_call_seconds": round(statistics.median(seconds), 3),
                    "p90_call_seconds": round(sorted(seconds)[int(0.9 * len(seconds)) - 1], 3),
                    "mean_call_seconds": round(statistics.fmean(seconds), 3),
                    "parse_valid_rate": round(statistics.fmean(r["parse_valid"] for r in model_rows), 4),
                    "context_truncations": sum(r["possible_context_truncation"] for r in model_rows),
                    "median_prompt_tps": round(statistics.median(
                        r["prompt_tokens_per_second"] for r in model_rows), 1),
                    "median_output_tps": round(statistics.median(
                        r["output_tokens_per_second"] for r in model_rows), 1),
                    "load_seconds": model_rows[0]["load_seconds"],
                    "min_free_bytes_seen": min(r["free_bytes_after"] for r in model_rows),
                    "max_swap_used_bytes": max(r["swap_used_after"] or 0 for r in model_rows),
                }
    estimates = {}
    total = len(rows)
    for model, stats in summary.items():
        per_call = stats["wall_seconds_for_sample"] / stats["measured_calls"]
        estimates[model] = {
            "planned_calls": total,
            "optimistic_hours": round(total * stats["median_call_seconds"] / 3600, 2),
            "median_hours": round(total * per_call / 3600, 2),
            "conservative_hours": round(total * stats["p90_call_seconds"] / 3600, 2),
        }
    out = {"summary": summary, "estimates": estimates, "planned_calls": total}
    print(json.dumps(out, indent=1))
    args.output.with_suffix(".summary.json").write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
