"""Local Ollama runner for the P1 supplemental quantized evaluation.

This runner executes the frozen P1 manifest against a local Ollama server on
Apple Silicon. It deliberately reuses the audited protocol pieces — message
construction (``inference.prompts.make_messages``, which re-verifies each
frozen prompt's SHA-256 at run time) and the strict answer parser
(``inference.parser.parse_answer``) — so prompts and answer-label semantics
are byte-identical to the registered pipeline. Everything else about this arm
is OUTSIDE the audited provider set: quantized GGUF weights are not the
registered fp/bf16 snapshots, so results are a separate supplemental arm and
must never be pooled with registered-model results.

Endpoint: ``/api/chat`` (not ``/api/generate``) so the model's own chat
template is applied to the same system/user messages the audited providers
send; the endpoint and all generation options are recorded on every record.

Determinism: temperature 0, top_k 1, fixed seed. Ollama's seed/top_k support
varies by backend; the requested and echoed options are both stored, and the
run never claims determinism beyond what was echoed.

Checkpointing: results append to one JSONL per (manifest, model); on start the
file is scanned and completed prompt IDs are skipped, so interruption and
resume never duplicate a call. Invalid outputs are parsed, flagged, and kept.

Usage:
    PYTHONPATH=src python3 -m inference.ollama_runner \
        --manifest data/manifests/p1_expanded_manifest.jsonl \
        --model qwen2.5:3b --output results/p1/qwen25-3b.jsonl
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from inference.parser import parse_answer
from inference.prompts import make_messages

OLLAMA_URL = "http://localhost:11434"
RUNNER_VERSION = "p1-0.1.0"

METADATA_FIELDS = (
    "prompt_id", "latent_instance_id", "query_instance_id", "grid_query_id",
    "split", "language", "condition", "relation_id", "relation_family", "n",
    "name_control", "query_type", "query_distance", "query_direction",
    "gold_label", "sampling_stratum")


def _utc() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="milliseconds")


def _post(path: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(OLLAMA_URL + path, data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _get(path: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(OLLAMA_URL + path, timeout=timeout) as response:
        return json.loads(response.read())


def model_digest(model: str) -> dict:
    for entry in _get("/api/tags")["models"]:
        if entry["name"] == model:
            return {"digest": entry["digest"],
                    "quantization": entry.get("details", {}).get("quantization_level"),
                    "parameter_size": entry.get("details", {}).get("parameter_size"),
                    "weights_bytes": entry["size"]}
    raise ValueError(f"model {model!r} is not installed in the local Ollama store")


def load_manifest(path: Path) -> list[dict]:
    rows = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        missing = [f for f in METADATA_FIELDS if f not in row]
        if missing or "prompt" not in row or "prompt_sha256" not in row:
            raise ValueError(f"manifest row missing fields: {missing or ['prompt']}")
        if row["prompt_id"] in seen:
            raise ValueError("duplicate prompt_id in manifest: " + str(row["prompt_id"]))
        seen.add(row["prompt_id"])
        rows.append(row)
    if not rows:
        raise ValueError("empty manifest")
    return rows


def completed_ids(output: Path) -> set:
    done = set()
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if record.get("record_kind") == "inference":
                    done.add(record["prompt_id"])
    return done


def run_one(model: str, row: dict, options: dict, timeout: float, retries: int) -> dict:
    built = make_messages(row)  # re-verifies the frozen prompt SHA-256
    payload = {"model": model, "messages": built["messages"], "stream": False,
               "options": options}
    attempts, response, error = 0, None, None
    started = time.monotonic()
    while attempts <= retries:
        attempts += 1
        try:
            response = _post("/api/chat", payload, timeout)
            break
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            time.sleep(min(2 ** attempts, 30))
    wall = time.monotonic() - started
    record = {
        "record_kind": "inference", "runner_version": RUNNER_VERSION,
        "endpoint": "/api/chat", "timestamp": _utc(),
        **{field: row[field] for field in METADATA_FIELDS},
        "prompt_sha256": row["prompt_sha256"],
        "messages_sha256": built["messages_sha256"],
        "prompt_version": built["prompt_version"],
        "model": model, "requested_options": options,
        "attempts": attempts, "transport_error": None if response else error,
        "wall_seconds": round(wall, 4),
    }
    if response is None:
        record.update(raw_response=None, parsed=None, correct=None,
                      parse_valid=False, failure="transport_failed_after_retries")
        return record
    text = response.get("message", {}).get("content", "")
    done_reason = response.get("done_reason")
    truncated = done_reason == "length"
    context_overflow = (type(response.get("prompt_eval_count")) is int
                        and response["prompt_eval_count"] >= options.get("num_ctx", 0) - 8)
    parsed = parse_answer(text, row["language"], truncated=truncated,
                          finish_reason=done_reason)
    prediction = parsed.get("prediction")
    record.update(
        raw_response=response, raw_text=text, parsed=parsed,
        parse_valid=bool(parsed.get("valid")),
        prediction=prediction,
        correct=(prediction == row["gold_label"]) if parsed.get("valid") else False,
        done_reason=done_reason, output_truncated=truncated,
        possible_context_truncation=bool(context_overflow),
        prompt_tokens=response.get("prompt_eval_count"),
        output_tokens=response.get("eval_count"),
        prompt_eval_seconds=(response.get("prompt_eval_duration") or 0) / 1e9,
        generation_seconds=(response.get("eval_duration") or 0) / 1e9,
        total_seconds=(response.get("total_duration") or 0) / 1e9,
        failure=None)
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--num-predict", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after N new calls (benchmark/smoke use)")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--pause-file", type=Path, default=None,
                        help="if this file exists, finish the current call, "
                             "checkpoint, and exit with code 75 (resumable)")
    args = parser.parse_args(argv)

    rows = load_manifest(args.manifest)
    digest = model_digest(args.model)
    done = completed_ids(args.output)
    pending = [row for row in rows if row["prompt_id"] not in done]
    if args.limit is not None:
        pending = pending[: args.limit]
    manifest_sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    options = {"temperature": 0, "top_k": 1, "seed": args.seed,
               "num_ctx": args.num_ctx, "num_predict": args.num_predict}
    print(json.dumps({"model": args.model, **digest, "manifest_rows": len(rows),
                      "already_completed": len(done), "to_run": len(pending),
                      "options": options}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    header = {"record_kind": "run_header", "runner_version": RUNNER_VERSION,
              "timestamp": _utc(), "model": args.model, **digest,
              "manifest": str(args.manifest), "manifest_sha256": manifest_sha,
              "requested_options": options,
              "arm_label": "supplemental_quantized_local_evaluation; not the audited registered-provider pipeline"}
    with args.output.open("a", encoding="utf-8") as sink:
        sink.write(json.dumps(header, ensure_ascii=False) + "\n")
        sink.flush()
        started = time.monotonic()
        for index, row in enumerate(pending, 1):
            if args.pause_file is not None and args.pause_file.exists():
                print(f"[{_utc()}] pause requested; {index - 1} new calls written, "
                      "checkpoint is consistent — exiting 75", flush=True)
                return 75
            record = run_one(args.model, row, options, args.timeout, args.retries)
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            if index % args.progress_every == 0 or index == len(pending):
                rate = index / (time.monotonic() - started)
                remaining = (len(pending) - index) / rate if rate else float("inf")
                print(f"[{_utc()}] {args.model}: {index}/{len(pending)} "
                      f"({rate:.2f} calls/s, ~{remaining/60:.1f} min left)",
                      flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
