"""Read immutable inference evidence and restore every planned denominator."""

from __future__ import annotations

from contextlib import contextmanager, closing
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from inference.dataset import GridDataset, fingerprint
from inference.planning import validate_plan
from inference.prompts import make_messages
from inference.runner import task_for, request_for, _normalize
from inference.storage import STORAGE_VERSION
from renderers.turkish import RELATIONS


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def decode(data):
    def nonfinite(value):
        raise ValueError("nonfinite JSON: " + value)
    return json.loads(data, object_pairs_hook=_object, parse_constant=nonfinite)


def _safe(root, relative):
    path = root / relative
    if any(p.is_symlink() for p in (path, *path.parents) if p != root.parent):
        raise ValueError("analysis evidence must not contain symlinks")
    if not path.resolve().is_relative_to(root):
        raise ValueError("evidence escapes project")
    return path


def discover_runs(root):
    root = Path(root).resolve()
    directory = root / "results/raw"
    return sorted(path.parent.name for path in directory.glob("*/manifest.json"))


@contextmanager
def _read_lock(path):
    # Open existing locks only. Analysis never initializes or repairs a run.
    with path.open("rb") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("run is active; analyse a stable checkpoint") from None
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def expected_tasks(root, plan):
    """Materialize frozen planned tasks, including never-started calls."""
    tasks = {}
    with GridDataset(root, plan["dataset_manifest"]) as data:
        if data.select(plan["cohort"]["selection"]) != plan["cohort"]:
            raise ValueError("cohort differs from frozen plan")
        for row in data.iter_rows(plan["cohort"]):
            built = make_messages(row)
            for model in plan["models"]:
                for replicate in range(plan["policy"]["replicates"]):
                    task = task_for(row, model, built, replicate)
                    if task["task_id"] in tasks:
                        raise ValueError("duplicate planned task")
                    encoding = model["tokenizer"]["encoding"]
                    tasks[task["task_id"]] = (task, model,
                        row["model_token_counts"].get(encoding), request_for(task, model, built))
    if len(tasks) != plan["generation_calls"]:
        raise ValueError("planned call denominator mismatch")
    return tasks


def _checkpoint(root, run_id, plan, normalized, evidence_hashes):
    path = _safe(root, "logs/" + run_id + ".sqlite")
    # mode=ro, not RunStore: analysis cannot reconcile or append inference files.
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        if metadata != {"storage_version": STORAGE_VERSION, "run_id": run_id,
                        "plan_sha256": fingerprint(plan)}:
            raise ValueError("checkpoint identity mismatch")
        outbox = list(connection.execute("SELECT * FROM normalized_outbox ORDER BY seq"))
        expected = b"".join((row["record_json"] + "\n").encode() for row in outbox)
        if normalized != expected:
            raise ValueError("normalized file differs from durable outbox; resume inference first")
        records = {}
        for row in outbox:
            data = (row["record_json"] + "\n").encode()
            record = decode(data)
            if (digest(data) != row["record_sha256"] or record.get("task_id") != row["task_id"]
                    or record.get("attempt_id") != row["attempt_id"]
                    or record.get("run_id") != run_id or row["task_id"] in records):
                raise ValueError("normalized record identity/hash mismatch")
            records[row["task_id"]] = record
        tasks = {row["task_id"]: dict(row) for row in connection.execute("SELECT * FROM tasks")}
        attempts = {row["attempt_id"]: dict(row) for row in connection.execute("SELECT * FROM attempts")}
        resolutions = {row["resolution_id"]: dict(row) for row in connection.execute("SELECT * FROM resolutions")}
    evidence_hashes[str(path.relative_to(root))] = digest(path.read_bytes())
    return tasks, attempts, records, resolutions


def load_run(root, run_id, *, allow_synthetic=False):
    """Return planned replicate-zero observations and provenance.

    Live API provenance is an assertion of the recorded runner, not a guarantee
    of independent server attestation. Fixtures are rejected unless explicitly
    enabled by library tests; the public CLI has no fixture override.
    """
    root = Path(root).resolve()
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", run_id):
        raise ValueError("unsafe run ID")
    hashes = {}

    def read(relative):
        path = _safe(root, relative)
        data = path.read_bytes()
        hashes[relative] = digest(data)
        return data

    raw = "results/raw/" + run_id
    with _read_lock(_safe(root, "logs/" + run_id + ".lock")):
        manifest = decode(read(raw + "/manifest.json"))
        plan = manifest["plan"]
        validate_plan(plan, root)
        if (manifest.get("run_id") != run_id or manifest.get("storage_version") != STORAGE_VERSION
                or manifest.get("plan_sha256") != fingerprint(plan)):
            raise ValueError("run manifest identity mismatch")
        execution = decode(read(raw + "/execution.json"))
        fixture = execution.get("execution_kind") == "test_fixture"
        if (execution.get("plan_sha256") != plan["plan_sha256"]
                or execution.get("execution_kind") not in {"live", "test_fixture"}
                or execution.get("contains_model_results") is not (not fixture)):
            raise ValueError("execution evidence mismatch")
        if fixture and not allow_synthetic:
            raise ValueError("synthetic fixture is not a real research result")
        normalized = read("results/normalized/" + run_id + ".jsonl")
        stored, attempts, records, resolutions = _checkpoint(root, run_id, plan, normalized, hashes)
        expected = expected_tasks(root, plan)
        if set(stored) - set(expected) or set(records) - set(expected):
            raise ValueError("unplanned observation/task")
        for task_id, registered in stored.items():
            if canonical(decode(registered["task_json"])) != canonical(expected[task_id][0]):
                raise ValueError("registered task differs from frozen source")
        attempt_records = {}
        for attempt_id, attempt in attempts.items():
            task_id = attempt["task_id"]
            if task_id not in stored or not re.fullmatch(r"call-[a-f0-9]{32}", attempt_id):
                raise ValueError("unplanned or unsafe attempt")
            raw_records = {}
            for kind in ("request", "response", "error"):
                sha = attempt[kind + "_sha256"]
                if sha is not None:
                    data = read(raw + "/attempts/" + attempt_id + "/" + kind + ".json")
                    if digest(data) != sha:
                        raise ValueError("immutable raw hash mismatch")
                    raw_records[kind] = decode(data)
            if canonical(raw_records.get("request")) != canonical(expected[task_id][3]):
                raise ValueError("raw request differs from frozen model/messages/source")
            attempt_records[attempt_id] = raw_records
        for resolution_id, resolution in resolutions.items():
            if not re.fullmatch(r"resolution-[a-f0-9]{32}", resolution_id):
                raise ValueError("unsafe resolution ID")
            data = read(raw + "/resolutions/" + resolution_id + ".json")
            value = decode(data)
            if (digest(data)!=resolution["record_sha256"] or value.get("resolution_id")!=resolution_id
                    or value.get("run_id")!=run_id or value.get("task_id")!=resolution["task_id"]
                    or value.get("attempt_id")!=resolution["attempt_id"]
                    or value.get("decision")!=resolution["decision"]
                    or value.get("task_id") not in stored
                    or value.get("attempt_id") not in attempts):
                raise ValueError("resolution evidence identity/hash mismatch")
        observations = []
        for task_id, (task, model, proxy, _) in sorted(expected.items()):
            record = records.get(task_id)
            parsed = None
            usage = {}
            reason = "missing_response"
            response = None
            if record is not None:
                attempt_id = record["attempt_id"]
                if not re.fullmatch(r"call-[a-f0-9]{32}", attempt_id):
                    raise ValueError("unsafe attempt ID")
                attempt = attempts.get(attempt_id)
                if attempt is None or attempt["task_id"] != task_id:
                    raise ValueError("attempt identity mismatch")
                if (stored[task_id]["status"] not in {"succeeded","permanent_failed","cached"}
                        or stored[task_id]["current_attempt_id"] != attempt_id):
                    raise ValueError("normalized task is not the current terminal attempt")
                raw_records = attempt_records[attempt_id]
                response = raw_records.get("response")
                error = raw_records.get("error")
                abandoned = record.get("error", {}).get("code") == "ambiguous_call_abandoned" if record.get("error") else False
                if abandoned:
                    resolution = resolutions.get(record.get("resolution_id"))
                    if (stored[task_id]["status"] != "permanent_failed" or resolution is None
                            or resolution["decision"]!="abandon" or resolution["task_id"]!=task_id
                            or resolution["attempt_id"]!=attempt_id):
                        raise ValueError("abandoned task is not terminal")
                    reason = "ambiguous_call_abandoned"
                else:
                    rebuilt = _normalize(task, model, response, fixture=fixture, error=error)
                    rebuilt.update(task_id=task_id, attempt_id=attempt_id, run_id=run_id)
                    if canonical(record) != canonical(rebuilt):
                        raise ValueError("normalized result differs from raw/parser/source")
                    if record.get("cache_origin") or record.get("independent_observation") is not True:
                        raise ValueError("cached responses cannot be independent research observations")
                    parsed = rebuilt["parse"]
                    usage = rebuilt["reported_usage"]
                    reason = parsed["invalid_reason"]
            if task["replicate"] != 0:
                continue
            source = task["source"]
            input_tokens = usage.get("input_tokens")
            if input_tokens is not None and (type(input_tokens) is not int or input_tokens < 0):
                raise ValueError("invalid reported input token count")
            observations.append({**source, "cluster_id": source["latent_instance_id"],
                "query_id": source["grid_query_id"], "model_id": model["id"],
                "model_family": model["family"], "model_revision": model["revision"],
                "model_spec_sha256": task["model_spec_sha256"],
                "relation_family": RELATIONS[source["relation_id"]].family,
                "prediction": parsed["prediction"] if parsed and parsed["valid"] else None,
                "response_status": "missing" if record is None else "valid" if parsed and parsed["valid"] else "invalid",
                "response_observed": response is not None,
                "invalid_reason": reason, "input_tokens": input_tokens,
                "source_prompt_tokens": proxy, "token_encoding": model["tokenizer"]["encoding"],
                "task_id": task_id, "run_id": run_id})
        return {"run_id": run_id, "plan": plan, "observations": observations,
                "synthetic": fixture, "input_files_sha256": hashes,
                "ignored_replicates": len(expected) - len(observations)}
