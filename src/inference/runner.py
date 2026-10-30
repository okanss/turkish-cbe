"""Explicit execution, immutable evidence and conservative crash recovery."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import uuid

from .dataset import GridDataset, fingerprint
from .planning import make_plan, validate_plan, price_usage
from .prompts import make_messages
from .parser import parse_answer
from .providers import create_provider, ProviderError, _http_transport
from .runtime import RateLimiter, ResponseCache
from .storage import RunStore, _publish, IntegrityError


class ExecutionBlocked(ValueError):
    pass


def _immutable(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _publish(path, (json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False)+"\n").encode())


def _source(row):
    keys = ("prompt_id", "latent_instance_id", "query_instance_id", "grid_query_id", "grid_design_id",
            "split", "n", "query_type", "query_distance", "query_direction", "query_relation",
            "query_subject", "query_object",
            "name_control", "relation_id", "language", "condition", "gold_label",
            "presentation_plan_sha256", "human_review_status", "eligible_for_confirmatory_evaluation")
    return {key: row[key] for key in keys}


def task_for(row, model, built, replicate=0):
    key = fingerprint({"model": model, "messages_sha256": built["messages_sha256"],
                       "prompt_id": row["prompt_id"], "replicate": replicate})
    return {"task_id": "call-"+key, "request_key": key, "model_id": model["id"],
            "model_spec_sha256": fingerprint(model), "replicate": replicate,
            "messages_sha256": built["messages_sha256"], "source_prompt_sha256": row["prompt_sha256"],
            "source": _source(row)}


def request_for(task, model, built):
    return {"task_id": task["task_id"], "request_key": task["request_key"],
            "model": deepcopy(model), "messages": built["messages"],
            "messages_sha256": built["messages_sha256"], "prompt_version": built["prompt_version"],
            "source_prompt_sha256": built["source_prompt_sha256"], "source": task["source"],
            "replicate": task["replicate"]}


def _error(error):
    return {"code": error.code, "retryable": error.retryable, "ambiguous": error.ambiguous,
            "status_code": error.status_code, "retry_after_seconds": error.retry_after_seconds,
            "raw_response": error.raw_response}


def _normalize(task, model, response, *, fixture, error=None):
    source = task["source"]
    parsed = parse_answer(response.get("text") if response else None, source["language"],
                          truncated=bool(response and response.get("truncated")),
                          refusal=bool(response and response.get("refusal")),
                          finish_reason=response.get("finish_reason") if response else None)
    if error is not None:
        parsed.update(valid=False, prediction=None, invalid_reason="provider_error")
    cached = response.get("cache_origin") if response else None
    usage = response.get("usage", {}) if response else {}
    return {"schema_version": 1, "execution_kind": "test_fixture" if fixture else "live",
            "contains_model_results": not fixture, "model_id": model["id"],
            "model_spec_sha256": fingerprint(model), "source": source,
            "messages_sha256": task["messages_sha256"], "source_prompt_sha256": task["source_prompt_sha256"],
            "replicate": task["replicate"], "parse": parsed,
            "correct": parsed["valid"] and parsed["prediction"] == source["gold_label"],
            "provider_model": response.get("provider_model") if response else None,
            "resolved_revision": response.get("resolved_revision") if response else None,
            "response_id": response.get("response_id") if response else None,
            "finish_reason": response.get("finish_reason") if response else None,
            "reported_usage": usage, "newly_billed_usage": None if cached else usage,
            "cost": {"amount": 0, "currency": model["pricing"]["currency"],
                     "status": "cache_reuse_no_new_generation"} if cached else price_usage(model, usage),
            "cache_origin": cached, "independent_observation": not bool(cached),
            "error": error}


def _selection_signature(plan):
    selection = deepcopy(plan["cohort"]["selection"])
    for key in ("split", "sizes", "groups_per_size"):
        selection.pop(key)
    return fingerprint({"selection": selection, "models": plan["models"],
                        "prompt_version": plan["prompt_version"], "parser_version": plan["parser_version"],
                        "replicates": plan["policy"]["replicates"],
                        "source_files_sha256": plan["source_files_sha256"]})


def _review_evidence(root, plan, review):
    from inference.linguistic_review import validate_review
    try:
        return validate_review(root, plan["dataset_manifest"], review, [plan["cohort"]["selection"]])
    except (ValueError, OSError) as error:
        raise ExecutionBlocked(str(error)) from error


def approve_pilot(root, run_id, output, *, approved_by, linguistic_review):
    """Create a user-supplied approval receipt, never manufacture a pilot result."""
    root = Path(root).resolve()
    manifest_path = root / "results/raw" / run_id / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    plan = manifest["plan"]
    review_path = Path(linguistic_review)
    review = json.loads(review_path.read_bytes())
    if not isinstance(approved_by, str) or not approved_by.strip():
        raise ExecutionBlocked("an identified human approver is required")
    if review.get("status") != "approved" or not review.get("reviewers"):
        raise ExecutionBlocked("an actual approved linguistic review artifact is required")
    if review.get("dataset_manifest_sha256") != plan["cohort"]["dataset_manifest_sha256"]:
        raise ExecutionBlocked("linguistic review does not bind this frozen dataset")
    _review_evidence(root, plan, review_path)
    with RunStore(root, run_id, plan) as store:
        summary = store.summary()
        rows = [json.loads(line) for line in store.normalized_path.read_text().splitlines()]
        if (plan["stage"] != "pilot" or len(rows) != plan["generation_calls"]
                or any(r.get("execution_kind") != "live" or r.get("cache_origin") or r.get("error") for r in rows)
                or summary["statuses"].get("succeeded", 0) != plan["generation_calls"]):
            raise ExecutionBlocked("approval requires a complete real, uncached pilot without provider failures")
        normalized_sha = hashlib.sha256(store.normalized_path.read_bytes()).hexdigest()
    receipt = {"schema_version": 1, "artifact_type": "pilot_approval", "scope": "full_evaluation",
               "approved_by": approved_by.strip(), "approved_at": datetime.now(timezone.utc).isoformat(),
               "pilot_run_id": run_id, "pilot_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
               "pilot_plan_sha256": plan["plan_sha256"], "pilot_normalized_sha256": normalized_sha,
               "selection_signature": _selection_signature(plan),
               "dataset_manifest_sha256": plan["cohort"]["dataset_manifest_sha256"],
               "linguistic_review": str(review_path.resolve()),
               "linguistic_review_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
               "invalid_answers": sum(not r["parse"]["valid"] for r in rows),
               "wrong_language_answers": sum(r["parse"]["wrong_language"] for r in rows)}
    _immutable(output, receipt)
    return receipt


def _check_approval(root, plan, approval):
    if approval is None:
        raise ExecutionBlocked("full evaluation needs a completed real pilot and explicit approval receipt")
    receipt = json.loads(Path(approval).read_bytes())
    if (receipt.get("artifact_type") != "pilot_approval" or receipt.get("scope") != "full_evaluation"
            or not receipt.get("approved_by") or receipt.get("selection_signature") != _selection_signature(plan)
            or receipt.get("dataset_manifest_sha256") != plan["cohort"]["dataset_manifest_sha256"]):
        raise ExecutionBlocked("pilot approval scope/model/prompt/dataset mismatch")
    run_id = receipt["pilot_run_id"]
    if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id:
        raise ExecutionBlocked("invalid approved pilot ID")
    original = root / "results/raw" / run_id / "manifest.json"
    normalized = root / "results/normalized" / (run_id+".jsonl")
    review = Path(receipt["linguistic_review"])
    for path, key in ((original, "pilot_manifest_sha256"), (normalized, "pilot_normalized_sha256"),
                      (review, "linguistic_review_sha256")):
        if hashlib.sha256(path.read_bytes()).hexdigest() != receipt.get(key):
            raise ExecutionBlocked("approved evidence changed")
    pilot = json.loads(original.read_bytes())["plan"]
    if (pilot.get("plan_sha256") != receipt.get("pilot_plan_sha256") or pilot.get("stage") != "pilot"
            or _selection_signature(pilot) != receipt["selection_signature"]):
        raise ExecutionBlocked("approved pilot plan changed")
    review_value = json.loads(review.read_bytes())
    if (review_value.get("status") != "approved" or not review_value.get("reviewers")
            or review_value.get("dataset_manifest_sha256") != receipt["dataset_manifest_sha256"]):
        raise ExecutionBlocked("linguistic review is not approved for this dataset")
    _review_evidence(root, plan, review)
    evidence = json.loads((original.parent / "execution.json").read_bytes())
    if evidence.get("execution_kind") != "live" or evidence.get("contains_model_results") is not True:
        raise ExecutionBlocked("fixtures cannot authorize full evaluation")
    with RunStore(root, run_id, pilot) as store:
        rows = [json.loads(line) for line in store.normalized_path.read_text().splitlines()]
        if (len(rows) != pilot["generation_calls"] or not rows
                or any(r.get("execution_kind") != "live" or r.get("cache_origin") or r.get("error") for r in rows)
                or store.summary()["statuses"].get("succeeded", 0) != pilot["generation_calls"]):
            raise ExecutionBlocked("approved pilot no longer complete or contains fixtures")


def _http_factory(spec, store, limiter):
    def transport(method, url, payload, headers, timeout):
        identifier = str(uuid.uuid4())
        directory = store.raw_dir / "http" / identifier
        _immutable(directory / "request.json", {"http_call_id": identifier, "method": method,
                   "url": url, "payload": payload, "credential_values_recorded": False})
        limiter.reserve(requests=1)
        try:
            status, response, response_headers = _http_transport(method, url, payload, headers, timeout)
        except ProviderError as error:
            _immutable(directory / "error.json", _error(error))
            raise
        _immutable(directory / "response.json", {"status": status, "raw": response,
                   "request_id": next((v for k,v in response_headers.items() if k.lower()=="x-request-id"), None)})
        return status, response, response_headers
    return create_provider(spec, transport=transport if spec["provider"] != "transformers" else None)


def run_plan(root, plan, run_id=None, *, execute=False, approval=None, linguistic_review=None,
             provider_factory=None, test_fixture=False, sleep=time.sleep, clock=time.time):
    root = Path(root).resolve()
    validate_plan(plan, root)
    # Never trust a re-hashed hand-edited count or cost to authorize execution.
    fresh = make_plan(root, plan["dataset_manifest"], plan["models"], plan["stage"],
                      plan["cohort"]["selection"], plan["policy"])
    if fresh != plan:
        raise ExecutionBlocked("plan/cohort/budgets/environment changed; create a new reviewed plan")
    if not execute:
        return {"dry_run": True, "plan_sha256": plan["plan_sha256"],
                "planned_generation_calls": plan["generation_calls"], "actual_generation_calls": 0,
                "contains_model_results": False, "execution_blockers": plan["execution_blockers"]}
    if provider_factory is not None and not test_fixture:
        raise ExecutionBlocked("injected providers are test fixtures, not live experiment evidence")
    if test_fixture and provider_factory is None:
        raise ExecutionBlocked("fixtures must never contact real providers")
    blockers = set(plan["execution_blockers"]) - {"possible_context_overflow_requires_exact_preflight",
                                                "completed_real_pilot_and_explicit_approval_required"}
    if blockers:
        raise ExecutionBlocked(", ".join(sorted(blockers)))
    if plan["stage"] == "full":
        _check_approval(root, plan, approval)
    # The call count, not the command name, defines a large pilot. Test
    # providers cannot create live evidence and retain their fixture route.
    from inference.linguistic_review import LARGE_EVALUATION_CALLS
    review_evidence = None
    if not test_fixture and plan["stage"] != "full" and plan["generation_calls"] > LARGE_EVALUATION_CALLS:
        review_evidence = _review_evidence(root, plan, linguistic_review)
    if run_id is None:
        run_id = ("fixture-" if test_fixture else "") + plan["stage"]+"-"+plan["plan_sha256"][:20]
    if (not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id
            or run_id in {".", ".."}):
        raise ValueError("run_id must be one safe path component")
    if test_fixture and not run_id.startswith("fixture-"):
        raise ExecutionBlocked("fixture run IDs must use the fixture- prefix")
    models = {m["id"]: m for m in plan["models"]}
    providers, limiters = {}, {}
    cache = ResponseCache(root / "results/cache" / ("fixtures" if test_fixture else "live")) if plan["policy"]["cache"] else None
    actual_calls, cache_hits, recovered = 0, 0, 0
    aborted = None
    with RunStore(root, run_id, plan) as store, GridDataset(root, plan["dataset_manifest"]) as data:
        evidence_path = store.raw_dir / "execution.json"
        evidence = {"execution_kind": "test_fixture" if test_fixture else "live",
                    "contains_model_results": not test_fixture, "plan_sha256": plan["plan_sha256"],
                    "large_pilot_linguistic_review": review_evidence,
                    "pilot_approval_sha256": hashlib.sha256(Path(approval).read_bytes()).hexdigest()
                        if plan["stage"] == "full" else None}
        if plan["stage"] == "full":
            _immutable(store.raw_dir / "pilot_approval.json", json.loads(Path(approval).read_bytes()))
        if evidence_path.exists():
            if json.loads(evidence_path.read_bytes()) != evidence:
                raise ExecutionBlocked("execution kind cannot change on resume")
        else:
            _immutable(evidence_path, evidence)
        if data.select(plan["cohort"]["selection"]) != plan["cohort"]:
            raise ExecutionBlocked("cohort changed")
        for row in data.iter_rows(plan["cohort"]):
            built = make_messages(row)
            for model in models.values():
                for replicate in range(plan["policy"]["replicates"]):
                    task = task_for(row, model, built, replicate)
                    store.register_tasks([task])
                    current = store.get_task(task["task_id"])
                    if current["status"] in {"succeeded", "cached", "permanent_failed"}:
                        continue
                    if current["recoverable"]:
                        response = store.get_response(current["attempt_id"])
                        store.finish(task["task_id"], current["attempt_id"],
                                     _normalize(task, model, response, fixture=test_fixture),
                                     "cached" if response.get("cache_origin") else "succeeded")
                        recovered += 1
                        continue
                    if current["status"] in {"inflight", "ambiguous"}:
                        aborted = "ambiguous_call_requires_explicit_resolution"
                        break
                    request = request_for(task, model, built)
                    while True:
                        current = store.get_task(task["task_id"])
                        if len(current["attempts"]) >= plan["policy"]["max_attempts"]:
                            error = store.get_error(current["attempt_id"])
                            store.finish(task["task_id"], current["attempt_id"],
                                         _normalize(task, model, None, fixture=test_fixture, error=error), "permanent_failed")
                            aborted = "maximum_attempts_exhausted"
                            break
                        attempt = store.claim(task["task_id"], request)
                        key = cache.key(model, built["messages_sha256"]) if cache else None
                        cached = cache.get(key) if cache else None
                        if cached:
                            response = deepcopy(cached["response"])
                            response["cache_origin"] = cached["origin"]
                            store.write_response(attempt, response)
                            store.finish(task["task_id"], attempt,
                                         _normalize(task, model, response, fixture=test_fixture), "cached")
                            cache_hits += 1
                            break
                        try:
                            identity = model["id"]
                            if identity not in providers:
                                endpoint_key = fingerprint({"provider": model["provider"],
                                                           "endpoint": model["serving"]["base_url"],
                                                           "api_key_env": model["serving"]["api_key_env"]})[:24]
                                limiter = RateLimiter(root / "logs" / ("rate-"+endpoint_key+".sqlite"),
                                                      plan["policy"], clock=clock, sleep=sleep)
                                limiters[identity] = limiter
                                provider = provider_factory(model) if provider_factory else _http_factory(model, store, limiter)
                                runtime = provider.preflight()
                                providers[identity] = provider
                                _immutable(store.raw_dir / "preflight" / (identity+"-"+str(uuid.uuid4())+".json"), runtime)
                            provider = providers[identity]
                            count = provider.count_input_tokens(built["messages"])
                            if type(count) is not int or count <= 0:
                                raise ProviderError("exact_input_token_count_unavailable")
                            if count+model["sampling"]["max_output_tokens"] > model["limits"]["context_tokens"]:
                                raise ProviderError("context_limit_exceeded")
                            try:
                                limiters[identity].reserve(count+model["sampling"]["max_output_tokens"],
                                                           requests=1 if model["provider"]=="transformers" or test_fixture else 0)
                            except ValueError:
                                raise ProviderError("single_request_exceeds_rate_budget") from None
                            actual_calls += 1
                            response = provider.generate(built["messages"], attempt)
                            store.write_response(attempt, response)
                        except ProviderError as error:
                            raw_error = _error(error)
                            if error.ambiguous:
                                store.mark_ambiguous(task["task_id"], attempt, raw_error)
                                aborted = error.code
                                break
                            if error.retryable:
                                store.set_retry(task["task_id"], attempt, raw_error)
                                delay = min(plan["policy"]["backoff_max_seconds"],
                                            plan["policy"]["backoff_initial_seconds"]*2**(len(current["attempts"])))
                                delay = max(delay, error.retry_after_seconds or 0)
                                while delay > 0:
                                    pause = min(60, delay)
                                    sleep(pause)
                                    delay -= pause
                                continue
                            store.write_error(attempt, raw_error)
                            store.finish(task["task_id"], attempt,
                                         _normalize(task, model, None, fixture=test_fixture, error=raw_error), "permanent_failed")
                            aborted = error.code
                            break
                        if cache:
                            cache.put(key, response, {"run_id": run_id, "task_id": task["task_id"],
                                                     "attempt_id": attempt, "execution_kind": evidence["execution_kind"],
                                                     "response_sha256": fingerprint(response)})
                        store.finish(task["task_id"], attempt,
                                     _normalize(task, model, response, fixture=test_fixture))
                        break
                    if aborted:
                        break
                if aborted:
                    break
            if aborted:
                break
        summary = store.summary()
        terminal = sum(summary["statuses"].get(k, 0) for k in ("succeeded", "cached", "permanent_failed"))
        complete = terminal == plan["generation_calls"] and not aborted and not summary["statuses"].get("permanent_failed", 0)
        report = {"run_id": run_id, "plan_sha256": plan["plan_sha256"], "dry_run": False,
                  "execution_kind": evidence["execution_kind"], "contains_model_results": not test_fixture,
                  "complete": complete, "aborted_reason": aborted,
                  "planned_generation_calls": plan["generation_calls"], "actual_generation_calls_this_invocation": actual_calls,
                  "cache_hits_this_invocation": cache_hits, "recovered_responses_this_invocation": recovered,
                  "summary": summary}
        _immutable(store.raw_dir / "invocations" / (str(uuid.uuid4())+".json"), report)
        return report
