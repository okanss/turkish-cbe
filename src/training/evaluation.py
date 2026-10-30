"""Separate, gated held-out evaluation of the base snapshot or saved adapters."""

import hashlib
import json
from pathlib import Path

from analysis.io import canonical, decode
from inference.dataset import GridDataset, fingerprint
from inference.parser import parse_answer
from inference.prompts import make_messages
from inference.runner import _immutable
from inference.linguistic_review import validate_review, LARGE_EVALUATION_CALLS
from renderers.turkish import RELATIONS
from .engine import execution_gate, load_runtime, TrainingBlocked, _eos_ids
from .planning import validate_plan

SOURCE_KEYS = ("prompt_id", "latent_instance_id", "grid_query_id", "n", "query_type", "query_distance",
    "query_direction", "query_subject", "query_object", "query_relation", "gold_label", "language",
    "condition", "name_control", "relation_id", "split", "human_review_status", "eligible_for_confirmatory_evaluation")

def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def adapter_identity(plan, arm, directory=None):
    if arm == "base":
        if directory is not None: raise ValueError("base evaluation cannot load an adapter")
        return {"arm": "base", "base_model_spec_sha256": plan["model_spec_sha256"], "adapter_sha256": None}
    if directory is None: raise ValueError("a completed adapter run is required")
    directory = Path(directory).resolve()
    value = decode((directory / "training_summary.json").read_bytes())
    original = decode((directory / "plan.json").read_bytes())
    if (value.get("status") != "completed" or value.get("contains_training_run") is not True
            or value.get("arm") != arm or value.get("plan_sha256") != plan["plan_sha256"] or original != plan
            or value.get("base_model") != plan["model"] or value.get("stats", {}).get("optimizer_steps", 0) <= 0):
        raise ValueError("adapter is not a completed run of this exact model/data/config/seed plan")
    files = value.get("files")
    if not isinstance(files, list) or not files: raise ValueError("missing adapter inventory")
    for item in files:
        relative = item["path"]
        path = directory / relative
        if (type(relative) is not str or not relative.startswith("adapter/") or ".." in Path(relative).parts
                or not path.is_file() or any(p.is_symlink() for p in (path, *path.parents))
                or path.stat().st_size != item["bytes"] or _hash(path) != item["sha256"]):
            raise ValueError("adapter payload hash/path mismatch")
    actual = {str(path.relative_to(directory)) for path in (directory / "adapter").rglob("*") if path.is_file()}
    if actual != {item["path"] for item in files}: raise ValueError("unbound adapter assets")
    evidence = value.get("evidence_files", [])
    if len(evidence) != 3 or {item["path"] for item in evidence} != {"plan.json", "execution.json", "steps.jsonl"}:
        raise ValueError("training evidence inventory mismatch")
    for item in evidence:
        path = directory / item["path"]
        if path.stat().st_size != item["bytes"] or _hash(path) != item["sha256"]:
            raise ValueError("training evidence hash mismatch")
    execution = decode((directory / "execution.json").read_bytes())
    steps = [decode(line) for line in (directory / "steps.jsonl").read_bytes().splitlines()]
    if (execution.get("execution_kind") != "live" or execution.get("arm") != arm or not steps
            or len(steps) != value["stats"]["optimizer_steps"]
            or any(step.get("step") != i for i, step in enumerate(steps, 1))
            or any(value["stats"].get(key) != number for key, number in steps[-1]["stats"].items())):
        raise ValueError("training step/stat provenance mismatch")
    return {"arm": arm, "base_model_spec_sha256": plan["model_spec_sha256"],
            "adapter_sha256": fingerprint(files), "training_summary_sha256": _hash(directory / "training_summary.json")}


def planned_calls(plan):
    return sum(cohort["prompt_count"] for cohort in plan["data"]["evaluation"].values())


def evaluation_sampling(model, tokenizer, cap, seed):
    """Fresh explicit greedy policy; no snapshot penalties/beams/forced tokens."""
    from transformers import GenerationConfig
    generation = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=cap,
        pad_token_id=tokenizer.eos_token_id, eos_token_id=sorted(_eos_ids(model,tokenizer)))
    return generation, {"generation_config":generation.to_dict(), "use_model_defaults":False, "seed":seed}


def _review(root, plan, receipt, *, samples_path=None):
    try:
        return validate_review(root,plan["data"]["dataset_manifest"],receipt,
            [c["selection"] for c in plan["data"]["evaluation"].values()],samples_path=samples_path)
    except (ValueError,OSError) as error:
        raise TrainingBlocked(str(error)) from error


def evaluate(root, plan, arm, output, *, adapter_run=None, execute=False, max_calls=2000,
             linguistic_review=None):
    validate_plan(plan, root)
    if arm not in plan["config"]["arms"] or type(max_calls) is not int or max_calls < 1:
        raise ValueError("explicit arm and positive call budget required")
    calls = planned_calls(plan)
    if not execute:
        return {"status": "evaluation_dry_run", "arm": arm, "planned_generation_calls": calls,
            "actual_generation_calls": 0, "contains_model_results": False,
            "over_call_budget": calls > max_calls, "training_blockers": plan["blockers"],
            "linguistic_review_required": calls > LARGE_EVALUATION_CALLS,
            "maximum_output_tokens": calls*plan["config"]["budget"]["output_tokens"]}
    if calls > max_calls: raise TrainingBlocked("evaluation_call_budget_exceeded")
    review = _review(root,plan,linguistic_review) if calls > LARGE_EVALUATION_CALLS else None
    execution_gate(root, plan)
    identity = adapter_identity(plan, arm, adapter_run)
    output, root = Path(output).absolute(), Path(root).resolve()
    if (output.exists() or not output.is_relative_to(root / "results/training/evaluations")
            or any(p.is_symlink() for p in (output, *output.parents))):
        raise ValueError("choose a new nonsymlink evaluation directory")
    model, tokenizer, torch, options = load_runtime(plan, "base", adapter=None if arm == "base" else Path(adapter_run) / "adapter")
    model.eval(); torch.manual_seed(plan["config"]["seed"]); torch.cuda.manual_seed_all(plan["config"]["seed"])
    torch.use_deterministic_algorithms(True)
    generation, sampling = evaluation_sampling(model,tokenizer,plan["config"]["budget"]["output_tokens"],plan["config"]["seed"])
    output.mkdir(parents=True)
    review_files = []
    if review:
        for source,name,expected in ((review["review_path"],"linguistic_review.json",review["receipt_sha256"]),
                                     (review["samples_path"],"reviewed_samples.csv",review["samples_sha256"])):
            raw = Path(source).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=expected: raise TrainingBlocked("linguistic review changed before execution")
            with (output/name).open("xb") as handle: handle.write(raw)
            review_files.append(output/name)
    _immutable(output / "plan.json", plan)
    _immutable(output / "identity.json", identity)
    count = 0
    path = output / "responses.jsonl"
    with path.open("x", encoding="utf-8") as handle, GridDataset(root, plan["data"]["dataset_manifest"]) as data:
        for transfer, cohort in sorted(plan["data"]["evaluation"].items()):
            for row in data.iter_rows(cohort):
                built = make_messages(row)
                prefix = list(tokenizer.apply_chat_template(built["messages"], tokenize=True, add_generation_prompt=True, **options))
                cap = plan["config"]["budget"]["output_tokens"]
                text, truncated, completion = None, False, []
                error = None
                if len(prefix)+cap > min(plan["model"]["limits"]["context_tokens"], int(model.config.max_position_embeddings)):
                    error = "context_limit_exceeded_without_truncation"
                else:
                    ids = torch.tensor([prefix], dtype=torch.long, device=next(model.parameters()).device)
                    with torch.no_grad():
                        generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                            generation_config=generation, use_model_defaults=False)
                    completion = generated[0, len(prefix):].tolist()
                    truncated = bool(completion) and len(completion) >= cap and completion[-1] not in _eos_ids(model, tokenizer)
                    text = tokenizer.decode(completion, skip_special_tokens=True)
                parsed = parse_answer(text, row["language"], truncated=truncated)
                if error: parsed.update(valid=False, prediction=None, invalid_reason=error)
                source = {key: row[key] for key in SOURCE_KEYS}
                record = {"task_id": "training-eval-"+fingerprint([identity, transfer, row["prompt_id"]]),
                    "transfer": transfer, "source": source, "source_prompt_sha256": row["prompt_sha256"],
                    "messages_sha256": built["messages_sha256"], "text": text, "truncated": truncated,
                    "error": error, "parse": parsed, "input_tokens": len(prefix), "output_tokens": len(completion),
                    "sampling_sha256":fingerprint(sampling), "generation_called":error is None,
                    "cost":{"amount":None,"currency":None,"status":"local_compute_not_measured"},
                    "execution_kind": "live", "contains_model_results": True}
                handle.write(canonical(record)+"\n"); handle.flush(); count += 1
    if count != calls: raise ValueError("evaluation did not retain its planned denominator")
    manifest = {"artifact_type": "training_adapter_evaluation", "training_version": plan["training_version"],
        "status": "completed", "execution_kind": "live", "contains_model_results": True,
        "plan_sha256": plan["plan_sha256"], "identity": identity,
        "planned_responses": calls, "recorded_responses": count,
        "sampling": sampling, "linguistic_review":review,
        "files": [{"path": p.name, "bytes": p.stat().st_size, "sha256": _hash(p)}
                  for p in (output / "plan.json", output / "identity.json", path, *review_files)]}
    _immutable(output / "evaluation_manifest.json", manifest)
    return manifest


def load_evaluation(root, directory):
    """Read completed evidence only; never silently analyse a partial run."""
    directory = Path(directory).absolute()
    if any(p.is_symlink() for p in (directory, *directory.parents)): raise ValueError("symlinked evidence")
    manifest = decode((directory / "evaluation_manifest.json").read_bytes())
    if manifest.get("status") != "completed" or manifest.get("execution_kind") != "live" or manifest.get("contains_model_results") is not True:
        raise ValueError("only completed real evaluations can be analysed")
    names={item["path"] for item in manifest["files"]}
    basic={"plan.json","identity.json","responses.jsonl"}
    if len(manifest["files"])!=len(names) or names not in (basic,basic|{"linguistic_review.json","reviewed_samples.csv"}):
        raise ValueError("evaluation inventory mismatch")
    for item in manifest["files"]:
        path = directory / item["path"]
        if path.stat().st_size != item["bytes"] or _hash(path) != item["sha256"]: raise ValueError("evaluation hash mismatch")
    plan = decode((directory / "plan.json").read_bytes()); validate_plan(plan, root)
    if planned_calls(plan)>LARGE_EVALUATION_CALLS:
        review=_review(root,plan,directory/"linguistic_review.json",samples_path=directory/"reviewed_samples.csv")
        recorded=manifest.get("linguistic_review",{})
        if any(review[k]!=recorded.get(k) for k in ("receipt_sha256","samples_sha256","required_axes","sample_count")):
            raise ValueError("evaluation review provenance mismatch")
    sampling=manifest.get("sampling",{})
    params=sampling.get("generation_config",{})
    if (sampling.get("use_model_defaults") is not False or params.get("do_sample") is not False
            or params.get("num_beams")!=1 or params.get("repetition_penalty")!=1
            or params.get("forced_bos_token_id") is not None or params.get("forced_eos_token_id") is not None
            or params.get("max_new_tokens")!=plan["config"]["budget"]["output_tokens"]):
        raise ValueError("evaluation sampling policy is not the recorded neutral greedy condition")
    identity = decode((directory / "identity.json").read_bytes())
    if identity != manifest["identity"] or plan["plan_sha256"] != manifest["plan_sha256"]: raise ValueError("evaluation identity mismatch")
    if identity.get("arm") not in plan["config"]["arms"] or identity.get("base_model_spec_sha256") != plan["model_spec_sha256"]:
        raise ValueError("evaluation arm/base identity mismatch")
    lines = (directory / "responses.jsonl").read_bytes().splitlines()
    expected = planned_calls(plan)
    if len(lines) != expected or manifest["recorded_responses"] != expected or manifest["planned_responses"] != expected:
        raise ValueError("partial/unplanned evaluation denominator; complete the run before analysis")
    observations, index = [], 0
    with GridDataset(root, plan["data"]["dataset_manifest"]) as data:
        for transfer, cohort in sorted(plan["data"]["evaluation"].items()):
            for row in data.iter_rows(cohort):
                record = decode(lines[index]); index += 1
                built = make_messages(row); source = record["source"]
                if (record["task_id"] != "training-eval-"+fingerprint([identity, transfer, row["prompt_id"]])
                        or record["transfer"] != transfer or record["source_prompt_sha256"] != row["prompt_sha256"]
                        or record["messages_sha256"] != built["messages_sha256"]
                        or set(source) != set(SOURCE_KEYS) or any(source[key] != row[key] for key in SOURCE_KEYS)):
                    raise ValueError("evaluation changed frozen query/gold/conditions")
                parsed = parse_answer(record["text"], row["language"], truncated=record["truncated"])
                if record["error"]:
                    if record["error"] != "context_limit_exceeded_without_truncation": raise ValueError("unknown evaluation error")
                    parsed.update(valid=False, prediction=None, invalid_reason=record["error"])
                if record.get("sampling_sha256")!=fingerprint(sampling):
                    raise ValueError("response decoding policy hash mismatch")
                if parsed != record["parse"] or record["execution_kind"] != "live" or record["contains_model_results"] is not True:
                    raise ValueError("evaluation parser/execution mismatch")
                if any(type(record[key]) is not int or record[key] < 0 for key in ("input_tokens", "output_tokens")):
                    raise ValueError("invalid evaluation token usage")
                observations.append({**source, "cluster_id": row["latent_instance_id"], "query_id": row["grid_query_id"],
                    "model_id": plan["model"]["id"]+"--"+identity["arm"], "model_family": plan["model"]["family"],
                    "model_revision": plan["model"]["revision"], "model_spec_sha256": fingerprint([plan["model"], identity]),
                    "relation_family": RELATIONS[row["relation_id"]].family,
                    "prediction": parsed["prediction"] if parsed["valid"] else None,
                    "response_status": "valid" if parsed["valid"] else "invalid", "response_observed": record["text"] is not None,
                    "invalid_reason": parsed["invalid_reason"], "input_tokens": record["input_tokens"],
                    "source_prompt_tokens": None, "token_encoding": None, "transfer": transfer,
                    "arm": identity["arm"], "task_id": record["task_id"], "run_id": directory.name})
    return {"plan": plan, "manifest": manifest, "observations": observations,
            "manifest_sha256": _hash(directory / "evaluation_manifest.json")}
