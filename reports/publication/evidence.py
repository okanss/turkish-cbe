"""Read verified evidence without calling models, repairing journals, or inventing results.

Only native immutable journals are accepted as model-result inputs. CSV files are
never accepted as model observations. Dataset counts are construction statistics,
not estimates of any model's performance.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import math
import hashlib
from pathlib import Path

from analysis.io import decode, discover_runs, load_run
from analysis.pipeline import build_tables
from inference.dataset import GridDataset
from reasoning.io import discover_runs as discover_reasoning_runs, load_run as load_reasoning_run
from reasoning.analysis import build_tables as build_reasoning_tables
from training.evaluation import load_evaluation, adapter_identity
from training.planning import validate_plan as validate_training_plan
from training.comparison import training_contrasts


MANIFEST = "data/complete_grid/manifest.json"
# Current immutable preprocessing-only snapshot. Earlier snapshots
# (baseline, with_llama) remain retained; their manifests pin the registry
# and source tree as of their run and no longer match the live tree.
LEXICAL = "results/lexical/preprocessing_current/lexical_manifest.json"


def _safe(root, value):
    if not isinstance(value, (str, Path)):
        raise ValueError("evidence paths must be strings or Paths")
    value = Path(value)
    if ".." in value.parts:
        raise ValueError("publication evidence path traversal is forbidden")
    path = value if value.is_absolute() else root / value
    if not path.absolute().is_relative_to(root):
        raise ValueError("publication evidence escapes project")
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("publication evidence must not traverse symlinks")
    if not path.resolve().is_relative_to(root):
        raise ValueError("publication evidence escapes project")
    return path


def _hash(root, value, sources, *, expected=None, size=None):
    path = _safe(root, value)
    if not path.is_file():
        raise ValueError("missing publication evidence: " + str(value))
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    sha = digest.hexdigest()
    if expected is not None and sha != expected or size is not None and path.stat().st_size != size:
        raise ValueError("publication evidence hash/size mismatch: " + str(value))
    key = str(path.relative_to(root))
    if key in sources and sources[key] != sha:
        raise ValueError("input changed during publication collection")
    sources[key] = sha
    return sha


def _merge_hashes(root, hashes, sources):
    for path, sha in hashes.items():
        _hash(root, path, sources, expected=sha)


def _dataset(root, sources):
    with GridDataset(root, MANIFEST) as data:
        manifest = data.manifest
        _hash(root, MANIFEST, sources, expected=data.manifest_sha256)
        parent = Path(MANIFEST).parent
        for item in manifest["files"]:
            _hash(root, parent / item["path"], sources, expected=item["sha256"], size=item["bytes"])
        binding = manifest["base_binding"]
        _hash(root, binding["manifest_path"], sources, expected=binding["sha256"])
        for path, sha in binding["source_files_sha256"].items():
            _hash(root, Path("src") / path, sources, expected=sha)
        for path, sha in manifest["source_files_sha256"].items():
            _hash(root, Path("src/complete_grid") / path, sources, expected=sha)
        connection = data.connection
        expected = manifest["expected_counts"]
        coverage = decode((root / parent / "coverage.json").read_bytes())
        query_counts = {(r[0],r[1]):r[2] for r in connection.execute(
            "SELECT g.split,g.n,count(*) FROM queries q JOIN groups g ON g.id=q.group_id GROUP BY g.split,g.n")}
        prompt_counts = {(r[0],r[1]):r[2] for r in connection.execute(
            "SELECT g.split,g.n,count(*) FROM measurements m JOIN groups g ON g.id=m.group_id GROUP BY g.split,g.n")}
        # comparison_cells is a virtual Cartesian view. Its exact cardinality
        # follows from verified rule counts and one endpoint per latent group;
        # enumerating every Cartesian row would add no independent evidence.
        rule_counts = dict(connection.execute("SELECT arm,count(*) FROM comparison_rules GROUP BY arm"))
        if connection.execute("SELECT count(*) FROM (SELECT group_id FROM queries GROUP BY group_id HAVING sum(query_type='endpoint') != 1)").fetchone()[0]:
            raise ValueError("each latent group must have exactly one endpoint query")
        nonendpoint = {(r[0],r[1]):r[2] for r in connection.execute(
            "SELECT g.split,g.n,count(*) FROM queries q JOIN groups g ON g.id=q.group_id WHERE q.query_type!='endpoint' GROUP BY g.split,g.n")}
        arm_counts = {key:{arm:count*(nonendpoint.get(key,0) if arm=='F' else queries)
                         for arm,count in rule_counts.items()} for key,queries in query_counts.items()}
        rows = []
        for split, n, orders in connection.execute(
                "SELECT split,n,count(*) FROM groups GROUP BY split,n ORDER BY split,n"):
            queries = query_counts[split,n]
            prompts = prompt_counts[split,n]
            arms = arm_counts[split,n]
            actual = {"orders":orders, "queries":queries, "prompts":prompts}
            if actual != expected["splits"][split]["by_n"][str(n)]:
                raise ValueError("dataset split/n counts differ from frozen manifest")
            rows.append({"split":split, "n":n, **actual, "latent_instances":orders,
                         "comparison_cells":sum(arms.values()), "cells_by_arm":arms})
        split_counts = {}
        for split in sorted({r["split"] for r in rows}):
            selected = [r for r in rows if r["split"] == split]
            counts = {key:sum(r[key] for r in selected) for key in ("orders","queries","prompts","comparison_cells")}
            counts["cells_by_arm"] = dict(sorted(Counter({arm:sum(r["cells_by_arm"].get(arm,0) for r in selected)
                                                           for arm in "ABCDEF"}).items()))
            if counts != coverage["splits"][split] or any(counts[key] != expected["splits"][split][key] for key in counts):
                raise ValueError("dataset aggregate counts differ from coverage/manifest")
            split_counts[split] = counts
        if set(split_counts) != set(expected["splits"]) or set(split_counts) != set(coverage["splits"]):
            raise ValueError("missing or additional dataset split")
        query_strata = [{"split":r[0], "n":r[1], "query_type":r[2], "query_distance":r[3],
            "queries":r[4], "gold_yes":r[5], "gold_no":r[4]-r[5]} for r in connection.execute(
            "SELECT g.split,g.n,q.query_type,q.query_distance,count(*),sum(q.gold_label) FROM queries q JOIN groups g ON g.id=q.group_id GROUP BY g.split,g.n,q.query_type,q.query_distance ORDER BY g.split,g.n,q.query_type,q.query_distance")]
        totals = {key:sum(r[key] for r in rows) for key in ("orders","queries","prompts","comparison_cells")}
        if any(totals[key] != expected["totals"][key] for key in totals):
            raise ValueError("dataset totals differ from frozen manifest")
        metadata = {"dataset_manifest":MANIFEST, "dataset_manifest_sha256":data.manifest_sha256,
            "grid_version":manifest["grid_version"], "totals":totals, "splits":split_counts,
            "human_review_status":manifest["human_review_status"], "query_strata":query_strata,
            "designs":[dict(r) for r in connection.execute("SELECT * FROM designs ORDER BY id")],
            "source_query_answer_cells":coverage["source_query_cells"],
            "comparison_rule_count":connection.execute("SELECT count(*) FROM comparison_rules").fetchone()[0],
            "count_scope":"Frozen construction statistics; prompts and comparison cells are dependent surface realizations, not independent samples.",
            "eligible_for_confirmatory_evaluation":False}
    return rows, metadata


def _lexical(root, sources):
    path = root / LEXICAL
    if not path.exists():
        return {"status":"not_available", "contains_model_results":False}, []
    _hash(root, LEXICAL, sources)
    value = decode(path.read_bytes())
    if (value.get("artifact_type") != "lexical_preprocessing_and_analysis"
            or value.get("status") != "preprocessing_only_no_model_results"
            or value.get("contains_model_results") is not False
            or value.get("n_observed_model_responses") != 0 or value.get("model_api_calls") != 0):
        raise ValueError("baseline must be immutable lexical preprocessing, not model results")
    for item in value["files"]:
        _hash(root, path.parent / item["path"], sources, expected=item["sha256"], size=item["bytes"])
    _merge_hashes(root, value["input_files_sha256"], sources)
    _merge_hashes(root, value["source_files_sha256"], sources)
    from analysis.lexical_bias.catalog import build_catalog
    from analysis.lexical_bias.corpus import count_corpus
    corpus_manifest = root / "data/corpora/ud_turkish_boun/r2.18/corpus_manifest.json"
    corpus = count_corpus(corpus_manifest, build_catalog())
    if {key:val for key,val in corpus.items() if key != "rows"} != value["corpus"]:
        raise ValueError("lexical corpus counts/provenance changed")
    for name, sha in corpus["input_files_sha256"].items():
        relative = name.removeprefix("manifest:")
        _hash(root, corpus_manifest.parent / relative, sources, expected=sha)
    for sha, provenance in value["tokenizer_provenance_records"].items():
        from inference.dataset import fingerprint
        if fingerprint(provenance) != sha or provenance.get("fixture_only") is not False:
            raise ValueError("synthetic or changed tokenizer provenance")
        assets = provenance.get("verified_local_assets", {})
        for item in assets.get("assets", []):
            _hash(root, Path(assets["cache_dir"]) / item["path"], sources, expected=item["sha256"], size=item["bytes"])
        bundle = provenance.get("frozen_bundle", {})
        for item in bundle.get("encodings", {}).values():
            _hash(root, Path("tokenizers") / item["file"], sources, expected=item["sha256"])
    # These CSV rows describe isolated-string preprocessing only. They are
    # accepted solely through the validated baseline inventory, never as model
    # predictions or performance summaries. Corpus rows are recomputed above.
    vocabulary = []
    with (path.parent/"tables/tokenization_vocabulary.csv").open(newline="",encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("duplicate/missing tokenizer table fields")
        integers = {"term_tokens","expression_tokens","lemma_token_count","chat_token_count",
                    "term_direction_delta_after_minus_before","expression_direction_delta_after_minus_before"}
        ratios = {"term_tr_en_ratio","expression_tr_en_ratio"}
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("malformed tokenizer table row")
            typed = {key:None if text == "NA" else text for key,text in row.items()}
            if typed.get("fixture_only") != "false" or typed.get("weights_loaded") != "false":
                raise ValueError("synthetic/model-weight output is not tokenizer preprocessing")
            for key in integers:
                if typed.get(key) is not None: typed[key] = int(typed[key])
            for key in ratios:
                if typed.get(key) is not None:
                    typed[key] = float(typed[key])
                    if not math.isfinite(typed[key]): raise ValueError("nonfinite tokenizer ratio")
            for key in ("fixture_only","network_used","weights_loaded"):
                if typed[key] not in {"true","false"}: raise ValueError("invalid tokenizer Boolean")
                typed[key] = typed[key] == "true"
            vocabulary.append(typed)
    if len(vocabulary) != value["table_rows"]["tokenization_vocabulary"]:
        raise ValueError("tokenizer preprocessing row count mismatch")
    metadata = {key:value[key] for key in ("status","contains_model_results","corpus","prompt_measurement_scope",
            "tokenizer_coverage","table_rows","limitations","tokenizer_measurement_models")}
    metadata.update(corpus_frequencies=corpus["rows"],tokenization_vocabulary=vocabulary,
                    preprocessing_scope="Independent corpus and isolated strings; no model-answer outcomes.")
    return metadata, corpus["rows"]


def _selection(root, supplied, discover, *, paths=False):
    values = discover(root) if supplied is None else supplied
    if not isinstance(values, (list, tuple)) or isinstance(values, (str, bytes)):
        raise ValueError("run selections must be lists, never CSV/JSONL observation inputs")
    if paths:
        values = [str(_safe(root, value)) for value in values]
    elif any(type(value) is not str for value in values):
        raise ValueError("run IDs must be strings")
    if len(set(values)) != len(values):
        raise ValueError("duplicate run selection")
    return sorted(values)


def _discover(root, relative, marker):
    return sorted(str(p.parent) for p in (root / relative).glob("*/"+marker))


def _counts(loaded):
    rows = [row for item in loaded for row in item["observations"]]
    return {"runs":len(loaded), "planned_responses":len(rows),
            "observed_responses":sum(row.get("response_observed",False) for row in rows),
            "valid_responses":sum(row["prediction"] is not None for row in rows)}


def _run_observations(root, selected, reader, sources, conditions, kind):
    loaded = [reader(root, run_id) for run_id in selected]
    seen, observations, cohorts = set(), [], set()
    for item in loaded:
        if item.get("synthetic") is not False:
            raise ValueError("synthetic fixture cannot be publication evidence")
        _merge_hashes(root, item["input_files_sha256"], sources)
        plan = item["plan"]
        _merge_hashes(root, plan["source_files_sha256"], sources)
        cohorts.add(plan["cohort"]["cohort_sha256"])
        for row in item["observations"]:
            if row.get("fixture_only") is True:
                raise ValueError("synthetic fixture row cannot be publication evidence")
            key = tuple(row.get(k) for k in ("model_id","model_spec_sha256","arm_id","effective_model_spec_sha256","cluster_id","query_id","language","condition","name_control","relation_id"))
            if key in seen:
                raise ValueError("overlapping or duplicate planned model observations")
            seen.add(key); observations.append(row)
        conditions.append({"source_kind":kind, "run_id":item["run_id"], "plan_sha256":plan["plan_sha256"],
            "cohort_sha256":plan["cohort"]["cohort_sha256"], "models":plan["models"],
            "arms":plan.get("arms",[]), "policy":plan["policy"], "environment":plan["environment"],
            "eligible_for_confirmatory_evaluation":False})
    if len(cohorts) > 1:
        raise ValueError("run cohorts differ; report distinct scientific cohorts separately")
    return loaded, observations


def collect_evidence(root, *, inference_runs=None, reasoning_runs=None,
                     training_evaluations=None, training_runs=None, seed=20261025, repeats=2000):
    """Collect hashed, verified native evidence; absence is never an accuracy score.

    ``None`` discovers native journals. An empty list explicitly selects none.
    Training paths must be inside their native results directories. Incomplete,
    synthetic, modified or duplicate evidence raises rather than being excluded.
    """
    if type(seed) is not int or seed < 0 or type(repeats) is not int or repeats < 1:
        raise ValueError("seed and bootstrap repeats must be positive/nonnegative integers")
    root = Path(root).resolve()
    sources, conditions, limitations = {}, [], [
        "Human linguistic review remains pending; no output is eligible for confirmatory evaluation.",
        "Runner journals establish recorded provenance, not independent attestation of provider execution.",
        "Dataset prompts/comparison cells share latent structures and must not be counted as independent samples.",
        "Corpus frequency associations are correlational, corpus-specific, and do not identify causal effects or model parameter weights.",
        "Held-out training predicate vocabulary is disjoint, but English presentation contexts share higher/lower; full-context lexical novelty is not established."]
    dataset_rows, dataset_metadata = _dataset(root, sources)
    backend_path = "results/training/backend_validation.json"
    if (root/backend_path).exists():
        _hash(root,backend_path,sources)
        backend = decode((root/backend_path).read_bytes())
        if backend.get("artifact_type") != "training_backend_validation":
            raise ValueError("invalid training backend validation artifact")
        dataset_metadata["training_backend_validation"] = backend
        optional = backend.get("optional_torch_numerical_gradient_tests",{})
        if not optional.get("executed",False):
            limitations.append(f"The recorded environment did not run {optional.get('test_count_not_executed','the optional')} Torch numerical/gradient tests; pure-Python tests do not establish autograd correctness.")
        if not backend.get("cuda_training_smoke",{}).get("executed",False):
            limitations.append("CUDA/model training smoke validation remains unexecuted in the recorded environment; no training compatibility or training success is established.")
    hardware_path = "results/training/hardware.json"
    if (root/hardware_path).exists():
        _hash(root,hardware_path,sources)
        dataset_metadata["recorded_training_preflight_hardware"] = decode((root/hardware_path).read_bytes())
    lexical, frequencies = _lexical(root, sources)
    dataset_metadata["lexical_preprocessing"] = lexical
    inference_ids = _selection(root, inference_runs, discover_runs)
    reasoning_ids = _selection(root, reasoning_runs, discover_reasoning_runs)
    loaded, observations = _run_observations(root,inference_ids,load_run,sources,conditions,"inference")
    reason_loaded, reason_observations = _run_observations(root,reasoning_ids,load_reasoning_run,sources,conditions,"reasoning")
    inference_counts, reasoning_counts = _counts(loaded), _counts(reason_loaded)
    inference_tables = build_tables(observations,seed=seed,repeats=repeats) if inference_counts["observed_responses"] else {}
    reasoning_tables = build_reasoning_tables(reason_observations,seed=seed,repeats=repeats) if reasoning_counts["observed_responses"] else {}
    lexical_tables = {}
    if inference_counts["observed_responses"]:
        from analysis.lexical_bias.__main__ import model_analysis
        lexical_tables = model_analysis(observations, frequencies)
        if not frequencies:
            limitations.append("No independent corpus baseline is available; lexical frequency fields remain missing.")
    evaluation_paths = _selection(root,training_evaluations,
        lambda r:_discover(r,"results/training/evaluations","evaluation_manifest.json"),paths=True)
    run_paths = _selection(root,training_runs,
        lambda r:_discover(r,"results/training/runs","training_summary.json"),paths=True)
    evaluations, training_tables, plans, trained, arm_seen = [], {}, {}, [], set()
    for directory in evaluation_paths:
        directory = _safe(root,directory)
        if not directory.is_relative_to(root/"results/training/evaluations"):
            raise ValueError("training evaluation must use its native directory")
        item = load_evaluation(root,directory)
        evaluations.append(item)
        plan, manifest = item["plan"], item["manifest"]
        plans[plan["plan_sha256"]] = plan
        arm = manifest["identity"]["arm"]
        if arm in arm_seen: raise ValueError("duplicate training evaluation arm")
        arm_seen.add(arm)
        _hash(root,directory/"evaluation_manifest.json",sources,expected=item["manifest_sha256"])
        for entry in manifest["files"]:
            _hash(root,directory/entry["path"],sources,expected=entry["sha256"],size=entry["bytes"])
        _merge_hashes(root,plan["source_files_sha256"],sources)
        conditions.append({"source_kind":"training_evaluation","run_id":directory.name,
            "plan_sha256":plan["plan_sha256"],"cohort_sha256":plan["data"]["data_sha256"],
            "models":[plan["model"]],"arm":arm,"policy":manifest["sampling"],
            "environment":plan["environment"],"eligible_for_confirmatory_evaluation":False})
    if len(plans)>1:
        raise ValueError("training evaluations require the same model/data/config/seed plan")
    for directory in run_paths:
        directory = _safe(root,directory)
        if not directory.is_relative_to(root/"results/training/runs"):
            raise ValueError("training run must use its native directory")
        plan = decode((directory/"plan.json").read_bytes())
        validate_training_plan(plan,root)
        plans[plan["plan_sha256"]] = plan
        if len(plans) > 1:
            raise ValueError("training runs/evaluations require the same model/data/config/seed plan")
        summary = decode((directory/"training_summary.json").read_bytes())
        arm = summary["arm"]
        adapter_identity(plan,arm,directory)
        key = (plan["plan_sha256"],arm)
        if any((r["plan_sha256"],r["arm"])==key for r in trained):
            raise ValueError("duplicate training arm/run")
        for entry in summary["files"]+summary["evidence_files"]:
            _hash(root,directory/entry["path"],sources,expected=entry["sha256"],size=entry["bytes"])
        _hash(root,directory/"training_summary.json",sources)
        _merge_hashes(root,plan["source_files_sha256"],sources)
        trained.append(summary)
        conditions.append({"source_kind":"training", "run_id":directory.name,
            "plan_sha256":plan["plan_sha256"],"cohort_sha256":plan["data"]["data_sha256"],
            "models":[plan["model"]],"arm":arm,"policy":plan["config"],
            "environment":plan["environment"],"actual_training_stats":summary["stats"],
            "fairness":plan["fairness"],"eligible_for_confirmatory_evaluation":False})
    for item in evaluations:
        identity = item["manifest"]["identity"]
        if identity["arm"] != "base" and not any(
                r["plan_sha256"]==item["plan"]["plan_sha256"] and r["arm"]==identity["arm"]
                and hashlib.sha256((Path(directory)/"training_summary.json").read_bytes()).hexdigest()==identity.get("training_summary_sha256")
                for directory,r in zip(run_paths,trained)):
            raise ValueError("adapter evaluation requires matching verified completed training evidence")
    if evaluations:
        by_transfer = defaultdict(list)
        for item in evaluations:
            for row in item["observations"]: by_transfer[row["transfer"]].append(row)
        for transfer,rows in sorted(by_transfer.items()):
            tables = build_tables(rows,seed=seed,repeats=repeats)
            arms = {r["model_id"]:r["arm"] for r in rows}
            for table in tables.values():
                for row in table: row.update(training_arm=arms[row["model_id"]],transfer=transfer)
            tables["training_arm_contrasts"] = training_contrasts(rows,seed=seed,repeats=repeats,
                declared_arms=[a for a in next(iter(plans.values()))["config"]["arms"] if a!="base"])
            training_tables[transfer] = tables
    training_counts = _counts(evaluations)
    availability = {"inference":inference_counts,"reasoning":reasoning_counts,
        "training_evaluations":training_counts,
        "training":{"runs":len(trained),"optimizer_steps":sum(s["stats"]["optimizer_steps"] for s in trained)},
        "lexical_preprocessing":{"available":lexical["status"]!="not_available","status":lexical["status"]}}
    contains = any(c["observed_responses"] for c in (inference_counts,reasoning_counts,training_counts))
    if not contains:
        limitations.append("No verified model responses are available. Model-performance tables, confidence intervals, significance tests and outcome figures are placeholders only.")
    return {"dataset_rows":dataset_rows,"dataset_metadata":dataset_metadata,
        "inference_tables":inference_tables,"reasoning_tables":reasoning_tables,
        "training_tables":training_tables,"lexical_tables":lexical_tables,
        "sources":dict(sorted(sources.items())),"availability":availability,
        "conditions":conditions,"limitations":limitations,"contains_model_results":contains,
        "eligible_for_confirmatory_evaluation":False}
