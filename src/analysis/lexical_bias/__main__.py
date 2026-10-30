"""Offline lexical preprocessing and analysis of already recorded model runs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import tempfile

from analysis.io import canonical, decode, digest, discover_runs, load_run
from analysis.metrics import summarize_pairs
from analysis.pipeline import PROTECTED, presentation_pairs
from analysis.reporting import export_tables
from inference.config import load_models, select_models
from . import LEXICAL_VERSION
from .bias import paired_direction_report, summarize_bias
from .catalog import build_catalog
from .corpus import count_corpus
from .statistics import fit_controlled, spearman
from .tokenization import measure_prompts, measure_vocabulary


def _flat(row):
    return {key:canonical(value) if isinstance(value,(dict,list,tuple)) else value for key,value in row.items()}


def _group(rows, keys):
    groups=defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return [(dict(zip(keys,key)),groups[key]) for key in sorted(groups,key=canonical)]


def _features(row, frequencies):
    frequency=frequencies.get((row["relation_id"],row["query_relation"]),{})
    return {"log_frequency":frequency.get("lemma_log10_per_million"),
        "frequency":frequency.get("lemma_frequency_per_million"),
        "vocabulary_id":row["relation_id"]+"|"+row["query_relation"],
        "independent_vocabulary_id":frequency.get("lemma",row["relation_id"]+"|"+row["query_relation"]),
        "frequency_scope":"canonical_query_lemma_all_senses",
        "frequency_status":frequency.get("status","missing_corpus")}


def model_analysis(observations, frequencies):
    """TR lexical features; English outcomes stay separate control diagnostics.

    Regressions use planned observation or pair units with latent IDs retained.
    R/invariance models explicitly condition on joint-valid pairs; marginal
    accuracy models retain invalid failures. No iid p-values are manufactured.
    """
    index={(r["relation_id"],r["direction"]):r for r in frequencies}
    bias=summarize_bias(observations)
    direction=paired_direction_report(observations)
    tr=[row for row in observations if row["language"]=="tr"]
    original={(r["model_spec_sha256"],r["cluster_id"],r["query_id"],r["name_control"],r["relation_id"],r["language"]):r
              for r in observations if r["condition"]=="canonical"}
    by_condition={(r["model_spec_sha256"],r["cluster_id"],r["query_id"],r["name_control"],r["relation_id"],r["language"],r["condition"]):r
                  for r in observations}
    pairs=[]
    for pair in presentation_pairs(tr):
        key=(pair["model_spec_sha256"],pair["cluster_id"],pair["query_id"],pair["name_control"],pair["relation_id"],pair["language"])
        left=original[key];right=by_condition[key+(pair["comparison"],)]
        valid=pair["original"] is not None and pair["equivalent"] is not None
        tokens=(left["input_tokens"]+right["input_tokens"])/2 if left["input_tokens"] is not None and right["input_tokens"] is not None else None
        pairs.append({**pair,"query_relation":left["query_relation"],"prompt_tokens":tokens,
            **_features(left,index),"joint_valid":valid,
            "robust_correct":float(pair["original"]==pair["gold"] and pair["equivalent"]==pair["gold"]) if valid else None,
            "agreement":float(pair["original"]==pair["equivalent"]) if valid else None})
    profile_keys=PROTECTED+("language","comparison","query_relation")
    lexical_profiles=[]
    for identity,rows in _group(pairs,profile_keys):
        summary=summarize_pairs(rows)
        feature={k:rows[0][k] for k in ("frequency","log_frequency","vocabulary_id","independent_vocabulary_id","frequency_scope","frequency_status")}
        tokens=[row["prompt_tokens"] for row in rows]
        mean_tokens=sum(tokens)/len(tokens) if all(t is not None for t in tokens) else None
        lexical_profiles.append({**identity,**_flat(summary),**feature,
            "prompt_tokens":mean_tokens,"prompt_token_scope":"mean_actual_reported_pair_input; no proxy substitution",
            "n_missing_prompt_tokens":sum(t is None for t in tokens)})
    correlations=[]
    # A profile is a descriptive lexical-direction cell, never a new model trial.
    corr_keys=tuple(k for k in PROTECTED if k not in {"relation_id","relation_family"})+("language","comparison")
    for identity,rows in _group(lexical_profiles,corr_keys):
        for predictor,outcome in (("frequency","original_accuracy"),("frequency","R"),("prompt_tokens","invariance")):
            result=spearman([r[predictor] for r in rows],[r[outcome] for r in rows])
            result["spearman_n"] = result.pop("n")
            correlations.append({**identity,"predictor":predictor,"outcome":outcome,**_flat(result),
                "n_lexical_profiles":len(rows),"n_unique_lemma_units":len({r["independent_vocabulary_id"] for r in rows}),
                "scope":"descriptive lexical-direction profiles across relation families; no iid inference"})
    # Controls must vary inside a regression; only non-target experiment axes
    # define a model. Main fit handles model/n/family/direction explicitly.
    regressions=[]
    regression_keys=("split","name_control","query_type","query_distance","condition","language")
    accuracy_rows=[{**row,**_features(row,index),"prompt_tokens":row["input_tokens"],
        "accuracy":float(row["prediction"] is not None and row["prediction"]==row["gold_label"])} for row in tr]
    # Endpoint distance grows structurally with n, so it cannot be fixed when
    # estimating the n control. Preserve it explicitly as a structural policy.
    for row in accuracy_rows:
        row["query_distance"]="n-1" if row["query_type"]=="endpoint" else row["query_distance"]
    for identity,rows in _group(accuracy_rows,regression_keys):
        for interaction in (False,True):
            result=fit_controlled(rows,outcome="accuracy",interaction=interaction)
            regressions.append({**identity,"analysis_support":"all_planned_invalid_wrong",**_flat(result)})
        valid=[{**row,"predicted_yes":float(row["prediction"])} for row in rows if row["prediction"] is not None]
        result=fit_controlled(valid,outcome="predicted_yes")
        regressions.append({**identity,"analysis_support":"valid_responses_only_for_answer_share",**_flat(result),
            "n_planned_responses":len(rows),"n_excluded_invalid_responses":len(rows)-len(valid)})
    pair_regression_keys=("split","name_control","query_type","query_distance","comparison","language")
    for row in pairs:
        row["query_distance"]="n-1" if row["query_type"]=="endpoint" else row["query_distance"]
    for identity,rows in _group(pairs,pair_regression_keys):
        valid=[row for row in rows if row["joint_valid"]]
        for outcome in ("robust_correct","agreement"):
            result=fit_controlled(valid,outcome=outcome)
            regressions.append({**identity,"analysis_support":"joint_valid_pairs_only",**_flat(result),
                "n_planned_pairs":len(rows),"n_excluded_invalid_pairs":len(rows)-len(valid)})
    return {"answer_bias":bias,"paired_direction":direction,"lexical_profiles":lexical_profiles,
            "correlations":correlations,"controlled_regressions":regressions}


def run(root, output, *, model_ids=None, run_ids=None,
        corpus_manifest="data/corpora/ud_turkish_boun/r2.18/corpus_manifest.json",
        preview="data/complete_grid/pilot_preview.jsonl"):
    root=Path(root).resolve();output=Path(output).absolute()
    if output.exists():raise FileExistsError("lexical output already exists")
    if any(p.is_symlink() for p in (output,*output.parents)):
        raise ValueError("lexical output must not traverse symlinks")
    for relative in ("src","tests","configs","logs","data","tokenizers","results/raw","results/normalized"):
        if output.is_relative_to(root/relative):raise ValueError("output overlaps protected input")
    registry=load_models(root/"configs/models.yaml")
    ids=[m["id"] for m in registry["models"]] if model_ids is None else model_ids
    models=select_models(registry,ids)
    catalog=build_catalog()
    manifest_path=Path(corpus_manifest)
    if not manifest_path.is_absolute():manifest_path=root/manifest_path
    corpus=count_corpus(manifest_path,catalog)
    preview_path=Path(preview)
    if not preview_path.is_absolute():preview_path=root/preview_path
    preview_bytes=preview_path.read_bytes()
    source_rows=[decode(line) for line in preview_bytes.splitlines() if line.strip()]
    if len(source_rows)>10000:raise ValueError("explicit prompt measurement limit exceeded; select a bounded source slice")
    vocabulary=measure_vocabulary(root,models,catalog)
    prompts=measure_prompts(root,models,source_rows)
    if any(row["fixture_only"] for row in vocabulary+prompts):
        raise ValueError("synthetic tokenizer counts cannot be research preprocessing")
    # Store repeated immutable asset provenance once, referenced by every row.
    tokenizer_provenance={}
    for row in vocabulary+prompts:
        serialized=row.pop("tokenizer_provenance")
        sha=row["tokenizer_provenance_sha256"]
        if digest(serialized.encode())!=sha:raise ValueError("tokenizer provenance fingerprint mismatch")
        tokenizer_provenance[sha]=decode(serialized)
    chosen=discover_runs(root) if run_ids is None else run_ids
    if not isinstance(chosen,list):raise ValueError("run IDs must be a list")
    if len(set(chosen))!=len(chosen):raise ValueError("duplicate run selection")
    loaded=[load_run(root,r) for r in sorted(chosen)]
    if run_ids is None:
        # Failed attempts are retained for provenance and contain no observed
        # responses; automatic discovery must not pool them with completed
        # retries of the same planned tasks. An explicit selection stays strict.
        loaded=[r for r in loaded if any(row["response_observed"] for row in r["observations"])]
    observations=[];seen=set();unselected=0
    for recorded in loaded:
        if recorded["synthetic"]:raise ValueError("fixture result cannot be lexical research data")
        if recorded["plan"]["cohort"]["cohort_sha256"]!=loaded[0]["plan"]["cohort"]["cohort_sha256"]:
            raise ValueError("run cohorts differ")
        for row in recorded["observations"]:
            if row["model_id"] not in ids:
                unselected+=1;continue
            if row["task_id"] in seen:raise ValueError("overlapping run tasks")
            seen.add(row["task_id"]);observations.append(row)
    tables={"corpus_frequencies":[_flat(row) for row in corpus["rows"]],
            "tokenization_vocabulary":vocabulary,"tokenization_prompt_preview":prompts}
    has_results=any(row["response_observed"] for row in observations)
    if has_results:
        tables.update(model_analysis(observations,corpus["rows"]))
    source={str(p.relative_to(root)):digest(p.read_bytes()) for p in sorted((root/"src/analysis/lexical_bias").glob("*.py"))}
    active=Path(__file__).resolve().parent
    if any(digest((active/Path(p).name).read_bytes())!=sha for p,sha in source.items()):
        raise ValueError("installed lexical code differs from project")
    versions={}
    for name in ("transformers","tokenizers","Jinja2","numpy","tiktoken","regex","PyYAML"):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    input_hashes={"configs/models.yaml":digest((root/"configs/models.yaml").read_bytes()),
        str(preview_path.relative_to(root)):digest(preview_bytes)}
    for name in ("tokenizers/encodings.json","tokenizers/hf/manifest.json","requirements-lexical.lock"):
        p=root/name
        if p.exists():input_hashes[name]=digest(p.read_bytes())
    result={"lexical_version":LEXICAL_VERSION,"artifact_type":"lexical_preprocessing_and_analysis",
        "status":"completed_model_analysis" if has_results else "preprocessing_only_no_model_results",
        "contains_model_results":has_results,"contains_model_weights":False,"model_api_calls":0,
        "source_files_sha256":source,"input_files_sha256":input_hashes,
        "tokenizer_measurement_models":models,"selected_model_ids":ids,
        "tokenizer_provenance_records":tokenizer_provenance,
        "n_unselected_model_responses":unselected,"corpus":{k:v for k,v in corpus.items() if k!="rows"},
        "environment":{"python":platform.python_version(),"platform":platform.platform(),"packages":versions},
        "prompt_measurement_scope":{"source":"frozen pilot preview","n_source_prompts":len(source_rows),
            "n_levels":sorted({r["n"] for r in source_rows}),"complete_bank_measurement":False},
        "tokenizer_coverage":{m["id"]:{"vocabulary_rows":sum(r["model_id"]==m["id"] for r in vocabulary),
            "vocabulary_counted":sum(r["model_id"]==m["id"] and r["term_tokens"] is not None for r in vocabulary),
            "prompt_rows":sum(r["model_id"]==m["id"] for r in prompts),
            "prompt_counted":sum(r["model_id"]==m["id"] and r["source_prompt_plain_tokens"] is not None for r in prompts),
            "chat_counted":sum(r["model_id"]==m["id"] and r["chat_token_count"] is not None for r in prompts)} for m in models},
        "n_planned_model_responses":len(observations),"n_observed_model_responses":sum(r["response_observed"] for r in observations),
        "input_runs":[{"run_id":r["run_id"],"plan_sha256":r["plan"]["plan_sha256"],
            "files_sha256":r["input_files_sha256"],"ignored_replicates":r["ignored_replicates"]} for r in loaded],
        "unavailable_analyses":[] if has_results else ["Frequency_vs_Accuracy","Frequency_vs_R",
            "Token_Count_vs_Invariance","Lexical_Direction_vs_Answer_Bias","Model_x_Vocabulary"],
        "limitations":["Correlational, not causal or model-weight analysis.",
            "Corpus genre/sense coverage is limited; root frequencies are not all P4 predicate frequencies.",
            "Plain text, local chat template and provider-billed token scopes are distinct.",
            "Prompt coverage is the bounded preview; missing/gated tokenizers remain null.",
            "Few distinct lexical units; descriptive Spearman/OLS without iid inference."]}
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix=".lexical-",dir=output.parent))
    try:
        paths=export_tables(tables,temporary/"tables")
        cp=temporary/"relation_vocabulary.json";cp.write_text(json.dumps(catalog,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8");paths.append(cp)
        result["table_rows"]={k:len(v) for k,v in tables.items()}
        result["files"]=[{"path":str(p.relative_to(temporary)),"bytes":p.stat().st_size,"sha256":digest(p.read_bytes())} for p in sorted(paths)]
        (temporary/"lexical_manifest.json").write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        if output.exists():raise FileExistsError("lexical output appeared concurrently")
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary,ignore_errors=True);raise
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path.cwd())
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--model",nargs="+",dest="model_ids")
    parser.add_argument("--run",nargs="*",dest="run_ids")
    parser.add_argument("--corpus-manifest",default="data/corpora/ud_turkish_boun/r2.18/corpus_manifest.json")
    parser.add_argument("--preview",default="data/complete_grid/pilot_preview.jsonl")
    args=parser.parse_args(argv)
    try:
        result=run(args.root,args.output,model_ids=args.model_ids,run_ids=args.run_ids,
                   corpus_manifest=args.corpus_manifest,preview=args.preview)
    except (ValueError,OSError,KeyError) as error:
        print("error: "+str(error),file=sys.stderr);return 2
    print(json.dumps({k:result[k] for k in ("status","contains_model_results","model_api_calls","table_rows","tokenizer_coverage")},ensure_ascii=False,sort_keys=True))
    return 0


if __name__=="__main__":raise SystemExit(main())
