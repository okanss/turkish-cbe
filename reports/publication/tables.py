"""Lossless paper tables from verified evidence, with explicit missing outcomes.

This layer does not estimate, pool, round, or impute an experimental result.
The analysis modules own all statistical estimands and multiple-test families.
"""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path


TABLES = (
    "table_1_dataset_statistics",
    "table_2_overall_model_performance",
    "table_3_turkish_english_comparison",
    "table_4_morphosyntactic_robustness",
    "table_5_lexical_relation_analysis",
    "table_6_complexity_inference_distance",
    "table_7_reasoning_training",
)
METRICS = ("original_accuracy", "equivalent_accuracy", "mean_accuracy", "R", "D", "U",
           "invariance", "kappa", "invalid_response_rate", "valid_pair_coverage",
           "strict_robust_correctness", "difference", "frequency", "predicted_yes_rate")
COMMON = ("status", "research_question", "source_kind", "source_table", "row_granularity",
          "sample_size", "sample_size_valid_pairs", "sample_size_valid_responses", "n_clusters", "denominator_scope",
          "ci_low", "ci_high", "confidence", "p_value", "p_adjusted", "statistical_test",
          "experiment_conditions", "interpretation_limit", "synthetic_test_fixture")


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _flat(value):
    return _json(value) if isinstance(value, (dict, list, tuple)) else value


def _base(evidence, *, rq, kind, source, status="observed"):
    row = {key: None for key in COMMON + METRICS}
    row.update(status=status, research_question=rq, source_kind=kind, source_table=source,
               experiment_conditions=_json(evidence.get("conditions", [])),
               synthetic_test_fixture=bool(evidence.get("synthetic_test_fixture", False)))
    return row


def _placeholder(evidence, rq, kind, reason):
    row = _base(evidence, rq=rq, kind=kind, source="unavailable", status="not_run")
    row.update(row_granularity="no empirical sample", denominator_scope="unavailable",
               interpretation_limit=reason)
    return row


def _native(evidence, tables, names, *, rq, kind, extra=None):
    result = []
    for name in names:
        rows = tables.get(name, [])
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("Native evidence tables must contain lists of mappings")
        for source in rows:
            row = _base(evidence, rq=rq, kind=kind, source=name)
            row.update({key: _flat(value) for key, value in source.items()})
            # A native analysis may have an unavailable contrast or undefined CI.
            # Keep that status rather than upgrading it to a completed test.
            row.setdefault("analysis_status", source.get("status"))
            row["status"] = ("unavailable" if "unavailable" in name or str(source.get("status", "")).startswith("unavailable") or source.get("status") in
                {"not_run", "unavailable", "undefined", "insufficient_data"} or
                (name in {"correlations","controlled_regressions"} and source.get("status")!="ok") else "observed")
            row["sample_size"] = source.get("n_pairs", source.get("n_planned_responses",
                source.get("n_responses", source.get("n_planned_units", source.get("n_planned",
                source.get("n_planned_pairs",source.get("spearman_n",source.get("n_rows"))))))))
            row["sample_size_valid_pairs"] = source.get("n_valid_pairs")
            row["sample_size_valid_responses"] = source.get("n_valid",source.get("n_valid_responses"))
            row["statistical_test"] = source.get("method")
            row["row_granularity"] = "native analysis stratum; no publication-layer pooling"
            row["denominator_scope"] = source.get("support_scope", source.get("analysis_support",
                "marginal accuracy: all planned; R/D/U/mean accuracy/invariance/kappa: joint-valid pairs"))
            if name=="answer_bias":
                row["denominator_scope"]="predicted YES share: valid responses; accuracy and invalid rate: all planned responses"
            if extra:
                row.update(extra)
            result.append(row)
    return result


def build_paper_tables(evidence):
    """Return seven lossless row collections, including explicit not-run rows.

    ``n`` always retains its native meaning (number of entities). Sample sizes
    use separate columns; invalid responses never disappear at this layer.
    Features measured without model responses are marked preprocessing only.
    """
    if not isinstance(evidence, dict) or type(evidence.get("contains_model_results")) is not bool:
        raise ValueError("Evidence must explicitly state whether model results exist")
    names = ("inference_tables", "reasoning_tables", "training_tables", "lexical_tables")
    if any(not isinstance(evidence.get(key, {}), dict) for key in names):
        raise ValueError("Evidence table collections must be mappings")
    if not evidence["contains_model_results"] and any(evidence.get(key) for key in names):
        raise ValueError("Empirical tables cannot be supplied as no-results evidence")
    output = {key: [] for key in TABLES}
    dataset = evidence.get("dataset_rows", [])
    if not isinstance(dataset, list) or any(not isinstance(row, dict) for row in dataset):
        raise ValueError("Dataset rows must be a list of mappings")
    for source in dataset:
        row = _base(evidence, rq="RQ1–RQ5 (design)", kind="dataset", source="frozen_dataset",
                    status="dataset_statistics")
        row.update({key: _flat(value) for key, value in source.items()})
        row.update(row_granularity="split × entity count; design counts only",
                   denominator_scope="frozen generated data; not empirical model performance",
                   interpretation_limit="Human linguistic review remains pending unless separately documented.")
        output[TABLES[0]].append(row)
    if not output[TABLES[0]]:
        output[TABLES[0]] = [_placeholder(evidence, "RQ1–RQ5 (design)", "dataset", "No verified dataset statistics supplied.")]

    inference = evidence.get("inference_tables", {})
    output[TABLES[1]] = _native(evidence, inference, ("profiles",), rq="RQ1–RQ3", kind="inference")
    output[TABLES[2]] = _native(evidence, inference,
        ("language_differences", "language_robustness", "language_presentation_interaction",
         "language_complexity_interaction", "unavailable_contrasts"), rq="RQ1", kind="inference")
    output[TABLES[3]] = _native(evidence, inference, ("profiles", "presentation_tests"),
        rq="RQ2", kind="inference", extra={"interpretation_limit":
            "P4/P5 may combine multiple transformations; do not infer a pure morphological effect. Human linguistic review is required."})
    lexical = evidence.get("lexical_tables", {})
    output[TABLES[4]] = _native(evidence, lexical,
        ("lexical_profiles", "answer_bias", "paired_direction", "correlations", "controlled_regressions"),
        rq="RQ4", kind="lexical", extra={"interpretation_limit":
            "Exploratory correlation; corpus frequency is not a causal intervention or a measurement of model parameters."})
    prep = evidence.get("dataset_metadata", {}).get("lexical_preprocessing", {})
    if prep:
        row = _base(evidence, rq="RQ4 (features only)", kind="lexical_preprocessing",
                    source="lexical_preprocessing_inventory", status="preprocessing_only")
        row.update(row_granularity="verified preprocessing inventory; no model outcomes",
            denominator_scope="feature coverage only", feature_inventory=_json({key:value for key,value in prep.items()
                if key not in {"corpus_frequencies","tokenization_vocabulary"}}),
            interpretation_limit="Frequency/tokenization features do not establish an accuracy, invariance, or answer-bias association.")
        output[TABLES[4]].append(row)
        for name in ("corpus_frequencies","tokenization_vocabulary"):
            rows=prep.get(name,[])
            if not isinstance(rows,list) or any(not isinstance(value,dict) for value in rows):
                raise ValueError("Preprocessing features must be lists of native records")
            for feature in rows:
                row=_base(evidence,rq="RQ4 (features only)",kind="lexical_preprocessing",source=name,status="preprocessing_only")
                row.update({key:_flat(value) for key,value in feature.items()})
                row.update(status="preprocessing_only",row_granularity="native lexical feature record; no model outcomes",
                    denominator_scope="independent corpus / tokenizer feature only",sample_size=None,sample_size_valid_pairs=None,
                    interpretation_limit="Preprocessing frequency and tokenizer measurements do not establish model performance or causal effects.")
                for metric in METRICS:
                    if metric!="frequency":row[metric]=None
                row.update(ci_low=None,ci_high=None,p_value=None,p_adjusted=None,statistical_test=None)
                output[TABLES[4]].append(row)
    output[TABLES[5]] = _native(evidence, inference,
        ("profiles", "endpoint_internal", "language_complexity_interaction"), rq="RQ3", kind="inference",
        extra={"interpretation_limit":"Endpoint distance is n−1; internal/distance strata remain separate. Do not interpret structural coupling as an isolated distance effect."})
    reasoning = evidence.get("reasoning_tables", {})
    output[TABLES[6]] = _native(evidence, reasoning, tuple(sorted(reasoning)), rq="RQ5", kind="reasoning",
        extra={"interpretation_limit":"Explicit reasoning prompts and provider-native reasoning effort are distinct interventions."})
    for transfer, tables in sorted(evidence.get("training_tables", {}).items()):
        if not isinstance(tables, dict):
            raise ValueError("Training evidence must be keyed by transfer cohort")
        output[TABLES[6]].extend(_native(evidence, tables, tuple(sorted(tables)), rq="RQ5", kind="training",
            extra={"transfer_cohort": transfer, "interpretation_limit":
                "Training-arm and transfer-cohort identities are protected; training budgets do not establish empirical success."}))
    reasons = {
        1: "No verified model evaluation; all model performance metrics are unavailable.",
        2: "No matched verified Turkish–English responses; paired effect, CI, and test unavailable.",
        3: "No verified paired presentation responses; robustness hypotheses remain untested.",
        4: "No verified model responses for lexical associations or prediction bias; preprocessing is not an experimental finding.",
        5: "No verified model responses across complexity and query-distance strata.",
        6: "No completed verified reasoning or optional training evaluation; intervention effects are unavailable.",
    }
    rqs = {1:"RQ1–RQ3", 2:"RQ1", 3:"RQ2", 4:"RQ4", 5:"RQ3", 6:"RQ5"}
    for index in range(1, 7):
        empirical = [row for row in output[TABLES[index]] if row["status"] == "observed"]
        if not empirical:
            output[TABLES[index]].append(_placeholder(evidence, rqs[index], "empirical_evaluation", reasons[index]))
    return output


def _cell(value):
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _latex(value):
    mapping = {"\\":r"\textbackslash{}", "&":r"\&", "%":r"\%", "$":r"\$", "#":r"\#",
        "_":r"\_\allowbreak{}", "{":r"\{", "}":r"\}", "~":r"\textasciitilde{}", "^":r"\textasciicircum{}",
        "<":r"\textless{}", ">":r"\textgreater{}", "\n":" ", "\r":" "}
    return "".join(mapping.get(char, char) for char in _cell(value))


def _markdown(value):
    return (_cell(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace("\\", "&#92;").replace("|", "&#124;").replace("`", "&#96;")
        .replace("\n", "<br>").replace("\r", ""))


def _compact(rows, dataset=False):
    """Readable companion view; full native records remain in the CSV."""
    if dataset:
        columns=("Stratum", "Split", "n", "Orders", "Queries", "Prompts", "Comparison cells", "Status")
        return columns, [{"Stratum":row["publication_row_id"], "Split":row.get("split"),"n":row.get("n"),
            "Orders":row.get("orders"),"Queries":row.get("queries"),"Prompts":row.get("prompts"),
            "Comparison cells":row.get("comparison_cells"),"Status":row.get("status")} for row in rows]
    columns=("Stratum", "Measure", "Estimate", "N planned / valid / clusters", "95% CI", "Test", "p / Holm p", "Status")
    view=[]
    metrics=METRICS+("all_planned_accuracy","all_planned_mean_accuracy","q","rho","coefficient",
        "direct_value","strategy_value","truncation_rate","mean_input_tokens","mean_output_tokens",
        "surface_count","lemma_count","surface_frequency_per_million","lemma_frequency_per_million",
        "term_tokens","inverse_term_tokens","forward_inverse_token_difference")
    for row in rows:
        support=row.get("sample_size_valid_pairs")
        if support is None:support=row.get("sample_size_valid_responses")
        sample_text=" / ".join(_cell(value) for value in (row.get("sample_size"),support,row.get("n_clusters")))
        observed=[metric for metric in metrics if row.get(metric) is not None]
        if not observed:observed=["not_available"]
        for metric in observed:
            low=row.get(metric+"_ci_low");high=row.get(metric+"_ci_high")
            interval_method=row.get("ci_method")
            if metric=="R" and row.get("R_wilson_low") is not None:
                low,high=row["R_wilson_low"],row.get("R_wilson_high")
                interval_method="Wilson; "+str(row.get("R_wilson_interpretation") or "independence unverified")
            if metric=="q" and row.get("q_wilson_low") is not None:
                low,high=row["q_wilson_low"],row.get("q_wilson_high")
                interval_method="Wilson; "+str(row.get("q_wilson_interpretation") or "independence unverified")
            if metric in {"difference","coefficient","rho"}:
                low=row.get("ci_low",low);high=row.get("ci_high",high)
            is_contrast=metric in {"difference","coefficient","rho"}
            interval=(None if low is None or high is None else f"[{low}, {high}]"+
                (" ("+str(interval_method).replace("_"," ")+")" if interval_method else " (method unavailable)"))
            view.append({"Stratum":row["publication_row_id"],"Measure":metric,"Estimate":row.get(metric),
                "N planned / valid / clusters":sample_text,
                "95% CI":interval,
                "Test":row.get("statistical_test") if is_contrast else None,
                "p / Holm p":" / ".join(_cell(row.get(key)) for key in ("p_value","p_adjusted")) if is_contrast else None,
                "Status":row.get("status")})
        coefficients=row.get("coefficients")
        if isinstance(coefficients,str):
            try:coefficients=json.loads(coefficients)
            except ValueError:coefficients=None
        if isinstance(coefficients,dict):
            for term,value in sorted(coefficients.items()):
                view.append({"Stratum":row["publication_row_id"],"Measure":"OLS: "+term,"Estimate":value,
                    "N planned / valid / clusters":sample_text,
                    "95% CI":None,"Test":None,"p / Holm p":None,"Status":row.get("status")})
    return columns,view


def publication_view(stem, rows):
    """Return compact companion rows for manuscript excerpts and exporters.

    Stable local row IDs resolve to the same table's full CSV records.
    """
    if stem not in TABLES or not isinstance(rows,list):
        raise ValueError("A registered table stem and row list are required")
    number=TABLES.index(stem)+1
    labeled=[{**row,"publication_row_id":f"t{number}-r{index:05d}"} for index,row in enumerate(rows,1)]
    _,view=_compact(labeled,dataset=number==1)
    return view


def write_exclusive(files):
    """All-or-nothing new-file publication, rejecting symlink traversal."""
    paths = [Path(path) for path, _ in files]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate output path")
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite report: {path}")
        aliases={Path("/var"):Path("/private/var"),Path("/tmp"):Path("/private/tmp")}
        if any(parent.is_symlink() and not (parent in aliases and parent.resolve()==aliases[parent])
               for parent in path.parents):
            raise ValueError("Report output must not traverse symlinks")
    created = []
    try:
        for path, data in files:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                created.append(path)
                handle.write(data)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return created


def export_paper_tables(tables, output_dir):
    """Export the seven tables as deterministic CSV, LaTeX, and Markdown.

    Cells retain full precision. NA denotes unavailable, never a zero score.
    LaTeX/Markdown provide compact metric-level review views. Their stratum IDs
    resolve to complete CSV records, retaining every model revision, design
    coordinate, condition, denominator, interval, and test-family annotation.
    """
    if not isinstance(tables, dict) or set(tables) != set(TABLES):
        raise ValueError("Exactly the seven registered paper tables are required")
    files = []
    output_dir = Path(output_dir)
    for index,name in enumerate(TABLES,1):
        source_rows = tables[name]
        rows = source_rows
        if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
            raise ValueError("Paper tables need rows or an explicit not-run placeholder")
        for row in rows:
            for key, value in row.items():
                if not isinstance(key, str) or not key:
                    raise ValueError("Column names must be nonempty strings")
                if value is not None and not isinstance(value, (str, bool, int, float)):
                    raise ValueError("Export requires scalar cells")
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError("Nonfinite table value")
        rows=[{**row,"publication_row_id":f"t{index}-r{number:05d}"} for number,row in enumerate(source_rows,1)]
        columns = list(dict.fromkeys(key for row in rows for key in row))
        out = io.StringIO(newline="")
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows([_cell(row.get(column)) for column in columns] for row in rows)
        synthetic = any(row.get("synthetic_test_fixture") for row in rows)
        label = "SYNTHETIC UNIT TEST ONLY; NOT MODEL RESULTS." if synthetic else "Verified evidence; NA means unavailable or undefined."
        compact_columns,compact_rows=_compact(rows,dataset=index==1)
        explanation="Stratum IDs resolve to publication_row_id in the companion CSV, which contains full conditions, model revisions, and native statistics. NA means unavailable; no significance is inferred."
        latex = ["% " + label, "% " + explanation, "% Requires \\usepackage{longtable} and a Unicode-capable LaTeX engine.",
                 "% Widths scale to the enclosing linewidth; longtable requires a single-column environment.",
                 r"\begingroup",r"\small",r"\setlength{\tabcolsep}{2pt}",
                 r"\begin{longtable}{@{}" + "".join(r"p{"+str(width)+r"\linewidth}" for width in
                    (.09,.14,.08,.14,.14,.10,.10,.11)) + "@{}}", r"\hline",
                 " & ".join(_latex(column) for column in compact_columns) + r" \\", r"\hline"]
        latex.extend(" & ".join(_latex(row.get(column)) for column in compact_columns) + r" \\" for row in compact_rows)
        latex.extend([r"\hline", r"\end{longtable}",r"\endgroup", ""])
        markdown = [label, "", explanation, "", "| " + " | ".join(_markdown(column) for column in compact_columns) + " |",
                    "| " + " | ".join("---" for _ in compact_columns) + " |"]
        markdown.extend("| " + " | ".join(_markdown(row.get(column)) for column in compact_columns) + " |" for row in compact_rows)
        files.extend([(output_dir / (name + ".csv"), out.getvalue().encode()),
                      (output_dir / (name + ".tex"), "\n".join(latex).encode()),
                      (output_dir / (name + ".md"), ("\n".join(markdown) + "\n").encode())])
    return write_exclusive(files)
