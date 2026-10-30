"""Build new auditable paper/report directories from native experiment evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import tempfile

from . import PUBLICATION_VERSION
from .evidence import collect_evidence
from .figures import export_paper_figures
from .manuscript import write_manuscript
from .tables import build_paper_tables, export_paper_tables

RQ_TABLES = {
    "RQ1": (3, "Matched Turkish-English performance"),
    "RQ2": (4, "Turkish morphosyntactic presentation robustness"),
    "RQ3": (6, "Complexity and inference distance"),
    "RQ4": (5, "Lexical frequency, tokenization and answer bias"),
    "RQ5": (7, "Reasoning and optional training interventions"),
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _source_hashes(root):
    active = Path(__file__).resolve().parent
    sources = sorted(active.glob("*.py"))
    project = root / "reports/publication"
    if {p.name for p in sources} != {p.name for p in project.glob("*.py")}:
        raise ValueError("installed publication package differs from project source inventory")
    for path in sources:
        if _sha(path) != _sha(project / path.name):
            raise ValueError("installed publication source differs from project: " + path.name)
    files = sorted((root / "src").rglob("*.py")) + sorted(project.glob("*.py"))
    files += [root / "pyproject.toml"]
    return {p.relative_to(root).as_posix(): _sha(p) for p in files}


def _destination(root, value, default, kind):
    path = (root / default) if value is None else Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("publication destinations cannot traverse symlinks")
    if not path.is_relative_to(root) or ".." in path.parts or path == root:
        raise ValueError("publication destination must stay inside the project")
    protected = ("src", "tests", "configs", "data", "logs", "tokenizers", "vendor", "results", "analysis", "reports/publication")
    if any(path.is_relative_to(root / item) or (root / item).is_relative_to(path) for item in protected):
        raise ValueError("publication destination overlaps protected input or code")
    if kind == "report" and not path.is_relative_to(root / "reports"):
        raise ValueError("status reports must stay in reports/")
    if path.exists():
        raise FileExistsError("publication output already exists; choose a new directory")
    return path


def rq_coverage(evidence, tables):
    rows = []
    for rq, (number, objective) in RQ_TABLES.items():
        keys = [k for k in tables if k.startswith(f"table_{number}_")]
        if len(keys) != 1:
            raise ValueError("exactly one table is required for " + rq)
        stem = keys[0]
        observed = [r for r in tables[stem] if r.get("status") == "observed"]
        # Observed rows identify available descriptive evidence; they do not
        # establish hypothesis support, complete power or a significant result.
        available = bool(evidence["contains_model_results"] and observed)
        rows.append({"research_question": rq, "objective": objective, "table": stem,
            "status": "evidence_available_review_required" if available else "results_pending",
            "observed_table_rows": len(observed) if available else 0,
            "hypothesis_supported": None,
            "interpretation": "No automated hypothesis endorsement; inspect matched N, coverage, uncertainty and adjusted tests." if available else
                "No validated corresponding model outcomes; no effect estimate, interval or significance claim."})
    return rows


def _check_inputs(root, sources):
    if not isinstance(sources, dict) or not sources:
        raise ValueError("publication requires explicit hashed input evidence")
    for relative, expected in sources.items():
        path = root / relative
        if (type(relative) is not str or Path(relative).is_absolute() or ".." in Path(relative).parts
                or any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file()
                or type(expected) is not str or _sha(path) != expected):
            raise ValueError("publication input changed or escaped the project: " + str(relative))


def build_report(root, *, paper_dir=None, report_dir=None, inference_runs=None,
                 reasoning_runs=None, training_evaluations=None, training_runs=None,
                 seed=20261025, repeats=2000):
    root = Path(root).resolve()
    if type(seed) is not int or seed < 0 or type(repeats) is not int or repeats != 2000:
        raise ValueError("publication requires a nonnegative seed and 2000 matched-instance bootstrap repeats")
    paper = _destination(root, paper_dir, "paper", "paper")
    report = _destination(root, report_dir, "reports/current", "report")
    if paper.is_relative_to(report) or report.is_relative_to(paper):
        raise ValueError("paper and report destinations must be disjoint")
    sources = _source_hashes(root)
    evidence = collect_evidence(root, inference_runs=inference_runs, reasoning_runs=reasoning_runs,
        training_evaluations=training_evaluations, training_runs=training_runs, seed=seed, repeats=repeats)
    _check_inputs(root, evidence["sources"])
    tables = build_paper_tables(evidence)
    coverage = rq_coverage(evidence, tables)
    paper.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".publication-", dir=report.parent))
    temporary_paper, temporary_report = temporary / "paper", temporary / "report"
    temporary_paper.mkdir(); temporary_report.mkdir()
    published_paper = False
    try:
        export_paper_tables(tables, temporary_paper / "tables")
        export_paper_figures(evidence, tables, temporary_paper / "figures")
        write_manuscript(evidence, tables, temporary_paper)
        (temporary_report / "evidence.json").write_text(_canonical(evidence)+"\n", encoding="utf-8")
        with (temporary_report / "rq_traceability.csv").open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(coverage[0]))
            writer.writeheader(); writer.writerows(coverage)
        status = "empirical_report_review_required" if evidence["contains_model_results"] else "results_pending"
        lines = ["# Turkish-CBE scientific reporting status", "", f"Status: **{status}**.", "",
            "This draft is not evidence of an executed model experiment or journal readiness.",
            "Dataset counts and architecture are distinguishable from model performance.", "",
            "| Research question | Table | Evidence status |", "| --- | --- | --- |"]
        lines += [f"| {r['research_question']} | {r['table']} | {r['status']} |" for r in coverage]
        lines += ["", "Effect estimates, confidence intervals and tests remain unavailable where results are pending.",
            "No hypothesis is automatically marked as supported.", "", "## Outstanding conditions", ""]
        lines += ["- "+str(item) for item in evidence["limitations"]]
        (temporary_report / "REPORT_STATUS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
        # Reject evidence or source changes during rendering, before publishing.
        _check_inputs(root, evidence["sources"])
        if _source_hashes(root) != sources:
            raise ValueError("publication sources changed during reporting")
        outputs = []
        for directory, destination in ((temporary_paper, paper), (temporary_report, report)):
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    relative = destination / path.relative_to(directory)
                    outputs.append({"path": relative.relative_to(root).as_posix(),
                                    "bytes": path.stat().st_size, "sha256": _sha(path)})
        config = {"seed": seed, "bootstrap_repeats": repeats, "confidence": .95,
            "cluster": "latent_instance_id", "strata": ["n"],
            "inference_runs": inference_runs, "reasoning_runs": reasoning_runs,
            "training_evaluations": None if training_evaluations is None else [str(p) for p in training_evaluations],
            "training_runs": None if training_runs is None else [str(p) for p in training_runs],
            "paper_dir": paper.relative_to(root).as_posix(), "report_dir": report.relative_to(root).as_posix()}
        packages = {}
        for name in ("matplotlib", "numpy"):
            try: packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: packages[name] = None
        manifest = {"artifact_type": "scientific_publication_report", "publication_version": PUBLICATION_VERSION,
            "status": status, "contains_model_results": evidence["contains_model_results"],
            "eligible_for_confirmatory_evaluation": evidence["eligible_for_confirmatory_evaluation"],
            "submission_ready": False, "config": config,
            "config_sha256": hashlib.sha256(_canonical(config).encode()).hexdigest(),
            "environment": {"python": platform.python_version(), "platform": platform.platform(), "packages": packages},
            "input_files_sha256": evidence["sources"], "source_files_sha256": sources,
            "availability": evidence["availability"], "conditions": evidence["conditions"],
            "rq_coverage": coverage, "table_rows": {k: len(v) for k, v in tables.items()},
            "files": outputs, "limitations": evidence["limitations"],
            "actual_generation_calls_by_reporting": 0, "actual_optimizer_steps_by_reporting": 0}
        (temporary_report / "report_manifest.json").write_text(_canonical(manifest)+"\n", encoding="utf-8")
        if paper.exists() or report.exists():
            raise FileExistsError("publication destination appeared concurrently")
        temporary_paper.rename(paper); published_paper = True
        try:
            if report.exists(): raise FileExistsError("report destination appeared concurrently")
            temporary_report.rename(report)
        except BaseException:
            # Only our newly published paper is removed on a failed paired commit.
            # Its files have not been exposed as a completed report manifest.
            if published_paper: shutil.rmtree(paper)
            raise
        return manifest
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--paper-dir", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--inference-run", nargs="+", dest="inference_runs")
    parser.add_argument("--reasoning-run", nargs="+", dest="reasoning_runs")
    parser.add_argument("--training-evaluation", nargs="+", dest="training_evaluations")
    parser.add_argument("--training-run", nargs="+", dest="training_runs")
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args(argv)
    try:
        result = build_report(args.project_root, paper_dir=args.paper_dir, report_dir=args.report_dir,
            inference_runs=args.inference_runs, reasoning_runs=args.reasoning_runs,
            training_evaluations=args.training_evaluations, training_runs=args.training_runs, seed=args.seed)
    except (ValueError, OSError, KeyError, ImportError) as error:
        print("error: "+str(error), file=sys.stderr)
        return 2
    print(_canonical({key: result[key] for key in ("status", "contains_model_results", "submission_ready", "table_rows")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
