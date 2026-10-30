"""Lossless tables and optional publication figures for observed analysis tables.

The exporter never aggregates input rows. A line joins only rows that differ in
the plotted axis (and, where shown, language). Every other profile dimension is
fixed in a separate file, except explicitly labeled endpoint distance = n−1.
Synthetic figures are available only through an
explicit, visibly watermarked option for tests and demonstrations.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
from typing import Any


REPORTING_VERSION = "0.6.0"
_SAFE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]*\Z")
_RATE_METRICS = {
    "original_accuracy", "equivalent_accuracy", "original_accuracy_valid",
    "equivalent_accuracy_valid", "mean_accuracy", "R", "D", "U",
    "invariance", "raw_invariance", "kappa", "invalid_response_rate",
    "original_invalid_response_rate", "equivalent_invalid_response_rate",
    "valid_pair_coverage", "strict_robust_correctness", "difference",
    "accuracy_difference", "q", "I_expected", "expected_invariance",
}
_SUMMARY_COLUMNS = {
    "total", "n_pairs", "n_valid_pairs", "n_original_valid", "n_equivalent_valid",
    "n_clusters", "n_invalid_original", "n_invalid_equivalent", "n_invalid_responses",
    "org_only", "eq_only", "undefined_reasons",
    "bootstrap_seed", "bootstrap_replicates", "bootstrap_method", "ci_method",
    "confidence_intervals", "R_interval_method", "R_wilson_low", "R_wilson_high",
    "R_wilson_method", "R_wilson_interpretation", "R_wilson_status",
    "total_pairs", "valid_pairs", "common_valid_pairs", "original_valid",
    "equivalent_valid", "original_correct", "equivalent_correct", "both_correct",
    "both_wrong", "discordant", "pooled_valid_answers", "pooled_yes_answers",
    "original_invalid", "equivalent_invalid", "invalid_responses", "responses",
    "n_instances", "n_latent_instances", "count", "bootstrap_replicates",
    "ci_low", "ci_high", "p_value", "p_adjusted",
    "b", "c", "mcnemar_n", "method", "status", "statistic", "reason",
    "holm_family", "holm_family_size",
    "input_tokens", "mean_input_tokens", "median_input_tokens", "min_input_tokens",
    "max_input_tokens", "output_tokens", "mean_output_tokens", "token_length",
}


def _rows(tables: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(tables, dict):
        raise ValueError("tables must be a mapping of safe names to row lists")
    result: dict[str, list[dict[str, Any]]] = {}
    for name, rows in tables.items():
        if not isinstance(name, str) or not _SAFE_NAME.fullmatch(name):
            raise ValueError(f"Unsafe table name: {name!r}")
        if not isinstance(rows, list):
            raise ValueError(f"Table {name} must contain a list of rows")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"Table {name} contains a non-mapping row")
            for key, value in row.items():
                if not isinstance(key, str) or not key:
                    raise ValueError("Column names must be nonempty strings")
                if value is not None and not isinstance(value, (str, bool, int, float)):
                    raise ValueError(f"Unsupported cell type in {name}.{key}")
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError(f"Nonfinite value in {name}.{key}")
        result[name] = rows
    return result


def _cell(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _latex(value: Any) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
        "<": r"\textless{}", ">": r"\textgreater{}", "\n": " ", "\r": " ",
    }
    return "".join(replacements.get(char, char) for char in _cell(value))


def _write_exclusive(files: list[tuple[Path, bytes]]) -> list[Path]:
    """Write new files only; roll back our own files if any write fails."""
    for path, _ in files:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite an existing report: {path}")
    created: list[Path] = []
    try:
        for path, data in files:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                created.append(path)
                handle.write(data)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return created


def export_tables(tables: dict[str, list[dict[str, Any]]], outputdir: str | Path) -> list[Path]:
    """Export each table as UTF-8 CSV and a standalone LaTeX ``tabular``.

    Columns follow their first appearance across the supplied rows. Missing and
    undefined cells are ``NA``; numeric values are not rounded. Input row order
    is preserved. Empty tables have a zero-byte CSV and a labeled empty tabular.
    All inputs and filename collisions are checked before any output is written.
    """
    clean = _rows(tables)
    outputdir = Path(outputdir)
    files: list[tuple[Path, bytes]] = []
    for name in sorted(clean):
        rows = clean[name]
        columns = list(dict.fromkeys(key for row in rows for key in row))
        csv_text = io.StringIO(newline="")
        if columns:
            writer = csv.writer(csv_text, lineterminator="\n")
            writer.writerow(columns)
            writer.writerows([_cell(row.get(column)) for column in columns] for row in rows)
        latex_lines = ["% Turkish-CBE analysis table; NA means unavailable/undefined."]
        latex_lines += [r"\begin{tabular}{" + "l" * max(1, len(columns)) + "}", r"\hline"]
        if columns:
            latex_lines += [" & ".join(_latex(column) for column in columns) + r" \\", r"\hline"]
            latex_lines += [" & ".join(_latex(row.get(column)) for column in columns) + r" \\" for row in rows]
        else:
            latex_lines += [r"\textit{No observations} \\"]
        latex_lines += [r"\hline", r"\end{tabular}", ""]
        files += [(outputdir / f"{name}.csv", csv_text.getvalue().encode("utf-8")),
                  (outputdir / f"{name}.tex", "\n".join(latex_lines).encode("utf-8"))]
    return _write_exclusive(files)


def _is_summary(key: str) -> bool:
    if key in _RATE_METRICS or key in _SUMMARY_COLUMNS:
        return True
    if key.startswith("bootstrap_"):
        return True
    if key.endswith(("_ci_low", "_ci_high")):
        return True
    for suffix in ("_ci_low", "_ci_high", "_low", "_high", "_count", "_se"):
        if key.endswith(suffix):
            return key[:-len(suffix)] in _RATE_METRICS or suffix in ("_count", "_se")
    return False


def _identity(row: dict[str, Any], excluded: set[str]) -> tuple[tuple[str, Any], ...]:
    # Unknown fields remain protected dimensions. This can produce more facets,
    # but cannot silently pool an unrecognized experimental factor.
    return tuple((key, value) for key, value in sorted(row.items())
                 if key not in excluded and not _is_summary(key))


def _number(value: Any, name: str, *, rate: bool = False) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite numeric data or None")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be finite numeric data or None") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite numeric data or None")
    if rate and not 0 <= value <= 1:
        raise ValueError(f"{name} must be a probability in [0, 1]")
    return number


def _x(row: dict[str, Any], axis: str) -> int:
    value = row.get(axis)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{axis} must be a positive integer")
    return value


def _ci(row: dict[str, Any], metric: str, value: float) -> tuple[float, float] | None:
    pairs = [(f"{metric}_ci_low", f"{metric}_ci_high"),
             (f"{metric}_low", f"{metric}_high")]
    if metric == "R" and row.get("R_wilson_interpretation") != "descriptive_dependent_pairs":
        pairs.insert(0, ("R_wilson_low", "R_wilson_high"))
    if metric == "difference":
        pairs.insert(0, ("ci_low", "ci_high"))
    for low_name, high_name in pairs:
        if low_name in row or high_name in row:
            low = _number(row.get(low_name), low_name)
            high = _number(row.get(high_name), high_name)
            if low is None and high is None:
                return None
            if low is None or high is None or low > high:
                raise ValueError(f"Invalid interval for {metric}")
            return low, high
    return None


def _groups(rows: list[dict[str, Any]], axis: str, *, language_series: bool) -> list[tuple[tuple, list[dict]]]:
    excluded = {axis}
    if language_series:
        excluded.add("language")
    grouped: dict[tuple, list[dict]] = {}
    seen: set[tuple] = set()
    for row in rows:
        value = _x(row, axis)
        row_excluded = set(excluded)
        if row.get("query_type") == "endpoint":
            if _x(row, "query_distance") != _x(row, "n") - 1:
                raise ValueError("Endpoint profiles require query_distance = n - 1")
            if axis == "n":
                row_excluded.add("query_distance")
        identity = _identity(row, row_excluded)
        observation = (identity, value, row.get("language") if language_series else None)
        if observation in seen:
            raise ValueError("Duplicate profile coordinates; aggregate with an explicit statistical estimand first")
        seen.add(observation)
        grouped.setdefault(identity, []).append(row)
    return sorted(grouped.items(), key=lambda pair: json.dumps(pair[0], ensure_ascii=False))


def _figure_jobs(tables: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    profiles = tables.get("profiles", [])
    specifications = (
        (1, "accuracy_vs_n", "n", ("original_accuracy", "equivalent_accuracy"), True),
        (2, "invariance_vs_n", "n", ("invariance",), True),
        (3, "robust_correctness_vs_n", "n", ("R",), True),
        (4, "rdu_decomposition", "n", ("R", "D", "U"), False),
        (6, "query_distance_vs_accuracy", "query_distance", ("original_accuracy", "equivalent_accuracy"), True),
    )
    for number, label, axis, metrics, language_series in specifications:
        relevant = [row for row in profiles if any(row.get(metric) is not None for metric in metrics)]
        for identity, rows in _groups(relevant, axis, language_series=language_series):
            for row in rows:
                for metric in metrics:
                    value = _number(row.get(metric), metric, rate=True)
                    if value is not None:
                        _ci(row, metric, value)
                if number == 4:
                    values = [row.get(metric) for metric in metrics]
                    if any(value is None for value in values) or not math.isclose(sum(values), 1, abs_tol=1e-9):
                        raise ValueError("R/D/U decomposition requires complete probabilities summing to one")
            jobs.append(dict(number=number, label=label, axis=axis, metrics=metrics,
                             identity=identity, rows=rows, language_series=language_series))
    differences = [row for row in tables.get("language_differences", []) if row.get("difference") is not None]
    for identity, rows in _groups(differences, "n", language_series=False):
        for row in rows:
            value = _number(row["difference"], "difference")
            if not -1 <= value <= 1:
                raise ValueError("Paired accuracy differences must lie in [-1, 1]")
            _ci(row, "difference", value)
        jobs.append(dict(number=5, label="turkish_minus_english", axis="n", metrics=("difference",),
                         identity=identity, rows=rows, language_series=False))
    return sorted(jobs, key=lambda job: (job["number"], json.dumps(job["identity"], ensure_ascii=False)))


def _stem(job: dict[str, Any]) -> str:
    identity = json.dumps(job["identity"], ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"figure_{job['number']}_{job['label']}_{digest}"


def _title(job: dict[str, Any]) -> str:
    fields = [f"{key}={_cell(value)}" for key, value in job["identity"]]
    heading = job["label"].replace("_", " ").capitalize()
    lines: list[str] = []
    current = ""
    for field in fields:
        if current and len(current) + len(field) > 100:
            lines.append(current)
            current = ""
        current += ("; " if current else "") + field
    if current:
        lines.append(current)
    if job["axis"] == "n" and dict(job["identity"]).get("query_type") == "endpoint":
        lines.append("Endpoint profile: query distance = n − 1 (structurally coupled to complexity)")
    if job["number"] != 4:
        method_key = "R_interval_method" if job["number"] == 3 else "ci_method"
        methods = sorted({str(row[method_key]) for row in job["rows"] if row.get(method_key)})
        if job["number"] == 3 and any(row.get("R_wilson_interpretation")=="descriptive_dependent_pairs" for row in job["rows"]):
            methods=["matched-instance cluster bootstrap; dependent Wilson intervals are descriptive only"]
        if not methods and job["number"] == 3 and any(row.get("R_wilson_low") is not None for row in job["rows"]):
            methods = ["Wilson"]
        if methods:
            lines.append("Confidence intervals: " + "; ".join(methods))
        if job["number"] == 3:
            statuses = sorted({str(row["R_wilson_status"]) for row in job["rows"] if row.get("R_wilson_status")})
            if statuses:
                lines.append("R interval status: " + "; ".join(statuses))
    return heading + ("\n" + "\n".join(lines) if lines else "")


def _draw(figure: Any, job: dict[str, Any], synthetic: bool) -> None:
    axis = figure.add_subplot(1, 1, 1)
    rows = sorted(job["rows"], key=lambda row: (_x(row, job["axis"]), str(row.get("language", ""))))
    labels = {"original_accuracy": "Original accuracy", "equivalent_accuracy": "Equivalent accuracy",
              "invariance": "Raw invariance", "R": "Robust correctness (R)", "D": "Shared wrong (D)",
              "U": "Presentation dependence (U)", "difference": "TR − EN paired accuracy"}
    if job["number"] == 4:
        x_values = list(range(len(rows)))
        bottoms = [0.0] * len(rows)
        for metric, color in zip(job["metrics"], ("#0072B2", "#D55E00", "#E69F00")):
            values = [float(row[metric]) for row in rows]
            axis.bar(x_values, values, bottom=bottoms, label=labels[metric], color=color, width=.7)
            bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
        axis.set_xticks(x_values, [str(_x(row, "n")) for row in rows])
    else:
        languages = sorted({str(row.get("language", "")) for row in rows}) if job["language_series"] else [None]
        series_index = 0
        colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
        markers = ("o", "s", "^", "D", "v", "P")
        styles = ("-", "--", "-.", ":")
        for language in languages:
            language_rows = [row for row in rows if language is None or str(row.get("language", "")) == language]
            for metric in job["metrics"]:
                observed = [row for row in language_rows if row.get(metric) is not None]
                if not observed:
                    continue
                xs = [_x(row, job["axis"]) for row in observed]
                ys = [float(row[metric]) for row in observed]
                series_label = (f"{language.upper()} · " if language else "") + labels[metric]
                line, = axis.plot(xs, ys, marker=markers[series_index % len(markers)],
                                  linestyle=styles[series_index % len(styles)],
                                  color=colors[series_index % len(colors)],
                                  linewidth=1.5, label=series_label)
                series_index += 1
                for x_value, y_value, row in zip(xs, ys, observed):
                    interval = _ci(row, metric, y_value)
                    if interval:
                        center = (interval[0] + interval[1]) / 2
                        axis.errorbar([x_value], [center],
                                      yerr=[[center - interval[0]], [interval[1] - center]],
                                      fmt="none", capsize=3, color=line.get_color())
        if len({_x(row, job["axis"]) for row in rows}) > 1:
            axis.set_xscale("log", base=2)
        axis.set_xticks(sorted({_x(row, job["axis"]) for row in rows}),
                        [str(value) for value in sorted({_x(row, job["axis"]) for row in rows})])
    axis.set_ylim((-1, 1) if job["number"] == 5 else (0, 1))
    if job["number"] == 5:
        axis.axhline(0, color="#777777", linewidth=.8, linestyle="--")
    axis.set_xlabel("Latent query distance" if job["axis"] == "query_distance" else "Number of entities (n)")
    axis.set_ylabel("TR − EN accuracy difference" if job["number"] == 5 else "Probability")
    axis.set_title(_title(job), fontsize=9, loc="left")
    axis.grid(axis="y", alpha=.25)
    axis.set_axisbelow(True)
    axis.legend(loc="best", fontsize=8)
    if synthetic:
        figure.text(.5, .5, "SYNTHETIC TEST DATA\nNOT MODEL RESULTS", ha="center", va="center",
                    fontsize=25, color="#A00000", alpha=.25, rotation=25)
    figure.tight_layout()


def export_figures(tables: dict[str, list[dict[str, Any]]], outputdir: str | Path, *,
                   synthetic: bool = False) -> list[Path]:
    """Export available six figure families as PDF/SVG/PNG without pooling.

    Matplotlib is optional and imported only when nonempty figure data exist.
    Figure filenames include a stable hash of all protected dimensions; the
    dimensions are also visible in each title and metadata. A distance plot fixes
    ``n``. Endpoint complexity plots validate distance = n−1 and explicitly label
    this structural coupling; other complexity plots keep distance fixed.
    No observations means no figures, rather than fabricated zero scores.
    Output paths must not already exist. Explicit synthetic mode watermarks every
    format and its metadata; it never generates synthetic observations.
    """
    if not isinstance(synthetic, bool):
        raise ValueError("synthetic must be boolean")
    clean = _rows(tables)
    jobs = _figure_jobs(clean)
    if not jobs:
        return []
    outputdir = Path(outputdir)
    paths = [outputdir / f"{_stem(job)}.{extension}" for job in jobs for extension in ("pdf", "svg", "png")]
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite an existing report: {path}")
    try:
        import matplotlib
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise RuntimeError("Figure export requires the optional analysis plotting dependencies") from exc
    created: list[Path] = []
    try:
        with matplotlib.rc_context({"svg.fonttype": "none", "svg.hashsalt": "turkish-cbe-analysis-v1",
                                    "font.family": "DejaVu Sans", "text.usetex": False}):
            for job in jobs:
                figure = Figure(figsize=(9, 6), dpi=150)
                FigureCanvasAgg(figure)
                try:
                    _draw(figure, job, synthetic)
                    title = _title(job)
                    description = "SYNTHETIC TEST DATA; NOT MODEL RESULTS" if synthetic else "Observed analysis table; no aggregation performed by exporter"
                    for extension in ("pdf", "svg", "png"):
                        if extension == "pdf":
                            metadata = {"Creator": "Turkish-CBE", "Title": title, "Subject": description,
                                        "CreationDate": None, "ModDate": None}
                        elif extension == "svg":
                            metadata = {"Creator": "Turkish-CBE", "Title": title,
                                        "Description": description, "Date": None}
                        else:
                            metadata = {"Software": "Turkish-CBE", "Title": title, "Description": description}
                        buffer = io.BytesIO()
                        figure.savefig(buffer, format=extension, metadata=metadata, dpi=150)
                        created.extend(_write_exclusive([(outputdir / f"{_stem(job)}.{extension}", buffer.getvalue())]))
                finally:
                    figure.clear()
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return created
