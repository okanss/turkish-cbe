"""Regenerate the P1 supplemental-evaluation figure (Figure 6).

Reads the P1 analysis tables produced by ``analysis.p1_statistics`` over the
supplemental quantized local evaluation (Ollama Q4_K_M; never pooled with
registered-provider results). Writes ``fig6_p1_quantized_scaling.pdf`` and a
plot manifest (``p1_plot_manifest.json``) recording source and asset SHA-256
hashes, mirroring ``generate_figures.py``.

Run from the project root:
    python3 paper/figures/manuscript/generate_p1_figures.py
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent

# Validated categorical palette, fixed order by model scale (never cycled).
MODEL_ORDER = ["qwen2.5:0.5b", "qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b"]
MODEL_LABELS = {"qwen2.5:0.5b": "0.5B", "qwen2.5:3b": "3B",
                "qwen2.5:7b": "7B", "qwen2.5:14b": "14B"}
MODEL_COLORS = {"qwen2.5:0.5b": "#eda100", "qwen2.5:3b": "#1baf7a",
                "qwen2.5:7b": "#eb6834", "qwen2.5:14b": "#2a78d6"}
FORWARD = "#2a78d6"   # English, as in Figures 4-5
INVERSE = "#eb6834"   # Turkish, as in Figures 4-5
TEXT = "#333333"
MUTED = "#777777"
GRID = "#d9d9d9"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": TEXT,
        "axes.linewidth": 0.6,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "pdf.fonttype": 42,
    }
)

SOURCES = {
    "p1_accuracy_by_axis": ROOT / "results/p1/analysis/p1_accuracy_by_axis.csv",
    "p1_paired_tests": ROOT / "results/p1/analysis/p1_paired_tests.csv",
    "p1_coverage": ROOT / "results/p1/analysis/p1_coverage.csv",
}

# Paired F2 contrasts, each against the Turkish canonical presentation.
F2_COMPARISONS = ["F2_tr_canonical_vs_equivalent", "F2_tr_canonical_vs_P1",
                  "F2_tr_canonical_vs_P2", "F2_tr_canonical_vs_P3",
                  "F2_tr_canonical_vs_P4", "F2_tr_canonical_vs_P5"]
F2_LABELS = ["Equiv.", "P1", "P2", "P3", "P4", "P5"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_accuracy() -> list[dict]:
    with SOURCES["p1_accuracy_by_axis"].open() as fh:
        return list(csv.DictReader(fh))


def load_paired_tests() -> dict[tuple[str, str], dict]:
    with SOURCES["p1_paired_tests"].open() as fh:
        return {(r["model"], r["comparison"]): r for r in csv.DictReader(fh)}


def _value(rows, model, axis, language, value) -> float:
    for r in rows:
        if (r["model"] == model and r["axis"] == axis
                and r["language"] == language and r["value"] == value):
            return float(r["all_planned_accuracy"])
    raise KeyError((model, axis, language, value))


def _interval(rows, model, axis, language, value) -> tuple[float, float]:
    for r in rows:
        if (r["model"] == model and r["axis"] == axis
                and r["language"] == language and r["value"] == value):
            return float(r["wilson_low"]), float(r["wilson_high"])
    raise KeyError((model, axis, language, value))


def fig6_p1_quantized_scaling() -> Path:
    rows = load_accuracy()

    fig, axes = plt.subplots(
        1, 3, figsize=(6.4, 2.45), gridspec_kw={"width_ratios": [1.0, 1.35, 1.0]}
    )

    # (a) accuracy by model scale and language.
    ax = axes[0]
    x = range(len(MODEL_ORDER))
    for language, color, label, marker in (
            ("en", FORWARD, "EN", "o"), ("tr", INVERSE, "TR", "s")):
        acc = [_value(rows, m, "language", language, language) for m in MODEL_ORDER]
        bounds = [_interval(rows, m, "language", language, language) for m in MODEL_ORDER]
        ax.fill_between(x, [b[0] for b in bounds], [b[1] for b in bounds],
                        color=color, alpha=0.15, linewidth=0)
        ax.plot(x, acc, color=color, linewidth=1.6, marker=marker, markersize=4.2)
        ax.annotate(label, (len(MODEL_ORDER) - 1, acc[-1]), textcoords="offset points",
                    xytext=(5, 0), va="center", fontsize=7.4, color=color)
    ax.axhline(0.5, color=MUTED, linewidth=0.7, linestyle=":")
    ax.set_xticks(list(x))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_xlim(-0.3, len(MODEL_ORDER) - 0.4)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("Model scale (Q4_K_M)", fontsize=8.5)
    ax.set_ylabel("All-planned accuracy", fontsize=8.5)
    ax.set_title("(a) By scale and language", fontsize=8.5, color=TEXT, loc="left")

    # (b) Paired presentation contrasts vs Turkish canonical (query-matched
    # delta with latent-cluster bootstrap CI), one series per model. The
    # paired estimand avoids the strata-mix confound of pooled accuracies.
    ax = axes[1]
    tests = load_paired_tests()
    xc = range(len(F2_COMPARISONS))
    offsets = {m: (i - 1.5) * 0.17 for i, m in enumerate(MODEL_ORDER)}
    for model in MODEL_ORDER:
        xs, deltas, lows, highs = [], [], [], []
        for i, comparison in enumerate(F2_COMPARISONS):
            row = tests[(model, comparison)]
            if not row["delta_accuracy"]:
                continue
            xs.append(i + offsets[model])
            deltas.append(float(row["delta_accuracy"]))
            lows.append(float(row["delta_boot_low"]))
            highs.append(float(row["delta_boot_high"]))
        ax.errorbar(xs, deltas,
                    yerr=[[d - lo for d, lo in zip(deltas, lows)],
                          [hi - d for d, hi in zip(deltas, highs)]],
                    color=MODEL_COLORS[model], fmt="o", markersize=3.2,
                    linewidth=0, elinewidth=1.0, capsize=1.6,
                    label=MODEL_LABELS[model])
    ax.axhline(0.0, color=MUTED, linewidth=0.7, linestyle=":")
    ax.set_xticks(list(xc))
    ax.set_xticklabels(F2_LABELS, fontsize=7.2)
    ax.set_ylim(-0.46, 0.12)
    ax.set_xlabel("Presentation vs. TR canonical", fontsize=8.5)
    ax.set_ylabel("$\\Delta$ paired accuracy", fontsize=8.5)
    ax.set_title("(b) Paired presentation effects", fontsize=8.5, color=TEXT, loc="left")
    ax.legend(fontsize=6.8, ncol=2, loc="lower left", handlelength=1.0,
              columnspacing=0.8, borderaxespad=0.1)

    # (c) qwen2.5:14b accuracy by entity count, both languages (mirrors Fig. 5b).
    ax = axes[2]
    sizes = sorted({int(r["value"]) for r in rows if r["axis"] == "n"})
    xs = range(len(sizes))
    for language, color, label, marker in (
            ("en", FORWARD, "EN", "o"), ("tr", INVERSE, "TR", "s")):
        acc = [_value(rows, "qwen2.5:14b", "n", language, str(n)) for n in sizes]
        ax.plot(xs, acc, color=color, linewidth=1.6, marker=marker, markersize=4.2)
        ax.annotate(label, (len(sizes) - 1, acc[-1]), textcoords="offset points",
                    xytext=(5, 0), va="center", fontsize=7.4, color=color)
    ax.axhline(0.5, color=MUTED, linewidth=0.7, linestyle=":")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([str(n) for n in sizes])
    ax.set_xlim(-0.3, len(sizes) - 0.4)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("Entity count, $n$", fontsize=8.5)
    ax.set_title("(c) 14B by entity count", fontsize=8.5, color=TEXT, loc="left")

    for ax in axes:
        ax.yaxis.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.tight_layout(w_pad=1.6)
    out = OUT / "fig6_p1_quantized_scaling.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    assets = [fig6_p1_quantized_scaling()]
    manifest = {
        "sources": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
                    for name, path in SOURCES.items()},
        "assets": {path.name: sha256(path) for path in assets},
        "note": ("P1 supplemental quantized local evaluation (Ollama Q4_K_M); "
                 "exploratory pending the two-reviewer linguistic review; never "
                 "pooled with registered-provider results."),
    }
    (OUT / "p1_plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for path in assets:
        print(path)


if __name__ == "__main__":
    main()
