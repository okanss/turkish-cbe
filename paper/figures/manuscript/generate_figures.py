"""Regenerate the quantitative manuscript figures (Figures 2-5).

Figures 2-4 read the retained dataset-statistics or lexical-publication
files and contain no model outcome. Figure 5 reads the normalized pilot
run records (exploratory pilot evaluation; see paper Section 6). A plot
manifest records source SHA-256 hashes and generated asset hashes (see
paper Appendix C).

Run from the project root:
    python3 paper/figures/manuscript/generate_figures.py
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent

# Validated categorical palette (colorblind-safe in fixed order).
SPLIT_COLORS = {
    "training": "#2a78d6",
    "development": "#eb6834",
    "pilot": "#1baf7a",
    "test": "#eda100",
}
SPLIT_LABELS = {
    "training": "Training",
    "development": "Development",
    "pilot": "Pilot",
    "test": "Frozen test",
}
FORWARD = "#2a78d6"
INVERSE = "#eb6834"
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
    "dataset_statistics": ROOT / "paper/tables/table_1_dataset_statistics.md",
    "corpus_frequencies": ROOT / "results/lexical/with_llama/tables/corpus_frequencies.csv",
    "tokenization_vocabulary": ROOT / "results/lexical/with_llama/tables/tokenization_vocabulary.csv",
    "pilot_gpt4o_normalized": ROOT / "results/normalized/pilot-gpt4o-main-003.jsonl",
    "pilot_qwen_normalized": ROOT / "results/normalized/pilot-qwen05b-main-001.jsonl",
}

RELATION_ORDER = [
    ("spatial_front", "Front / back"),
    ("spatial_right", "Right / left"),
    ("spatial_above", "Above / below"),
    ("temporal_arrival", "Arrival"),
    ("temporal_completion", "Completion"),
    ("comparison_height", "Height"),
    ("comparison_speed", "Speed"),
    ("comparison_weight", "Weight"),
    ("ranking", "Rank"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dataset_statistics() -> list[dict]:
    rows = []
    for line in SOURCES["dataset_statistics"].read_text().splitlines():
        line = line.strip()
        if not line.startswith("|") or line.startswith("| ---") or line.startswith("| Stratum"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(
            {
                "split": cells[1],
                "n": int(cells[2]),
                "orders": int(cells[3]),
                "queries": int(cells[4]),
                "prompts": int(cells[5]),
            }
        )
    return rows


def fig2_construction_counts() -> Path:
    rows = load_dataset_statistics()
    sizes = sorted({r["n"] for r in rows})
    splits = ["training", "development", "pilot", "test"]
    panels = [
        ("orders", "Source orders", 1),
        ("queries", "Queries ($10^3$)", 1e3),
        ("prompts", "Prompt identities ($10^3$)", 1e3),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(6.4, 5.2), sharex=True)
    x = range(len(sizes))
    for ax, (key, label, scale) in zip(axes, panels):
        bottom = [0.0] * len(sizes)
        raw_totals = [0] * len(sizes)
        for split in splits:
            by_n = {r["n"]: r[key] for r in rows if r["split"] == split}
            vals = [by_n.get(n, 0) / scale for n in sizes]
            raw_totals = [t + by_n.get(n, 0) for t, n in zip(raw_totals, sizes)]
            ax.bar(
                x,
                vals,
                bottom=bottom,
                width=0.62,
                color=SPLIT_COLORS[split],
                edgecolor="white",
                linewidth=0.8,
                label=SPLIT_LABELS[split],
            )
            bottom = [b + v for b, v in zip(bottom, vals)]
        for xi, total, raw in zip(x, bottom, raw_totals):
            ax.annotate(
                f"{raw:,.0f}".replace(",", " "),
                (xi, total),
                textcoords="offset points",
                xytext=(0, 2),
                ha="center",
                va="bottom",
                fontsize = 6.8,
                color=MUTED,
            )
        ax.set_ylabel(label, fontsize=8.5)
        ax.set_ylim(0, max(bottom) * 1.16)
        ax.yaxis.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[-1].set_xticks(list(x))
    axes[-1].set_xticklabels([str(n) for n in sizes])
    axes[-1].set_xlabel("Entity count, $n$ (categorical spacing)")
    handles = [Patch(facecolor=SPLIT_COLORS[s], label=SPLIT_LABELS[s]) for s in splits]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.015),
        fontsize=8.5,
        handlelength=1.2,
        handleheight=1.0,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = OUT / "fig2_construction_counts.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def load_corpus_frequencies() -> dict[str, dict[str, dict]]:
    """relation_id -> direction -> row (retained BOUN lemma measurements)."""
    data: dict[str, dict[str, dict]] = {}
    with SOURCES["corpus_frequencies"].open() as fh:
        for row in csv.DictReader(fh):
            data.setdefault(row["relation_id"], {})[row["direction"]] = row
    return data


def fig3_lemma_frequencies() -> Path:
    data = load_corpus_frequencies()
    labels, fwd, inv, fwd_lemma, inv_lemma = [], [], [], [], []
    for rel_id, label in RELATION_ORDER:
        d = data[rel_id]
        f, i = d["before"], d["after"]
        labels.append(label)
        fwd.append(float(f["lemma_frequency_per_million"]))
        inv.append(float(i["lemma_frequency_per_million"]))
        fwd_lemma.append(f["lemma"])
        inv_lemma.append(i["lemma"])

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    y = list(range(len(labels)))[::-1]
    for yi, fv, iv in zip(y, fwd, inv):
        ax.plot([fv, iv], [yi, yi], color=GRID, linewidth=1.4, zorder=1)
    ax.scatter(fwd, y, s=34, color=FORWARD, zorder=3, label="Forward lemma")
    ax.scatter(inv, y, s=34, color=INVERSE, marker="s", zorder=3, label="Inverse lemma")
    for yi, fv, iv, fl, il in zip(y, fwd, inv, fwd_lemma, inv_lemma):
        f_right = fv >= iv
        ax.annotate(
            fl, (fv, yi), textcoords="offset points",
            xytext=(7 if f_right else -7, -0.5),
            ha="left" if f_right else "right",
            va="center", fontsize=7, color=MUTED,
        )
        ax.annotate(
            il, (iv, yi), textcoords="offset points",
            xytext=(-7 if f_right else 7, -0.5),
            ha="right" if f_right else "left",
            va="center", fontsize=7, color=MUTED,
        )
    ax.set_xscale("log")
    ax.set_xlim(70, 6000)
    ax.set_xticks([100, 300, 1000, 3000])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.75, len(labels) - 0.25)
    ax.set_xlabel("Lemma frequency per million eligible words (log scale)")
    ax.xaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.12), ncol=2, fontsize=8.5)
    ax.text(
        0.0,
        -0.26,
        "All 18 canonical surface forms: observed count = 0. "
        "Log surface frequencies remain undefined; no smoothing is applied.",
        transform=ax.transAxes,
        fontsize=7.2,
        color=MUTED,
    )
    fig.tight_layout()
    out = OUT / "fig3_lemma_frequencies.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def load_token_ratios() -> dict[str, dict[tuple[str, str], float | None]]:
    """config label -> (relation_id, direction) -> TR/EN ratio or None."""
    configs: dict[str, dict[tuple[str, str], float | None]] = {}
    with SOURCES["tokenization_vocabulary"].open() as fh:
        for row in csv.DictReader(fh):
            cfg = row["model_id"]
            key = (row["relation_id"], row["direction"])
            ratio = row.get("term_tr_en_ratio", "")
            value = None if ratio in ("", "NA") else float(ratio)
            configs.setdefault(cfg, {})[key] = value
    return configs


def fig4_token_ratio_heatmap() -> Path:
    configs = load_token_ratios()
    col_spec = [
        ("openai-gpt4o-snapshot", "GPT-4o\no200k_base"),
        ("qwen25-05b-hf", "Qwen 0.5B\nHF"),
        ("qwen25-15b-hf", "Qwen 1.5B\nHF"),
        ("qwen25-15b-vllm", "Qwen 1.5B\nvLLM"),
        ("llama32-1b-hf", "Llama 1B\nHF"),
        ("llama32-3b-hf", "Llama 3B\nHF"),
    ]

    rows = []
    for rel_id, label in RELATION_ORDER:
        rows.append((rel_id, "before", f"{label} · F"))
        rows.append((rel_id, "after", f"{label} · I"))

    cmap = LinearSegmentedColormap.from_list("blues", ["#eaf1fb", "#2a78d6", "#123a6b"])
    vmin, vmax = 0.5, 6.0

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    for yi, (rel_id, direction, _) in enumerate(rows):
        for xi, (cfg, _) in enumerate(col_spec):
            value = configs.get(cfg, {}).get((rel_id, direction)) if cfg else None
            if value is None:
                ax.add_patch(
                    plt.Rectangle(
                        (xi, yi), 1, 1, facecolor="#f2f2f0",
                        edgecolor="white", linewidth=1.5, hatch="///",
                    )
                )
                ax.text(xi + 0.5, yi + 0.5, "—", ha="center", va="center",
                        fontsize=7.5, color=MUTED)
            else:
                t = (min(max(value, vmin), vmax) - vmin) / (vmax - vmin)
                color = cmap(t)
                ax.add_patch(
                    plt.Rectangle((xi, yi), 1, 1, facecolor=color,
                                  edgecolor="white", linewidth=1.5)
                )
                lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
                txt = "white" if lum < 0.55 else TEXT
                ax.text(xi + 0.5, yi + 0.5, f"{round(value, 2):g}", ha="center",
                        va="center", fontsize=7.2, color=txt)
    ax.set_xlim(0, len(col_spec))
    ax.set_ylim(len(rows), 0)
    ax.set_xticks([i + 0.5 for i in range(len(col_spec))])
    ax.set_xticklabels([lbl for _, lbl in col_spec], fontsize=7.8)
    ax.set_yticks([i + 0.5 for i in range(len(rows))])
    ax.set_yticklabels([lbl for _, _, lbl in rows], fontsize=7.8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        "Matched isolated-term token count: Turkish / English",
        fontsize=9, color=TEXT, loc="left", pad=10,
    )
    ax.text(
        0.0,
        -0.075,
        "F: forward; I: inverse. Darker blue indicates a larger token ratio.\n"
        "The two Llama configurations share one tokenizer, as do the three Qwen\n"
        "configurations. Isolated strings only; no model weights or model responses.",
        transform=ax.transAxes,
        fontsize=7.2,
        color=MUTED,
        va="top",
    )
    fig.tight_layout()
    out = OUT / "fig4_token_ratios.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def load_pilot_gpt4o() -> list[dict]:
    rows = []
    with SOURCES["pilot_gpt4o_normalized"].open() as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows


def _accuracy(rows: list[dict], predicate) -> float:
    sel = [r for r in rows if predicate(r)]
    return sum(1 for r in sel if r["correct"]) / len(sel)


def fig5_pilot_gpt4o() -> Path:
    rows = load_pilot_gpt4o()
    tr = [r for r in rows if r["source"]["language"] == "tr"]
    en = [r for r in rows if r["source"]["language"] == "en"]

    conditions = ["canonical", "equivalent", "P1", "P2", "P3", "P4", "P5"]
    cond_labels = ["P0\ncan.", "Equiv.", "P1", "P2", "P3", "P4", "P5"]
    tr_by_cond = [
        _accuracy(tr, lambda r, c=c: r["source"]["condition"] == c) for c in conditions
    ]
    en_pooled = _accuracy(en, lambda r: True)

    sizes = sorted({r["source"]["n"] for r in rows})
    en_by_n = [_accuracy(en, lambda r, n=n: r["source"]["n"] == n) for n in sizes]
    tr_by_n = [_accuracy(tr, lambda r, n=n: r["source"]["n"] == n) for n in sizes]

    dc = [r for r in rows if r["source"]["query_type"] == "distance_controlled"]
    distances = sorted({r["source"]["query_distance"] for r in dc})
    acc_by_d = [
        _accuracy(dc, lambda r, d=d: r["source"]["query_distance"] == d) for d in distances
    ]

    fig, axes = plt.subplots(
        1, 3, figsize=(6.4, 2.45), gridspec_kw={"width_ratios": [1.55, 1.0, 1.0]}
    )

    ax = axes[0]
    x = range(len(conditions))
    ax.bar(x, tr_by_cond, width=0.62, color=INVERSE, edgecolor="white", linewidth=0.8)
    for xi, v in zip(x, tr_by_cond):
        ax.annotate(
            f"{100 * v:.1f}", (xi, v), textcoords="offset points", xytext=(0, -3),
            ha="center", va="top", fontsize=6.6, color="white",
        )
    ax.axhline(en_pooled, color=FORWARD, linewidth=0.9, linestyle=(0, (4, 2)))
    ax.annotate(
        "EN pooled 72.0", (len(conditions) - 0.45, en_pooled),
        textcoords="offset points", xytext=(0, 2), ha="right", va="bottom",
        fontsize=6.6, color=FORWARD,
    )
    ax.axhline(0.5, color=MUTED, linewidth=0.7, linestyle=":")
    ax.set_xticks(list(x))
    ax.set_xticklabels(cond_labels, fontsize=7.2)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("All-planned accuracy", fontsize=8.5)
    ax.set_title("(a) Turkish, by presentation", fontsize=8.5, color=TEXT, loc="left")

    ax = axes[1]
    xs = range(len(sizes))
    ax.plot(xs, en_by_n, color=FORWARD, linewidth=1.6, marker="o", markersize=4.4)
    ax.plot(xs, tr_by_n, color=INVERSE, linewidth=1.6, marker="s", markersize=4.2)
    ax.annotate("EN", (len(sizes) - 1, en_by_n[-1]), textcoords="offset points",
                xytext=(5, 0), va="center", fontsize=7.4, color=FORWARD)
    ax.annotate("TR", (len(sizes) - 1, tr_by_n[-1]), textcoords="offset points",
                xytext=(5, -3), va="center", fontsize=7.4, color=INVERSE)
    ax.axhline(0.5, color=MUTED, linewidth=0.7, linestyle=":")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([str(n) for n in sizes])
    ax.set_xlim(-0.3, len(sizes) - 0.45)
    ax.set_ylim(0.3, 1.05)
    ax.set_xlabel("Entity count, $n$", fontsize=8.5)
    ax.set_title("(b) By entity count", fontsize=8.5, color=TEXT, loc="left")

    ax = axes[2]
    xs = range(len(distances))
    ax.plot(xs, acc_by_d, color="#1baf7a", linewidth=1.6, marker="o", markersize=4.4)
    for xi, v in zip(xs, acc_by_d):
        ax.annotate(
            f"{100 * v:.1f}", (xi, v), textcoords="offset points", xytext=(0, 4),
            ha="center", va="bottom", fontsize=6.6, color=MUTED,
        )
    ax.axhline(0.5, color=MUTED, linewidth=0.7, linestyle=":")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([str(d) for d in distances])
    ax.set_ylim(0.3, 1.05)
    ax.set_xlabel("Query distance, $d$", fontsize=8.5)
    ax.set_title("(c) Distance-controlled", fontsize=8.5, color=TEXT, loc="left")

    for ax in axes:
        ax.yaxis.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.tight_layout(w_pad=1.6)
    out = OUT / "fig5_pilot_gpt4o.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    outputs = [
        fig2_construction_counts(),
        fig3_lemma_frequencies(),
        fig4_token_ratio_heatmap(),
        fig5_pilot_gpt4o(),
    ]
    manifest = {
        "sources": {k: {"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for k, p in SOURCES.items()},
        "outputs": {p.name: sha256(p) for p in outputs},
        "note": (
            "Figures 2-4: construction/preprocessing measurements only. "
            "Figure 5: exploratory pilot model outcomes from the normalized "
            "run records (pilot split; human linguistic review pending)."
        ),
    }
    (OUT / "plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote:", ", ".join(p.name for p in outputs))


if __name__ == "__main__":
    main()
