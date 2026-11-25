"""Analyze the revision follow-up (Table 14): direct vs. structured reasoning.

Direct-arm records are read from the existing sweep file (identical messages);
structured-arm records from the follow-up run. All-planned estimand (invalid =
wrong), latent-order cluster bootstrap from analysis.statistics (2000 reps,
size strata), exactly as for the main contrasts.

    PYTHONPATH=src python3 scripts/analyze_followup.py \
        --direct results/p1/qwen2-5-14b.jsonl \
        --structured results/followup/qwen2-5-14b_structured.jsonl
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, "src")
from analysis.p1_statistics import _pairs
from analysis.statistics import bootstrap

MAN = Path("data/manifests/followup_reasoning_manifest.jsonl")

def load(path, ids):
    recs = {}
    for line in open(path, encoding="utf-8"):
        x = json.loads(line)
        if x.get("record_kind") == "inference" and x["prompt_id"] in ids:
            recs[x["prompt_id"]] = x
    return recs

def allplanned(rows):
    c0 = sum(1 for r in rows if r["original"] is not None and r["original"] == r["gold"])
    c1 = sum(1 for r in rows if r["equivalent"] is not None and r["equivalent"] == r["gold"])
    return {"d": (c1 - c0) / len(rows)}

def arm(name, recs, man):
    out = {"arm": name, "n_records": len(recs), "n_planned": len(man)}
    missing = len(man) - len(recs)
    inv = sum(1 for p in man if p["prompt_id"] not in recs or not recs[p["prompt_id"]]["parse_valid"])
    out["invalid_pct"] = 100 * inv / len(man)
    out["missing"] = missing
    for key, (lang, cond) in {"EN_P0": ("en", "canonical"), "TR_P0": ("tr", "canonical"), "TR_P3": ("tr", "P3")}.items():
        rows = [p for p in man if p["language"] == lang and p["condition"] == cond]
        out[key] = 100 * sum(1 for p in rows if p["prompt_id"] in recs and recs[p["prompt_id"]].get("correct")) / len(rows)
        for n in (4, 16, 64):
            rn = [p for p in rows if p["n"] == n]
            out[f"{key}_n{n}"] = 100 * sum(1 for p in rn if p["prompt_id"] in recs and recs[p["prompt_id"]].get("correct")) / len(rn)
    for key, left, right in (("d_P3_P0", ("tr", "canonical", "native_shared"), ("tr", "P3", "native_shared")),
                             ("d_TR_EN", ("en", "canonical", "native_shared"), ("tr", "canonical", "native_shared"))):
        pairs = _pairs(man, recs, left, right)
        b = bootstrap(pairs, allplanned)["d"]
        out[key] = (b["estimate"], b["low"], b["high"], len(pairs))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direct", required=True)
    ap.add_argument("--structured", default=None)
    ap.add_argument("--output", default="results/followup/followup_summary.json")
    a = ap.parse_args()
    man = [json.loads(l) for l in MAN.open(encoding="utf-8")]
    ids = {r["prompt_id"] for r in man}
    results = [arm("direct", load(a.direct, ids), man)]
    if a.structured:
        results.append(arm("structured", load(a.structured, ids), man))
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(results, indent=1))
    for r in results:
        f = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"
        print(f"{r['arm']:>10}: invalid {r['invalid_pct']:.2f}% | EN P0 {r['EN_P0']:.1f} | TR P0 {r['TR_P0']:.1f} | "
              f"TR P3 {r['TR_P3']:.1f} | dP3-P0 {f(r['d_P3_P0'])} | dTR-EN {f(r['d_TR_EN'])}")
        print("            by n (TR P0/TR P3/EN P0): " + "; ".join(
            f"n={n}: {r[f'TR_P0_n{n}']:.1f}/{r[f'TR_P3_n{n}']:.1f}/{r[f'EN_P0_n{n}']:.1f}" for n in (4, 16, 64)))
main()
