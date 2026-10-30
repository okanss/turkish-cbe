"""Scientific diagrams and observed-data-only publication figures.

Architecture diagrams describe the method, not an empirical performance result.
Missing outcome panels are textual placeholders rather than fabricated charts.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import tempfile

from .tables import TABLES, write_exclusive


PLACEHOLDERS = {
    "figure_3_accuracy_robustness_profiles": "Verified matched model profiles with planned/valid denominators and uncertainty are required.",
    "figure_4_turkish_english_difference": "Verified matched Turkish–English response contrasts and their intervals are required.",
    "figure_5_rdu_distribution": "Verified common-valid paired R/D/U profiles are required; R+D+U must equal one.",
    "figure_6_relation_vocabulary_bias": "Verified forward/inverse predicted-answer shares, gold balance, and response denominators are required. Corpus/tokenization preprocessing alone is insufficient.",
}


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _formats(figure, stem, title, description):
    files = []
    for extension in ("pdf", "svg", "png"):
        if extension == "pdf":
            metadata = {"Creator":"Turkish-CBE", "Title":title, "Subject":description,
                        "CreationDate":None, "ModDate":None}
        elif extension == "svg":
            metadata = {"Creator":"Turkish-CBE", "Title":title, "Description":description, "Date":None}
        else:
            metadata = {"Software":"Turkish-CBE", "Title":title, "Description":description}
        buffer = io.BytesIO()
        figure.savefig(buffer, format=extension, metadata=metadata, dpi=150)
        files.append((Path(str(stem) + "." + extension), buffer.getvalue()))
    return files


def _box(axis, xy, width, height, text, *, color="#E8F1F8"):
    from matplotlib.patches import FancyBboxPatch
    x, y = xy
    patch = FancyBboxPatch((x,y), width,height,boxstyle="round,pad=0.012,rounding_size=0.02",
                          facecolor=color, edgecolor="#315B75", linewidth=1.2)
    axis.add_patch(patch)
    axis.text(x+width/2,y+height/2,text,ha="center",va="center",fontsize=10,linespacing=1.45)


def _arrow(axis, begin, end):
    axis.annotate("", xy=end,xytext=begin,arrowprops={"arrowstyle":"->","color":"#315B75","lw":1.4})


def _diagram(kind, Figure, FigureCanvasAgg):
    figure = Figure(figsize=(11, 6), dpi=150)
    FigureCanvasAgg(figure)
    axis = figure.add_axes((.025,.04,.95,.90))
    axis.set_xlim(0,1);axis.set_ylim(0,1);axis.axis("off")
    if kind == "architecture":
        title = "Turkish-CBE architecture"
        axis.text(.5,.97,title,ha="center",va="top",fontsize=16,weight="bold")
        _box(axis,(.02,.41),.21,.21,"Language-independent\nstrict total order\n+ exact oracle")
        _box(axis,(.33,.65),.29,.18,"Matched English presentations\ncanonical / equivalent")
        _box(axis,(.33,.22),.29,.24,"Matched Turkish presentations\ncanonical / P1–P5\nlinguistic transformations\nhuman review pending")
        _box(axis,(.73,.41),.24,.21,"Versioned model evaluation\nstrict answer parsing\nimmutable raw responses")
        _arrow(axis,(.23,.52),(.33,.74));_arrow(axis,(.23,.52),(.33,.34))
        _arrow(axis,(.62,.74),(.73,.52));_arrow(axis,(.62,.34),(.73,.52))
        _box(axis,(.33,.01),.64,.13,"Matched-pair CBE analysis: R, D, U, invariance, κ, coverage\nAll-planned marginal accuracy; joint-valid pair metrics",color="#EAF3EA")
        _arrow(axis,(.85,.41),(.85,.14))
        axis.text(.04,.12,"Same latent ID, query,\ngold label, and split",fontsize=10,ha="left",va="center")
    else:
        title = "Dataset generation and validation pipeline"
        axis.text(.5,.97,title,ha="center",va="top",fontsize=16,weight="bold")
        labels = ["Seeded latent generator\nunique entities + total order", "Adjacent-edge premises\nendpoint / internal / distance queries",
            "Exact oracle + invariant checks\nacyclicity, distance, gold balance", "Latent-ID split assignment\ntraining / pilot / validation / test",
            "Matched TR / EN rendering\nname controls + presentation factors", "Semantic normalization + hashes\nhuman review gate: pending"]
        positions=[(.025,.69),(.37,.69),(.715,.69),(.715,.29),(.37,.29),(.025,.29)]
        for position,label in zip(positions,labels):_box(axis,position,.26,.20,label)
        _arrow(axis,(.285,.79),(.37,.79));_arrow(axis,(.63,.79),(.715,.79))
        _arrow(axis,(.845,.69),(.845,.49));_arrow(axis,(.715,.39),(.63,.39));_arrow(axis,(.37,.39),(.285,.39))
        axis.text(.5,.10,"Design diagram only. Automated semantic checks do not replace human linguistic review.",ha="center",fontsize=10)
    return figure,title


def _bias_jobs(rows):
    """One bar per native stratum; never pool directions or dependent rows."""
    jobs=[]
    for row in rows:
        value = row.get("predicted_yes_rate", row.get("predicted_yes_share", row.get("q")))
        if value is None:
            continue
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Predicted YES share must be a finite rate")
        denominator = row.get("n_valid", row.get("n_valid_responses", row.get("valid_responses")))
        if type(denominator) is not int or denominator <= 0:
            continue
        identity={key:value for key,value in row.items() if key not in {
            "predicted_yes_rate","predicted_yes_share","q","n_yes","n_no","n_valid","n_valid_responses","valid_responses",
            "predicted_yes_ci_low","predicted_yes_ci_high","yes_wilson_low","yes_wilson_high"}}
        digest=hashlib.sha256(_canonical(identity).encode()).hexdigest()[:16]
        jobs.append((digest,row,float(value),denominator,identity))
    if len({job[0] for job in jobs}) != len(jobs):
        raise ValueError("Duplicate lexical bias coordinates; select a single verified cohort")
    return jobs


def _cbe_profiles(evidence):
    profiles,differences=[],[]
    sources=[("inference",None,evidence.get("inference_tables",{})),
             ("reasoning",None,evidence.get("reasoning_tables",{}))]
    sources.extend(("training",transfer,tables) for transfer,tables in
                   sorted(evidence.get("training_tables",{}).items()))
    for kind,transfer,tables in sources:
        extra={"publication_source":kind}
        if transfer is not None:extra["transfer_cohort"]=transfer
        profiles.extend({**row,**extra} for row in tables.get("profiles",[]))
        differences.extend({**row,**extra} for row in tables.get("language_differences",[]))
    return profiles,differences


def export_paper_figures(evidence, tables, output_dir, *, synthetic=False):
    """Export architecture/pipeline and genuine empirical panels only.

    Synthetic fixtures require an explicit test-only switch and watermark.
    All output files are new; failures roll back this invocation's outputs.
    CBE plotting reuses the established exporter whose facets protect model,
    relation, query type/distance, name control, language, and intervention.
    """
    if type(synthetic) is not bool or type(evidence.get("contains_model_results")) is not bool:
        raise ValueError("Explicit Boolean evidence/test provenance is required")
    if bool(evidence.get("synthetic_test_fixture")) != synthetic:
        raise ValueError("Synthetic fixture provenance and explicit export switch must agree")
    if set(tables) != set(TABLES):
        raise ValueError("Seven paper tables are required")
    try:
        import matplotlib
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise RuntimeError("Scientific figures require the pinned analysis plotting environment") from exc
    output_dir=Path(output_dir)
    files=[]
    with matplotlib.rc_context({"svg.fonttype":"none", "svg.hashsalt":"turkish-cbe-publication-v1",
                                "font.family":"DejaVu Sans", "text.usetex":False}):
        for index,kind in enumerate(("architecture","generation_pipeline"),1):
            figure,title=_diagram(kind,Figure,FigureCanvasAgg)
            try:
                files.extend(_formats(figure,output_dir/f"figure_{index}_{kind}",title,
                    "Methodological diagram; not experimental results. Human review status is separate."))
            finally:figure.clear()
        present={key:False for key in PLACEHOLDERS}
        if evidence["contains_model_results"]:
            from analysis.reporting import export_figures
            profiles,differences=_cbe_profiles(evidence)
            with tempfile.TemporaryDirectory(prefix="turkish-cbe-paper-figures-") as scratch:
                generated=export_figures({"profiles":profiles,"language_differences":differences},scratch,synthetic=synthetic)
                for path in generated:
                    files.append((output_dir/"observed_profiles"/path.name,path.read_bytes()))
                present["figure_3_accuracy_robustness_profiles"]=any("accuracy_vs_n" in p.name or "robust_correctness_vs_n" in p.name for p in generated)
                present["figure_4_turkish_english_difference"]=any("turkish_minus_english" in p.name for p in generated)
                present["figure_5_rdu_distribution"]=any("rdu_decomposition" in p.name for p in generated)
            for digest,row,value,denominator,identity in _bias_jobs(evidence.get("lexical_tables",{}).get("answer_bias",[])):
                figure=Figure(figsize=(9,6),dpi=150);FigureCanvasAgg(figure)
                try:
                    axis=figure.add_subplot(1,1,1)
                    label=str(row.get("query_relation",row.get("direction","lexical stratum")))
                    axis.bar([0],[value],width=.5,color="#0072B2")
                    lower=row.get("q_wilson_low",row.get("predicted_yes_ci_low"))
                    upper=row.get("q_wilson_high",row.get("predicted_yes_ci_high"))
                    if lower is not None or upper is not None:
                        if lower is None or upper is None or not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in (lower,upper)) or not 0 <= lower <= upper <= 1:
                            raise ValueError("Invalid predicted-answer confidence interval")
                        center=(lower+upper)/2
                        axis.errorbar([0],[center],yerr=[[center-lower],[upper-center]],fmt="none",capsize=4,color="#111111")
                    axis.set_xticks([0],[label]);axis.set_ylim(0,1)
                    axis.set_ylabel("Predicted YES share among valid responses")
                    axis.axhline(.5,color="#777777",linestyle="--",linewidth=.8)
                    axis.set_title("Relation vocabulary answer bias",loc="left")
                    caption="; ".join(f"{key}={identity[key]}" for key in ("model_id","model_revision","model_family",
                        "split","n","query_type","query_distance","relation_id","relation_family","name_control",
                        "language","condition","direction_axis","query_relation","query_direction",
                        "gold_yes_planned","gold_no_planned","n_planned","n_valid","q_wilson_interpretation") if key in identity)
                    import textwrap
                    figure.text(.06,.02,"\n".join(textwrap.wrap(caption,125)),fontsize=7,va="bottom")
                    axis.text(.5,.98,f"N valid={denominator}; gold distribution retained in source table",transform=axis.transAxes,ha="center",va="top",fontsize=9)
                    if synthetic:figure.text(.5,.5,"SYNTHETIC TEST DATA\nNOT MODEL RESULTS",ha="center",rotation=25,fontsize=22,color="#A00000",alpha=.25)
                    figure.subplots_adjust(bottom=.30)
                    files.extend(_formats(figure,output_dir/f"figure_6_relation_vocabulary_bias_{digest}",
                        "Relation vocabulary answer bias",("SYNTHETIC TEST ONLY; " if synthetic else "")+caption))
                    present["figure_6_relation_vocabulary_bias"]=True
                finally:figure.clear()
        for name,available in present.items():
            if not available:
                text=f"# {name.replace('_',' ').capitalize()}\n\nStatus: NOT RUN / NO VERIFIED COMPATIBLE EMPIRICAL DATA.\n\n{PLACEHOLDERS[name]}\n\nNo score, interval, effect, or significance result has been fabricated.\n"
                files.append((output_dir/(name+".md"),text.encode()))
    return write_exclusive(files)
