"""Matched base-versus-adapter effects, retaining invalids and latent clusters."""

from collections import defaultdict

from analysis.pipeline import DESIGN, MODEL, _contrast, _seed
from analysis.statistics import bootstrap, holm_adjust, mcnemar
from analysis.metrics import summarize_pairs
from inference.dataset import fingerprint

TRAINED_ARMS = ("sft_lora","paired_sft","sft_consistency","grpo_correctness","grpo_consistency")
KEYS = ("cluster_id","query_id","condition")


def _correct(row):
    return row["prediction"] is not None and row["prediction"] == row["gold_label"]


def training_contrasts(observations, *, seed=42, repeats=2000, declared_arms=TRAINED_ARMS):
    """Adapter minus base on identical planned coordinates.

    Conditional R/invariance/kappa effects use common-four-valid support and a
    bootstrap of *all planned* latent clusters. Arm-specific valid pair counts
    and all-planned accuracy/strict R/coverage effects accompany them. Missing
    baseline/arms are unavailable, never silently complete-case comparisons.
    Holm is across the declared trained arms within each model/design/metric
    contrast, not study-wide over all n/relations/distances.
    """
    if type(seed) is not int or seed<0 or type(repeats) is not int or repeats<1:
        raise ValueError("invalid paired-training bootstrap settings")
    declared_arms=tuple(declared_arms)
    if len(set(declared_arms))!=len(declared_arms) or any(a not in TRAINED_ARMS for a in declared_arms):
        raise ValueError("unregistered or duplicate declared training arm")
    groups=defaultdict(lambda:defaultdict(dict))
    for row in observations:
        arm=row.get("arm",row["model_id"].rsplit("--",1)[-1])
        if arm not in ("base",*TRAINED_ARMS) or not row["model_id"].endswith("--"+arm):
            raise ValueError("explicit training arm and model identity required")
        if type(row["gold_label"]) is not bool or (row["prediction"] is not None and type(row["prediction"]) is not bool):
            raise ValueError("Boolean gold and parsed Boolean/None predictions required")
        registry=row["model_id"][:-(len(arm)+2)]
        identity={"base_model_id":registry, "model_family":row["model_family"],"model_revision":row["model_revision"],
                  **{k:row[k] for k in DESIGN},"language":row["language"],"transfer":row.get("transfer")}
        group_key=tuple(identity.items()); key=tuple(row[k] for k in KEYS)
        if key in groups[group_key][arm]: raise ValueError("duplicate planned training coordinate")
        groups[group_key][arm][key]=row
    result=[]
    for raw_identity,arms in sorted(groups.items(),key=lambda item:str(item[0])):
        identity=dict(raw_identity); baseline=arms.get("base")
        for arm in declared_arms:
            trained=arms.get(arm)
            model=next(iter((trained or baseline or next(iter(arms.values()))).values()))
            metadata={**identity,**{k:model[k] for k in MODEL},"training_arm":arm,"baseline_arm":"base",
                      "baseline_model_id":identity["base_model_id"]+"--base",
                      "effect_direction":"trained_minus_base","bootstrap_replicates":repeats,
                      "holm_scope":"declared_training_arms_within_base_model_design_metric_not_study_wide"}
            if baseline is None or trained is None:
                result.append({**metadata,"metric":"all_intervention_effects","status":"unavailable_missing_arm",
                    "difference":None,"ci_low":None,"ci_high":None,"p_value":None,
                    "n_pairs":None,"n_clusters":None,"support_scope":"unavailable"})
                continue
            if set(baseline)!=set(trained): raise ValueError("training arms have unequal planned coordinates")
            for key in baseline:
                left,right=baseline[key],trained[key]
                if left["gold_label"]!=right["gold_label"] or any(left.get(k)!=right.get(k) for k in
                        ("query_subject","query_object","query_direction","query_relation")):
                    raise ValueError("paired training arms changed query or gold")
            conditions=sorted({key[-1] for key in baseline})
            if not {"canonical","equivalent"}<=set(conditions): raise ValueError("paired presentations required")
            for condition in conditions:
                rows=[]
                for key,left in sorted(baseline.items(),key=str):
                    if key[-1]!=condition:continue
                    right=trained[key]
                    a,b=_correct(left),_correct(right)
                    rows.append({"cluster_id":left["cluster_id"],"n":left["n"],"left":a,"right":b,"difference":int(b)-int(a)})
                tags={**metadata,"metric":"accuracy","condition":condition,"support_scope":"all_planned_invalid_wrong"}
                result.append(_contrast(rows,tags,seed,repeats))
            units=sorted({key[:2] for key in baseline},key=str)
            for condition in [c for c in conditions if c!="canonical"]:
                pairs=[]
                for cluster,query in units:
                    left=[baseline[cluster,query,c]["prediction"] for c in ("canonical",condition)]
                    right=[trained[cluster,query,c]["prediction"] for c in ("canonical",condition)]
                    gold=baseline[cluster,query,"canonical"]["gold_label"]
                    pairs.append({"cluster_id":cluster,"n":identity["n"],"gold":gold,
                        "base":left,"trained":right,"base_valid":all(x is not None for x in left),
                        "trained_valid":all(x is not None for x in right),
                        "common_four":all(x is not None for x in left+right)})
                support={"n_planned_pairs":len(pairs),"n_common_four_valid":sum(p["common_four"] for p in pairs),
                    "n_base_valid_pairs":sum(p["base_valid"] for p in pairs),
                    "n_trained_valid_pairs":sum(p["trained_valid"] for p in pairs)}
                for metric in ("strict_robust_correctness","valid_pair_coverage"):
                    rows=[]
                    for p in pairs:
                        if metric=="valid_pair_coverage":a,b=p["base_valid"],p["trained_valid"]
                        else:
                            a=p["base_valid"] and all(x==p["gold"] for x in p["base"])
                            b=p["trained_valid"] and all(x==p["gold"] for x in p["trained"])
                        rows.append({**p,"left":a,"right":b,"difference":int(b)-int(a)})
                    tags={**metadata,**support,"metric":metric,"comparison":condition,"support_scope":"all_planned_invalid_wrong"}
                    result.append(_contrast(rows,tags,seed,repeats))
                def summaries(sample):
                    valid=[p for p in sample if p["common_four"]]
                    return [summarize_pairs([{"gold":p["gold"],"original":p[a][0],"equivalent":p[a][1],"cluster_id":p["cluster_id"],"n":p["n"]} for p in valid])
                            for a in ("base","trained")]
                before,after=summaries(pairs)
                for metric in ("R","invariance","kappa"):
                    tags={**metadata,**support,"metric":metric,"comparison":condition,"support_scope":"common_four_valid_sensitivity"}
                    def statistic(sample, metric=metric):
                        a,b=summaries(sample);x,y=a[metric],b[metric]
                        return {"difference":None if x is None or y is None else y-x}
                    ci=bootstrap(pairs,statistic,seed=_seed(seed,tags),repeats=repeats,cluster_key="cluster_id",strata_keys=("n",))["difference"]
                    row={**tags,"difference":statistic(pairs)["difference"],"base_value":before[metric],"trained_value":after[metric],
                        "ci_low":ci["low"],"ci_high":ci["high"],"ci_method":"matched_instance_cluster_percentile",
                        "bootstrap_status":ci["status"],"n_pairs":len(pairs),"n_clusters":len({p["cluster_id"] for p in pairs}),
                        "p_value":None,"method":None,"status":"descriptive_conditional_bootstrap"}
                    if metric in ("R","invariance"):
                        valid=[p for p in pairs if p["common_four"]]
                        values=lambda p,a: all(x==p["gold"] for x in p[a]) if metric=="R" else p[a][0]==p[a][1]
                        test=mcnemar([values(p,"base") for p in valid],[values(p,"trained") for p in valid],clusters=[p["cluster_id"] for p in valid])
                        row.update({**{k:v for k,v in test.items() if k!="n"},"mcnemar_n":test["n"]})
                    result.append(row)
    families=defaultdict(list)
    for row in result:
        family={k:row.get(k) for k in ("base_model_id","model_family","model_revision",*DESIGN,"language","transfer","metric","condition","comparison","support_scope")}
        row["holm_family"]=fingerprint(family);families[row["holm_family"]].append(row)
    for rows in families.values():
        # Unobserved declared arms occupy a correction slot as well.
        padded={a:next((r["p_value"] for r in rows if r["training_arm"]==a),None) for a in declared_arms}
        adjusted=holm_adjust(padded)
        for row in rows:
            row.update(p_adjusted=adjusted[row["training_arm"]],holm_family_size=len(declared_arms))
    return result
