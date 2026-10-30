"""Synthetic audit regressions; no model runs or human decisions are produced."""

from copy import deepcopy
import unittest
from test_analysis_pipeline import observation
from training.comparison import training_contrasts
from analysis.reporting import _ci


def sample(*, arms=("base","sft_lora"), clusters=6, queries=1):
    return [observation(f"fixture-{i}",lang,condition,prediction=arm!="base",gold=True,
                model_id="fixture-model--"+arm,model_spec_sha256=("a" if arm=="base" else "b")*64,
                arm=arm,query_id=f"fixture-{i}-q{q}",transfer="in_domain")
            for arm in arms for i in range(clusters) for q in range(queries)
            for lang in ("tr","en") for condition in ("canonical","equivalent")]


def find(rows, **conditions):
    matching=[r for r in rows if all(r.get(k)==v for k,v in conditions.items())]
    if len(matching)!=1:raise AssertionError((conditions,len(matching)))
    return matching[0]


class TrainingInterventionAuditTests(unittest.TestCase):
    def analyse(self,rows,**kwargs):
        return training_contrasts(rows,repeats=30,seed=111,**kwargs)

    def test_base_wrong_trained_correct_has_paired_effect_interval_and_test(self):
        rows=self.analyse(sample(),declared_arms=["sft_lora"])
        for metric in ("strict_robust_correctness","R","invariance"):
            r=find(rows,metric=metric,language="tr",comparison="equivalent")
            self.assertEqual(r["n_pairs"],6)
            self.assertEqual(r["n_clusters"],6)
            self.assertIsNotNone(r["ci_low"])
            self.assertEqual(r["difference"],0 if metric=="invariance" else 1)
        r=find(rows,metric="accuracy",language="tr",condition="canonical")
        self.assertEqual(r["difference"],1)
        self.assertEqual(r["p_value"],.03125)
        self.assertEqual(r["ci_low"],1)
        self.assertEqual(r["ci_high"],1)

    def test_invalids_stay_wrong_and_common_four_support_is_explicit(self):
        rows=sample()
        for r in rows:
            if r["arm"]=="sft_lora" and r["cluster_id"]=="fixture-0" and r["condition"]=="equivalent":r["prediction"]=None
        out=self.analyse(rows,declared_arms=["sft_lora"])
        strict=find(out,metric="strict_robust_correctness",language="tr",comparison="equivalent")
        common=find(out,metric="R",language="tr",comparison="equivalent")
        self.assertAlmostEqual(strict["difference"],5/6)
        self.assertEqual(common["difference"],1)
        self.assertEqual(common["n_planned_pairs"],6)
        self.assertEqual(common["n_common_four_valid"],5)
        self.assertEqual(common["n_base_valid_pairs"],6)
        self.assertEqual(common["n_trained_valid_pairs"],5)

    def test_dependent_latent_queries_do_not_receive_iid_mcnemar(self):
        out=self.analyse(sample(queries=2),declared_arms=["sft_lora"])
        row=find(out,metric="accuracy",language="en",condition="canonical")
        self.assertIsNone(row["p_value"])
        self.assertEqual(row["status"],"undefined_dependent_clusters")
        self.assertEqual(row["n_clusters"],6)
        self.assertEqual(row["n_pairs"],12)
        self.assertEqual(row["ci_low"],1)

    def test_holm_includes_all_declared_arms_in_the_same_family(self):
        out=self.analyse(sample(arms=("base","sft_lora","paired_sft")))
        a=find(out,metric="accuracy",language="tr",condition="canonical",training_arm="sft_lora")
        b=find(out,metric="accuracy",language="tr",condition="canonical",training_arm="paired_sft")
        self.assertEqual(a["holm_family"],b["holm_family"])
        self.assertEqual(a["holm_family_size"],5)
        self.assertEqual(a["p_adjusted"],.15625)
        self.assertEqual(a["p_value"],.03125)

    def test_duplicate_missing_coordinate_or_changed_gold_rejected(self):
        for mutate in (lambda rows:rows.append(deepcopy(rows[0])),lambda rows:rows.pop(),
                       lambda rows:rows[-1].update(gold_label=False),
                       lambda rows:rows[-1].update(query_subject="changed")):
            rows=sample();mutate(rows)
            with self.assertRaises(ValueError):self.analyse(rows,declared_arms=["sft_lora"])

    def test_missing_baseline_or_arm_is_unavailable_not_zero_performance(self):
        for arms in (("base",),("sft_lora",)):
            out=self.analyse(sample(arms=arms),declared_arms=["sft_lora"])
            self.assertTrue(all(r["difference"] is None and r["n_pairs"] is None for r in out))
            self.assertTrue(all(r["status"]=="unavailable_missing_arm" for r in out))
        self.assertEqual(self.analyse([]),[])

    def test_all_invalid_common_support_and_constant_kappa_are_undefined(self):
        rows=sample()
        for row in rows:
            if row["arm"]=="sft_lora":row["prediction"]=None
        out=self.analyse(rows,declared_arms=["sft_lora"])
        r=find(out,metric="R",language="tr",comparison="equivalent")
        self.assertIsNone(r["difference"]);self.assertIsNone(r["ci_low"])
        self.assertEqual(r["n_common_four_valid"],0)
        out=self.analyse(sample(),declared_arms=["sft_lora"])
        k=find(out,metric="kappa",language="tr",comparison="equivalent")
        self.assertIsNone(k["difference"])
        self.assertIsNone(k["p_value"])

    def test_seed_order_stability_and_input_immutability(self):
        rows=sample();before=deepcopy(rows)
        a=self.analyse(rows,declared_arms=["sft_lora"])
        b=self.analyse(list(reversed(rows)),declared_arms=["sft_lora"])
        self.assertEqual(a,b);self.assertEqual(rows,before)

    def test_dependent_r_plot_prefers_cluster_interval_over_descriptive_wilson(self):
        row={"R_ci_low":.1,"R_ci_high":.9,"R_wilson_low":.45,"R_wilson_high":.55,
             "R_wilson_interpretation":"descriptive_dependent_pairs"}
        self.assertEqual(_ci(row,"R",.5),(.1,.9))
        row["R_wilson_interpretation"]="binomial_iid"
        self.assertEqual(_ci(row,"R",.5),(.45,.55))


if __name__=="__main__":unittest.main()


class AdapterEvidenceAuditTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name).resolve()
        self.plan={'training_version':'fixture-only','plan_sha256':'p'*64,'config':{'arms':['base','sft_lora']},
            'arm_budgets':[{'arm':'base'},{'arm':'sft_lora'}],
            'data':{'evaluation':{},'overlap_audit':{'lexical':{'clean_full_context_holdout':False}}}}
        self.identity={'arm':'sft_lora','adapter_sha256':'a'*64,'training_summary_sha256':'b'*64}
        self.loaded={'plan':self.plan,'manifest':{'identity':self.identity},'manifest_sha256':'c'*64,'observations':[]}

    def test_nonbase_evaluation_cannot_be_reported_without_completed_adapter_evidence(self):
        from unittest.mock import patch
        from training.reporting import report
        output=self.root/'results/training/reports/no-evidence'
        with patch('training.reporting.validate_plan'),patch('training.reporting.load_evaluation',return_value=self.loaded):
            with self.assertRaisesRegex(ValueError,'exact verified completed training'):
                report(self.root,self.plan,output,evaluations=['synthetic-only'])
        self.assertFalse(output.exists())

    def test_wrong_summary_or_adapter_fingerprint_rejected(self):
        import json
        from unittest.mock import patch
        from training.reporting import report
        run=self.root/'synthetic-only-training';run.mkdir()
        (run/'training_summary.json').write_text(json.dumps({'arm':'sft_lora','stats':{'optimizer_steps':1}}))
        for field in ('adapter_sha256','training_summary_sha256'):
            wrong={**self.identity,field:'0'*64}
            with patch('training.reporting.validate_plan'),patch('training.reporting.load_evaluation',return_value=self.loaded),patch('training.reporting.adapter_identity',return_value=wrong):
                with self.assertRaisesRegex(ValueError,'exact verified completed training'):
                    report(self.root,self.plan,self.root/'results/training/reports/wrong',evaluations=['synthetic-only'],training_runs=[run])

    def test_exact_verified_identity_can_continue_to_reporting(self):
        import json
        from unittest.mock import patch
        from training.reporting import report
        run=self.root/'synthetic-only-training';run.mkdir()
        (run/'training_summary.json').write_text(json.dumps({'arm':'sft_lora','stats':{'optimizer_steps':1}}))
        with patch('training.reporting.validate_plan'),patch('training.reporting.load_evaluation',return_value=self.loaded),patch('training.reporting.adapter_identity',return_value=self.identity),patch('training.reporting.export_tables',return_value=[]) as exporter:
            result=report(self.root,self.plan,self.root/'results/training/reports/fixture',evaluations=['synthetic-only'],training_runs=[run])
        exporter.assert_called_once()
        self.assertTrue(result['contains_training_run'])
        self.assertEqual(result['actual_optimizer_steps'],1)


if __name__=='__main__':unittest.main()
