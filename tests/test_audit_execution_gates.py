"""Temporary synthetic reviewer decisions cannot authorize live model calls."""

from copy import deepcopy
import csv
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_inference_runner as fixtures
from inference.dataset import GridDataset,default_selection
from inference.planning import make_plan
from inference.runner import run_plan,ExecutionBlocked,_review_evidence
from inference.linguistic_review import validate_review,AXES,RATINGS
from complete_grid import dataset_builder as grid
from training.config import load_config
from training.planning import make_plan as training_plan
from training.evaluation import evaluate,evaluation_sampling
from training.engine import TrainingBlocked

PROJECT=Path(__file__).resolve().parents[1]


class ExecutionReviewAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):fixtures.InferenceRunnerTests.setUpClass.__func__(cls)

    def setUp(self):
        fixtures.InferenceRunnerTests.setUp(self)
        shutil.copytree(PROJECT/"configs",self.root/"configs")
        self.selection=default_selection()
        self.selection.update(sizes=[4],groups_per_size=4)
        with GridDataset(self.root,fixtures.MANIFEST) as data:
            qids=[r[0] for r in data.connection.execute("SELECT q.id FROM queries q JOIN groups g ON g.id=q.group_id WHERE g.split='pilot' AND g.n=4 AND q.query_type='endpoint' ORDER BY q.id LIMIT 50")]
            designs=[r[0] for r in data.connection.execute("SELECT id FROM designs WHERE name_control='native_shared' AND relation_id='spatial_front' AND language='tr' AND condition IN('canonical','equivalent') ORDER BY id")]
            tokenizers=grid.ModelTokenizers(self.root/'tokenizers')
            self.rows=[]
            for did in designs:
                for qid in qids:
                    r=grid._materialized(self.root,data.connection,qid,did,tokenizers)
                    self.rows.append({"query_id":qid,"design_id":did,"prompt_sha256":r['prompt_sha256'],"prompt":r['prompt'],
                        **{k:r[k] for k in AXES},**{k:"pass" for k in RATINGS},"resolution":"approved"})
        self.csv=self.root/'fixture-review.csv';self.receipt=self.root/'fixture-review.json'
        self.review={"schema_version":1,"artifact_type":"turkish_linguistic_review","status":"approved",
            "fixture_only":True,"reviewers":["Synthetic test reviewer A","Synthetic test reviewer B"],
            "reviewer_qualification":"two_independent_turkish_proficient_reviewers",
            "dataset_manifest_sha256":hashlib.sha256((self.root/fixtures.MANIFEST).read_bytes()).hexdigest(),
            "samples":{"path":self.csv.name,"rows":100}}
        self.write_review()

    def write_review(self):
        with self.csv.open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(self.rows[0]));writer.writeheader();writer.writerows(self.rows)
        self.review['samples'].update(rows=len(self.rows),sha256=hashlib.sha256(self.csv.read_bytes()).hexdigest())
        self.receipt.write_text(json.dumps(self.review))

    def check_fixture(self,selection=None):
        return validate_review(self.root,fixtures.MANIFEST,self.receipt,[selection or self.selection],allow_fixture=True)

    def test_exact_sample_and_two_rater_contract_validates_only_with_test_switch(self):
        evidence=self.check_fixture()
        self.assertEqual(evidence['sample_count'],100)
        self.assertEqual(evidence['required_axes']['n'],[4])
        with self.assertRaisesRegex(ValueError,'synthetic'):
            validate_review(self.root,fixtures.MANIFEST,self.receipt,[self.selection])

    def test_pending_or_weak_self_declared_approval_rejected(self):
        self.review['status']='pending';self.write_review()
        with self.assertRaisesRegex(ValueError,'approved'):self.check_fixture()
        self.review={'status':'approved','reviewers':['one']};self.receipt.write_text(json.dumps(self.review))
        with self.assertRaisesRegex(ValueError,'versioned'):self.check_fixture()

    def test_two_independent_identified_reviewers_required(self):
        self.review['reviewers']=['same','same'];self.write_review()
        with self.assertRaisesRegex(ValueError,'two identified'):self.check_fixture()

    def test_unresolved_or_failed_human_decisions_cannot_authorize(self):
        for key in ('reviewer1_naturalness','reviewer2_equivalence','resolution'):
            old=self.rows[0][key];self.rows[0][key]='pending';self.write_review()
            with self.assertRaisesRegex(ValueError,'human decisions'):self.check_fixture()
            self.rows[0][key]=old

    def test_duplicate_prompts_and_fewer_than_100_rejected(self):
        self.rows[-1]=deepcopy(self.rows[0]);self.write_review()
        with self.assertRaisesRegex(ValueError,'duplicate'):self.check_fixture()
        self.rows=self.rows[:99];self.write_review()
        with self.assertRaisesRegex(ValueError,'100'):self.check_fixture()

    def test_sample_hash_prompt_text_and_metadata_tampering_rejected(self):
        self.csv.write_text(self.csv.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'hash'):self.check_fixture()
        self.rows[0]['prompt']+=' changed';self.write_review()
        with self.assertRaisesRegex(ValueError,'identified frozen'):self.check_fixture()
        self.rows[0]['prompt']=self.rows[1]['prompt'];self.rows[0]['n']=8;self.write_review()
        with self.assertRaises(ValueError):self.check_fixture()

    def test_review_scope_does_not_extend_to_unseen_sizes_or_query_types(self):
        for axis,value in [('sizes',[4,8]),('query_types',['endpoint','internal']),('name_controls',['symbolic_shared'])]:
            selection=deepcopy(self.selection);selection[axis]=value
            with self.assertRaisesRegex(ValueError,'cover'):self.check_fixture(selection)

    def test_template_cell_minimum_cannot_be_replaced_by_extra_other_cells(self):
        first=self.rows[:19];equivalent=self.rows[50:]
        with GridDataset(self.root,fixtures.MANIFEST) as data:
            did=data.connection.execute("SELECT id FROM designs WHERE name_control='native_shared' AND relation_id='spatial_front' AND language='tr' AND condition='P1'").fetchone()[0]
            tokenizers=grid.ModelTokenizers(self.root/'tokenizers')
            more=[]
            for item in self.rows[:31]:
                r=grid._materialized(self.root,data.connection,int(item['query_id']),did,tokenizers)
                more.append({**item,'design_id':did,'prompt':r['prompt'],'prompt_sha256':r['prompt_sha256'],'condition':'P1'})
        self.rows=first+equivalent+more;self.write_review()
        self.assertEqual(len(self.rows),100)
        with self.assertRaisesRegex(ValueError,'20 reviewed'):self.check_fixture()

    def test_wrong_dataset_and_missing_receipt_rejected(self):
        self.review['dataset_manifest_sha256']='0'*64;self.write_review()
        with self.assertRaisesRegex(ValueError,'another'):self.check_fixture()
        with self.assertRaisesRegex(ValueError,'required'):
            validate_review(self.root,fixtures.MANIFEST,None,[self.selection])

    def test_full_native_runner_uses_the_same_review_contract(self):
        plan=make_plan(self.root,fixtures.MANIFEST,[self.model],selection=self.selection,policy=self.policy)
        with self.assertRaisesRegex(ExecutionBlocked,'synthetic'):_review_evidence(self.root,plan,self.receipt)
        from reasoning.runner import _review_evidence as reasoning_review
        with self.assertRaisesRegex(ExecutionBlocked,'synthetic'):reasoning_review(self.root,plan,self.receipt)

    def test_large_pilot_cannot_bypass_review_by_renaming_stage(self):
        selection=default_selection()
        selection.update(groups_per_size=8,query_types=['endpoint','internal','distance_controlled'],name_controls=['native_shared','symbolic_shared'],
            relations=['spatial_front','spatial_right','spatial_above'],conditions=['canonical','equivalent','P1','P2','P3','P4','P5'])
        policy={**self.policy,'max_calls':1000000}
        plan=make_plan(self.root,fixtures.MANIFEST,[self.model],selection=selection,policy=policy)
        self.assertGreater(plan['generation_calls'],2000)
        with patch('inference.runner._http_factory',side_effect=AssertionError('must not load provider')) as provider:
            with self.assertRaisesRegex(ExecutionBlocked,'review'):run_plan(self.root,plan,execute=True,run_id='unreviewed-large')
            provider.assert_not_called()
        self.assertFalse((self.root/'results/raw').exists())

    def test_large_training_evaluation_stops_before_cuda_or_model_loading(self):
        plan=training_plan(self.root,load_config(PROJECT/'configs/training/pilot.toml'))
        with patch('training.evaluation.execution_gate') as gpu,patch('training.evaluation.load_runtime') as loader:
            with self.assertRaisesRegex(TrainingBlocked,'review'):
                evaluate(self.root,plan,'base',self.root/'results/training/evaluations/unreviewed',execute=True,max_calls=1000000)
            gpu.assert_not_called();loader.assert_not_called()
        self.assertFalse((self.root/'results/training/evaluations').exists())

    def test_eval_generation_config_cannot_inherit_snapshot_decoding_penalties(self):
        from transformers import GenerationConfig
        model=SimpleNamespace(generation_config=GenerationConfig(repetition_penalty=1.7,num_beams=4,forced_bos_token_id=42,eos_token_id=3))
        tokenizer=SimpleNamespace(eos_token_id=3)
        generation,record=evaluation_sampling(model,tokenizer,32,42)
        self.assertFalse(generation.do_sample);self.assertEqual(generation.num_beams,1)
        # transformers>=5 reports an unset neutral penalty as None instead of 1.
        self.assertIn(generation.repetition_penalty,(None,1))
        self.assertIsNone(generation.forced_bos_token_id)
        self.assertFalse(record['use_model_defaults']);self.assertEqual(record['seed'],42)
        self.assertEqual(model.generation_config.repetition_penalty,1.7)


if __name__=='__main__':unittest.main()
