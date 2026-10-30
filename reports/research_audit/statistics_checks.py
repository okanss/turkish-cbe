"""Independent small-domain arithmetic oracles and source-backed paper checks."""

import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

from analysis.metrics import summarize_pairs
from analysis.statistics import bootstrap,holm_adjust,mcnemar
from publication.tables import build_paper_tables


def _check(condition,message):
    if not condition:raise ValueError(message)


def run_statistics_checks(root):
    root=Path(root).resolve();count=0
    for gold in (False,True):
        for answers in itertools.product((None,False,True),repeat=6):
            rows=[{'cluster_id':str(i),'n':4,'gold':gold,'original':answers[2*i],'equivalent':answers[2*i+1]} for i in range(3)]
            s=summarize_pairs(rows);v=[r for r in rows if None not in (r['original'],r['equivalent'])]
            _check(s['n_pairs']==3 and s['n_valid_pairs']==len(v),'metric support mismatch')
            for key,side in [('original_accuracy','original'),('equivalent_accuracy','equivalent')]:
                _check(s[key]==sum(r[side] is not None and r[side]==gold for r in rows)/3,'invalid marginal denominator')
            _check(s['strict_robust_correctness']==sum(r['original']==gold and r['equivalent']==gold for r in v)/3,'strict R denominator')
            if v:
                _check(math.isclose(s['R']+s['D']+s['U'],1),'R+D+U identity')
                _check(math.isclose(s['mean_accuracy'],s['R']+s['U']/2),'mean accuracy identity')
                _check(math.isclose(s['invariance'],s['R']+s['D']),'invariance identity')
                q=sum(int(r['original'])+int(r['equivalent']) for r in v)/(2*len(v))
                _check(s['q']==q,'pooled valid q')
                if q in (0,1):_check(s['kappa'] is None,'constant-answer kappa must be undefined')
            else:_check(all(s[k] is None for k in ('R','D','U','mean_accuracy','invariance','kappa','q')),'empty valid support')
            count+=1
    exact=0
    for b in range(31):
        for c in range(31):
            if b+c==0:continue
            expected=min(1.,2*sum(math.comb(b+c,k) for k in range(min(b,c)+1))/2**(b+c))
            _check(mcnemar([True]*b+[False]*c,[False]*b+[True]*c,method='exact')['p_value']==expected,'exact binomial oracle')
            exact+=1
    _check(mcnemar([True,True],[False,False],clusters=['same','same'])['p_value'] is None,'dependent McNemar must be undefined')
    _check(holm_adjust({'a':.01,'b':.04,'missing':None})=={'a':.03,'b':.08,'missing':None},'Holm missing declared slot')
    rows=[{'cluster_id':'a','n':4,'x':1},{'cluster_id':'a','n':4,'x':1},{'cluster_id':'b','n':4,'x':0}]
    statistic=lambda rows:{'mean':sum(r['x'] for r in rows)/len(rows)}
    first=bootstrap(rows,statistic,seed=11,repeats=2000)
    _check(first==bootstrap(list(reversed(rows)),statistic,seed=11,repeats=2000),'cluster bootstrap order determinism')
    _check(first['mean']['independent_clusters']==2,'cluster rather than row resampling')
    manifest=json.loads((root/'reports/current/report_manifest.json').read_bytes())
    evidence=json.loads((root/'reports/current/evidence.json').read_bytes())
    for item in manifest['files']:
        p=root/item['path']
        _check(p.stat().st_size==item['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],'paper output integrity')
    tables=build_paper_tables(evidence)
    for stem,rows in tables.items():
        with (root/'paper/tables'/f'{stem}.csv').open(newline='') as handle:stored=list(csv.DictReader(handle))
        _check(len(stored)==len(rows),'published table reconstruction')
        if stem.startswith(('table_2','table_3','table_4','table_6','table_7')):
            _check(all(r['status']=='not_run' and r['original_accuracy']=='NA' and r['p_value']=='NA' for r in stored),'invented performance result')
    _check(not manifest['contains_model_results'] and not evidence['contains_model_results'] and not manifest['submission_ready'],'current study has no genuine model results')
    return {'artifact_type':'independent_statistics_paper_checks','synthetic_oracles_only':True,
        'exhaustive_metric_cases':count,'exact_mcnemar_cases':exact,'bootstrap_repeats':2000,
        'dependent_mcnemar_rejected':True,'holm_missing_slots_preserved':True,
        'published_output_hashes_verified':len(manifest['files']),'paper_tables_reconstructed':len(tables),
        'real_model_results':False,'model_calls':0,'training_steps':0,
        'scope':'Arithmetic probes are synthetic contracts. Dataset construction counts are real. No model-performance statistics are generated.'}
