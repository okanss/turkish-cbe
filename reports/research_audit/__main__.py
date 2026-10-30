"""Read-only scientific audit; writes a new evidence directory, never model results."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import tempfile

from . import AUDIT_VERSION
from .data_checks import run_data_checks
from .statistics_checks import run_statistics_checks

CHECKS = (
 (1,'Dataset leakage','pass_exact_instances','Source order, alias order and visible prompt hashes disjoint; no distribution-free claim.'),
 (2,'Train/test latent separation','pass','All frozen source groups and query seeds replayed.'),
 (3,'Turkish semantic equivalence','bounded_formal_pass_human_pending','All normal-form metadata and stratified actual surfaces; shared-code parser limitation remains.'),
 (4,'Turkish morphology','bounded_tests_human_pending','Independent suffix fixtures; qualified naturalness judgments not supplied.'),
 (5,'Query direction/gold balance','pass','Test cells50 each; non-test12/13, margins balanced.'),
 (6,'Endpoint versus internal/distance','pass_with_construct_limit','Strata remain separate; degree-only endpoint shortcut demonstrates n is not inference depth.'),
 (7,'Matched models/languages','design_pass_no_model_results','Frozen coordinates match; genuine multi-model responses do not exist.'),
 (8,'Invalid answers','pass_contract','Invalids wrong in marginal accuracy and retained in planned denominators.'),
 (9,'R+D+U identities','pass_exhaustive_small_domain','1458 independent Boolean/invalid/gold cases; joint-valid support only.'),
 (10,'Accuracy versus invariance support','pass_contract','All-planned versus joint-valid estimands labelled; common-four training sensitivity added.'),
 (11,'Truncation cost completion','limited_no_real_backend','Neutral decoding recorded, context-invalid retained, unknown costs null; no genuine usage to certify.'),
 (12,'Paired statistical tests','pass_contract','Exact McNemar integer oracle, dependent McNemar refusal, latent-cluster bootstrap.'),
 (13,'Multiplicity','pass_declared_scope','Holm within model/design/metric hypotheses; not study-wide FWER.'),
 (14,'Linguistic length/tokenizer confounding','open_design_limitation','P4/P5 multiple transformations, language/scaffold/token differences, higher/lower overlap.'),
 (15,'Reproducibility','pass_cpu_scoped','All seed queries and output hashes checked; optional Torch/CUDA remains unexecuted.'),
 (16,'Call-level result lineage','no_real_runs_contract_only','Native raw/normalized/plan provenance and exact adapter identities enforced; no server attestation.'),
 (17,'Paper numerical claims','pass_construction_only','Published hash-bound construction counts; model performance remains NA.'),
 (18,'Inherited versus new methods','pass_draft_scope','CBE inheritance and Turkish/query/lexical/training extensions distinguished; no replication claim.'),
)


def _sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def audit(root, output):
    root=Path(root).resolve();output=Path(output)
    if not output.is_absolute():output=root/output
    output=output.absolute()
    if (output.exists() or '..' in output.parts or not output.is_relative_to(root/'reports/audit')
            or any(p.is_symlink() for p in (output,*output.parents))):
        raise ValueError('choose a new nonsymlink reports/audit/<name> directory')
    active=Path(__file__).resolve().parent
    local=root/'reports/research_audit'
    if ({p.name for p in active.glob('*.py')}!={p.name for p in local.glob('*.py')}
            or any(_sha(p)!=_sha(local/p.name) for p in active.glob('*.py'))):
        raise ValueError('installed audit sources differ from project')
    sources={p.relative_to(root).as_posix():_sha(p) for base in ('src','reports/publication','reports/research_audit')
             for p in sorted((root/base).rglob('*.py'))}
    input_paths=['pyproject.toml','data/complete_grid/manifest.json','data/complete_grid/grid.sqlite','data/complete_grid/coverage.json',
                 'data/validation/turkish_samples.csv','data/validation/turkish_samples_manifest.json',
                 'reports/current/report_manifest.json','reports/current/evidence.json']
    original=json.loads((root/'reports/current/report_manifest.json').read_bytes())
    input_paths+= [item['path'] for item in original['files']]
    inputs={rel:_sha(root/rel) for rel in sorted(set(input_paths))}
    data=run_data_checks(root);statistics=run_statistics_checks(root)
    if any(_sha(root/rel)!=sha for rel,sha in inputs.items()) or any(_sha(root/rel)!=sha for rel,sha in sources.items()):
        raise ValueError('audit inputs or code changed during read-only checks')
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix='.research-audit-',dir=output.parent))
    try:
        for name,value in [('data_checks.json',data),('statistics_checks.json',statistics)]:
            (temporary/name).write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')
        with (temporary/'checklist.csv').open('x',newline='',encoding='utf-8') as handle:
            writer=csv.writer(handle);writer.writerow(['item','check','status','scope']);writer.writerows(CHECKS)
        manifest={'artifact_type':'research_software_audit','audit_version':AUDIT_VERSION,
            'status':'not_ready_for_empirical_publication','contains_model_results':False,
            'human_linguistic_approval':False,'actual_model_calls':0,'actual_training_steps':0,
            'environment':{'python':platform.python_version(),'platform':platform.platform()},
            'source_files_sha256':sources,'input_files_sha256':inputs,
            'checklist_items':len(CHECKS),'dataset_manifest_sha256':data['dataset_manifest_sha256'],
            'files':[{'path':p.name,'sha256':_sha(p),'bytes':p.stat().st_size} for p in sorted(temporary.iterdir())],
            'limitations':['AI/software audit is not qualified human linguistic sign-off.',
                'Real inference and optional Torch/CUDA execution were not performed.',
                'Surface replay is stratified and bounded; metadata and query seed replay covers the full frozen bank.',
                'Historical Prompt10 paper is numerically verified; code updates require new plans and new report targets.']}
        (temporary/'audit_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
        if output.exists():raise FileExistsError('audit destination appeared concurrently')
        temporary.rename(output)
    finally:shutil.rmtree(temporary,ignore_errors=True)
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path.cwd())
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    try:result=audit(args.project_root,args.output)
    except (ValueError,OSError,KeyError) as error:parser.exit(2,'error: '+str(error)+'\n')
    print(json.dumps({k:result[k] for k in ('status','audit_version','checklist_items','contains_model_results','human_linguistic_approval')},sort_keys=True))
    return 0


if __name__=='__main__':raise SystemExit(main())
