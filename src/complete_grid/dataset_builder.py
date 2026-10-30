"""Materialize a complete factor grid as normalized, measured prompt recipes.

The SQLite bank contains a measurement row for every actual prompt, not a
sample or an extrapolated count. JSONL export reconstructs the exact prompt.
Old frozen sources and data remain byte-identical.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sqlite3
import sys
import unicodedata

from data_generation.generator import (DEFAULT_DISTANCES, GenerationPlan,
    canonical_bytes, generate_record, valid_distances)
from data_generation.validation import validate_record
from experiments.dataset_builder import (DEFAULT_PLAN, DatasetPlan, SplitSpec,
    balanced_indices, digest, file_hash, presentation_plans, _tr_clause, _legacy_groups, load_plan,
    _source_hashes as base_source_hashes)
from experiments.names import name_maps, validate_name_maps
from experiments.token_lengths import TOKENIZER_SPEC
from .text_lengths import fastmeasure_text as measure_text
from renderers import english, turkish
from .grid_design import (GRID_VERSION, NAME_CONTROLS, RELATION_IDS, PROMPT_CONDITIONS,
    iter_designs, comparison_designs, expected_counts, validate_grid_pair)
from .model_tokens import ModelTokenizers, MODEL_TO_ENCODING
from .surface_reader import SurfaceReader

SCHEMA_VERSION = 1
APPLICATION_ID = 0x54434245

SCHEMA = """
CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
CREATE TABLE designs(id INTEGER PRIMARY KEY, name_control TEXT NOT NULL,
 relation_id TEXT NOT NULL, language TEXT NOT NULL, condition TEXT NOT NULL,
 factor_axes TEXT NOT NULL, UNIQUE(name_control,relation_id,language,condition));
CREATE TABLE groups(id INTEGER PRIMARY KEY, latent_instance_id TEXT NOT NULL UNIQUE,
 split TEXT NOT NULL, n INTEGER NOT NULL, sample_index INTEGER NOT NULL,
 source_json TEXT NOT NULL, UNIQUE(split,n,sample_index));
CREATE TABLE queries(id INTEGER PRIMARY KEY, group_id INTEGER NOT NULL,
 query_instance_id TEXT NOT NULL UNIQUE, query_type TEXT NOT NULL,
 query_distance INTEGER NOT NULL, query_subject TEXT NOT NULL, query_object TEXT NOT NULL,
 query_relation TEXT NOT NULL, query_direction TEXT NOT NULL, gold_label INTEGER NOT NULL CHECK(gold_label IN(0,1)),
 UNIQUE(id,group_id), FOREIGN KEY(group_id) REFERENCES groups(id));
CREATE UNIQUE INDEX endpoint_per_group ON queries(group_id) WHERE query_type='endpoint';
CREATE TABLE components(group_id INTEGER NOT NULL, design_id INTEGER NOT NULL,
 prefix_sha256 BLOB NOT NULL CHECK(length(prefix_sha256)=32), prefix_bytes INTEGER NOT NULL,
 prefix_chars INTEGER NOT NULL, prefix_whitespace INTEGER NOT NULL, prefix_reference INTEGER NOT NULL,
 cl100k_prefix INTEGER NOT NULL, o200k_prefix INTEGER NOT NULL,
 normal_form_sha256 BLOB NOT NULL CHECK(length(normal_form_sha256)=32),
 plan_sha256 BLOB NOT NULL CHECK(length(plan_sha256)=32),
 PRIMARY KEY(group_id,design_id), FOREIGN KEY(group_id) REFERENCES groups(id),
 FOREIGN KEY(design_id) REFERENCES designs(id)) WITHOUT ROWID;
CREATE TABLE measurements(query_id INTEGER NOT NULL, design_id INTEGER NOT NULL, group_id INTEGER NOT NULL,
 prompt_sha256 BLOB NOT NULL CHECK(length(prompt_sha256)=32), cl100k_base INTEGER NOT NULL CHECK(cl100k_base>0),
 o200k_base INTEGER NOT NULL CHECK(o200k_base>0), utf8_bytes INTEGER NOT NULL,
 unicode_codepoints INTEGER NOT NULL, whitespace_tokens INTEGER NOT NULL, reference_tokens INTEGER NOT NULL,
 PRIMARY KEY(query_id,design_id),
 FOREIGN KEY(query_id,group_id) REFERENCES queries(id,group_id),
 FOREIGN KEY(group_id,design_id) REFERENCES components(group_id,design_id)) WITHOUT ROWID;
CREATE TABLE comparison_rules(id INTEGER PRIMARY KEY, arm TEXT NOT NULL,
 left_design_id INTEGER NOT NULL, right_design_id INTEGER NOT NULL,
 UNIQUE(arm,left_design_id,right_design_id),
 FOREIGN KEY(left_design_id) REFERENCES designs(id), FOREIGN KEY(right_design_id) REFERENCES designs(id));
CREATE VIEW comparison_cells AS
 SELECT r.id AS rule_id,r.arm,q.group_id,q.id AS left_query_id,r.left_design_id,
 q.id AS right_query_id,r.right_design_id FROM comparison_rules r JOIN queries q ON r.arm!='F'
 UNION ALL
 SELECT r.id,r.arm,q.group_id,e.id,r.left_design_id,q.id,r.right_design_id
 FROM comparison_rules r JOIN queries q ON r.arm='F' AND q.query_type!='endpoint'
 JOIN queries e ON e.group_id=q.group_id AND e.query_type='endpoint';
"""


def _json(value):
    return canonical_bytes(value).decode('utf-8')


def _source_hashes():
    root = Path(__file__).resolve().parent
    return {p.name: file_hash(p) for p in sorted(root.glob('*.py')) if not p.is_symlink()}


def _check_schema(connection):
    expected=sqlite3.connect(':memory:')
    try:
        expected.executescript(SCHEMA)
        expected.execute('CREATE INDEX component_measurements ON measurements(group_id,design_id,query_id)')
        sql="SELECT type,name,tbl_name,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name"
        if [tuple(r) for r in connection.execute(sql)]!=expected.execute(sql).fetchall():
            raise ValueError('database schema/comparison view changed')
    finally:expected.close()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _query_spec(plan, spec):
    return GenerationPlan(seed=spec.seed, sizes=spec.sizes,
        examples_per_size=4*((spec.count+3)//4), entity_types=('turkish_names',))


def _source_group(endpoint, presentation_seed):
    maps = name_maps(endpoint)
    validate_name_maps(endpoint, maps)
    return {'endpoint': endpoint, 'name_maps': maps,
            'presentation_plans': presentation_plans(endpoint, presentation_seed)}


def _queries(endpoint, generation_plan, n, index):
    yield endpoint
    yield generate_record(generation_plan,n,index,'internal')
    for distance in valid_distances(n,DEFAULT_DISTANCES):
        yield generate_record(generation_plan,n,index,'distance_controlled',distance)


def _compact_query(record):
    return {k: record[k] for k in ('instance_id','query_type','query_distance',
        'query_subject','query_object','query_relation','query_direction','gold_label')}


def _record(source, row):
    result = dict(source['endpoint'])
    for key in ('query_type','query_distance','query_subject','query_object','query_relation','query_direction'):
        result[key] = row[key]
    result['instance_id'] = row['query_instance_id']
    result['gold_label'] = bool(row['gold_label'])
    return result


def _axes(language, condition):
    if condition=='canonical': return []
    if condition=='equivalent': return ['list_order','expression_direction']
    return list(turkish._AXES[condition])


def _design_rows():
    return [(i,*design,_json(_axes(design[2],design[3])))
            for i,design in enumerate(iter_designs(),1)]


def _rules():
    lookup={tuple(row[1:5]):row[0] for row in _design_rows()}
    result=[]
    for control in NAME_CONTROLS:
        for relation in RELATION_IDS:
            for arm,left,right in comparison_designs('internal'):
                result.append((len(result)+1,arm,lookup[(control,relation,*left)],lookup[(control,relation,*right)]))
    return result


def _component(source, design, cache=None):
    _,control,relation,language,condition,*_=design
    cache={} if cache is None else cache
    effective_control='native_shared' if control=='en_transliterated' and language=='tr' else control
    component_key=(effective_control,relation,language,condition)
    if component_key in cache.setdefault('components',{}):return cache['components'][component_key]
    endpoint=source['endpoint']; aliases=source['name_maps'][control][language]
    plan=source['presentation_plans'][condition]
    alphabetical=sorted(endpoint['latent_order'])
    names=[aliases[alphabetical[i]] for i in plan['entity_order']]
    reader_key=(effective_control,relation,language)
    readers=cache.setdefault('readers',{})
    if reader_key not in readers:readers[reader_key]=SurfaceReader(names,relation,language)
    reader=readers[reader_key]
    clauses=cache.setdefault('clauses',{})
    ranks={name:i for i,name in enumerate(endpoint['latent_order'])}
    reverse={value:key for key,value in aliases.items()}
    mask=set(plan['converse_chain_positions']); premises=[]; parsed=[]
    for index in plan['premise_order']:
        original_a,original_b=endpoint['premise_edges'][index]
        a,b=aliases[original_a],aliases[original_b]
        if ranks[original_a] in mask: a,b,direction=b,a,'after'
        else: direction='before'
        clause_key=(effective_control,relation,language,plan['style'],index,direction)
        if clause_key not in clauses:
            text=(english.statement(a,b,relation,direction) if language=='en'
                  else _tr_clause(a,b,relation,direction,plan['style']))
            a_read,b_read=reader.statement(text)
            edge=(reverse[a_read],reverse[b_read])
            if edge!=(original_a,original_b):
                raise ValueError('actual surface premise changed the source edge')
            clauses[clause_key]=(text,edge)
        text,edge=clauses[clause_key]
        parsed.append(edge);premises.append(text)
    if len(set(parsed))!=endpoint['n']-1 or set(parsed)!={tuple(e) for e in endpoint['premise_edges']}:
        raise ValueError('actual surface normal form differs from source chain')
    if language=='en':
        context=english.context(relation,symbolic=control=='symbolic_shared')
        prefix=context+'\nEntities: '+', '.join(names)+'.\nPremises:\n'+'\n'.join('- '+s for s in premises)+'\nQuestion:'
    else:
        context=turkish.expected_context(relation,names)
        prefix=context+'\nVarlıklar: '+', '.join(names)+'.\nÖncüller:\n'+'\n'.join('- '+s for s in premises)+'\nSoru:'
    component={'prefix':prefix,'reader':reader,'aliases':aliases,'relation':relation,'language':language,
        'plan_sha256':digest(plan),'normal_form_sha256':digest(sorted(parsed))}
    cache['components'][component_key]=component
    return component


def _suffix(component, record):
    aliases=component['aliases'];subject=aliases[record['query_subject']];object_=aliases[record['query_object']]
    language,relation=component['language'],component['relation']
    query=(english.question(subject,object_,relation,record['query_relation']) if language=='en'
        else turkish._question(subject,object_,turkish.RELATIONS[relation],record['query_relation']))
    if component['reader'].query(query)!=(subject,object_,record['query_relation']):
        raise ValueError('actual surface query changed target/direction')
    instruction=('\nProvide your answer only as YES or NO.' if language=='en'
                 else '\nYanıtınızı yalnız YES veya NO olarak verin.')
    return ' '+query+instruction


def _component_measurement(group_id,design_id,component,tokenizers):
    cached=component.get('_prefix_measurement')
    if cached is None:
        prefix=component['prefix'];prepared=tokenizers.prepare_prefix(prefix);lengths=measure_text(prefix)
        bpe=dict(prepared.counts);prefix_sha=hashlib.sha256(prefix.encode())
        values=(prefix_sha.digest(),lengths['utf8_bytes'],lengths['unicode_codepoints'],
            lengths['whitespace_token_count'],lengths['reference_token_count'],
            bpe['cl100k_base'],bpe['o200k_base'],bytes.fromhex(component['normal_form_sha256']),
            bytes.fromhex(component['plan_sha256']))
        cached=(values,prepared,lengths,prefix_sha);component['_prefix_measurement']=cached
    values,prepared,lengths,prefix_sha=cached
    return (group_id,design_id,*values),prepared,lengths,prefix_sha


def _measurement(query_id,design_id,group_id,component,record,tokenizers,prepared,prefix_lengths,prefix_sha):
    suffix=_suffix(component,record); bpe=tokenizers.count_parts(prepared,suffix)
    # Boundary is ':' followed by space: lexical/whitespace spans cannot cross.
    suffix_lengths=measure_text(suffix);sha=prefix_sha.copy();sha.update(suffix.encode())
    return (query_id,design_id,group_id,sha.digest(),bpe['cl100k_base'],bpe['o200k_base'],
        prefix_lengths['utf8_bytes']+suffix_lengths['utf8_bytes'],
        prefix_lengths['unicode_codepoints']+suffix_lengths['unicode_codepoints'],
        prefix_lengths['whitespace_token_count']+suffix_lengths['whitespace_token_count'],
        prefix_lengths['reference_token_count']+suffix_lengths['reference_token_count'])


def _rows_digest(rows):
    sha=hashlib.sha256()
    for row in rows:
        sha.update(canonical_bytes([x.hex() if isinstance(x,bytes) else x for x in row])+b'\n')
    return sha.hexdigest()


def _check_prompt_isolation(connection):
    row=connection.execute("SELECT 1 FROM measurements m JOIN groups g ON g.id=m.group_id GROUP BY m.prompt_sha256 HAVING min(g.split)!=max(g.split) LIMIT 1").fetchone()
    if row is not None:raise ValueError('visible prompt hash leaked across splits')


def _coverage(connection):
    counts={}
    for split in ('training','development','pilot','test'):
        groups=connection.execute('SELECT count(*) FROM groups WHERE split=?',(split,)).fetchone()[0]
        queries=connection.execute('SELECT count(*) FROM queries q JOIN groups g ON g.id=q.group_id WHERE g.split=?',(split,)).fetchone()[0]
        prompts=connection.execute('SELECT count(*) FROM measurements m JOIN groups g ON g.id=m.group_id WHERE g.split=?',(split,)).fetchone()[0]
        arms={arm:count for arm,count in connection.execute('SELECT r.arm,count(*) FROM comparison_cells r JOIN groups g ON g.id=r.group_id WHERE g.split=? GROUP BY r.arm',(split,))}
        counts[split]={'orders':groups,'queries':queries,'prompts':prompts,'comparison_cells':sum(arms.values()),'cells_by_arm':arms}
    cells=[]
    query_cells=connection.execute('SELECT g.split,g.n,q.query_type,q.query_distance,q.query_direction,q.gold_label,count(*) FROM queries q JOIN groups g ON g.id=q.group_id GROUP BY 1,2,3,4,5,6 ORDER BY 1,2,3,4,5,6').fetchall()
    # Fixed-distance and internal-observed-distance are distinct strata.
    for row in query_cells:
        cells.append({'split':row[0],'n':row[1],'query_type':row[2],'query_distance':row[3],
                      'query_direction':row[4],'gold_label':bool(row[5]),'source_queries':row[6],
                      'prompts_per_design':row[6],'designs_per_query':243})
    return {'splits':counts,'source_query_cells':cells,'designs':[dict(zip(('id','name_control','relation_id','language','condition','factor_axes'),row)) for row in _design_rows()],
        'comparison_rules':[dict(zip(('id','arm','left_design_id','right_design_id'),row)) for row in _rules()]}


def _base_binding(root):
    path=root/'experiment_dataset_manifest.json'
    manifest=_read_json(path)
    if manifest['source_files_sha256']!=base_source_hashes():
        raise ValueError('original v0.4.0 source code changed')
    return {'manifest_path':'experiment_dataset_manifest.json','sha256':file_hash(path),
        'source_files_sha256':manifest['source_files_sha256']}


def _plan_from(value):
    plan=DatasetPlan(tuple(SplitSpec(s['name'],s['seed'],tuple(s['sizes']),s['count']) for s in value['splits']),value['presentation_seed'],value['vocabulary_pilot'])
    plan.validate();return plan


def _connect(path,readonly=False):
    c=sqlite3.connect(f'file:{Path(path).resolve()}?mode=ro',uri=True) if readonly else sqlite3.connect(path)
    c.row_factory=sqlite3.Row;c.execute('PRAGMA foreign_keys=ON')
    return c


def build_grid(project_root:Path,output_dir:Path,plan:DatasetPlan=DEFAULT_PLAN,resource_dir:Path|None=None,workers:int=1):
    from .parallel import iter_groups,MAX_WORKERS
    if type(workers) is not int or not 1<=workers<=MAX_WORKERS:raise ValueError("invalid worker count")
    plan.validate();root=Path(project_root).resolve();output_dir=Path(output_dir).resolve()
    if not output_dir.is_relative_to(root):raise ValueError('output must be inside project root')
    if output_dir.exists() or output_dir.is_symlink():raise FileExistsError(output_dir)
    binding=_base_binding(root);tokenizers=ModelTokenizers(resource_dir or root/'tokenizers')
    expected=expected_counts(plan);output_dir.parent.mkdir(parents=True,exist_ok=True)
    output_dir.mkdir();connection=None
    try:
        db=output_dir/'grid.sqlite';connection=_connect(db)
        connection.execute('PRAGMA page_size=4096');connection.execute('PRAGMA journal_mode=DELETE')
        connection.execute(f'PRAGMA application_id={APPLICATION_ID}');connection.execute('PRAGMA user_version=401')
        connection.executescript(SCHEMA)
        designs=_design_rows();connection.executemany('INSERT INTO designs VALUES(?,?,?,?,?,?)',designs)
        connection.executemany('INSERT INTO comparison_rules VALUES(?,?,?,?)',_rules())
        meta={'version':GRID_VERSION,'plan':asdict(plan),'tokenizers':tokenizers.descriptor(),
            'reference_tokenizer':TOKENIZER_SPEC,'base_binding':binding}
        connection.executemany('INSERT INTO metadata VALUES(?,?)',[(k,_json(v)) for k,v in meta.items()]);connection.commit()
        group_id=query_id=0;seen_groups=set();view_splits={};logical=hashlib.sha256()
        legacy,legacy_meta=_legacy_groups(root/'dataset_manifest.json') if (root/'dataset_manifest.json').exists() else ({},None)
        spec_counts={spec.name:spec.count for spec in plan.splits}
        ordinals=Counter()
        for result in iter_groups(plan,resource_dir or root/'tokenizers',workers):
            group_id=result['group_id'];split=result['split'];n=result['n']
            if result['ordering_id'] in seen_groups or 'id:'+result['ordering_id'] in legacy:
                raise ValueError('latent group leakage')
            seen_groups.add(result['ordering_id'])
            for view in result['views']:
                if (view in view_splits and view_splits[view]!=split) or 'view:'+view in legacy:
                    raise ValueError('aliased order leakage')
                view_splits[view]=split
            connection.execute('INSERT INTO groups VALUES(?,?,?,?,?,?)',result['group_row'])
            connection.executemany('INSERT INTO queries VALUES(?,?,?,?,?,?,?,?,?,?)',result['query_rows'])
            connection.executemany('INSERT INTO components VALUES(?,?,?,?,?,?,?,?,?,?,?)',result['component_rows'])
            connection.executemany('INSERT INTO measurements VALUES(?,?,?,?,?,?,?,?,?,?)',result['measurement_rows'])
            for value in result['logical_digests']:logical.update(value)
            query_id=result['query_rows'][-1][0]
            connection.commit()
            ordinals[(split,n)]+=1
            if ordinals[(split,n)]%10==0 or ordinals[(split,n)]==spec_counts[split]:
                print(f'progress {split} n={n}: {ordinals[(split,n)]}/{spec_counts[split]} orders; all 243 designs/query',flush=True)
        if connection.execute('PRAGMA foreign_key_check').fetchone() or connection.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('SQLite integrity failure')
        connection.execute('CREATE INDEX component_measurements ON measurements(group_id,design_id,query_id)')
        connection.commit()
        _check_prompt_isolation(connection)
        coverage=_coverage(connection)
        for name,counts in coverage['splits'].items():
            for key in ('orders','queries','prompts','comparison_cells','cells_by_arm'):
                if counts[key]!=expected['splits'][name][key]:raise ValueError('coverage differs from Cartesian plan: '+name+':'+key)
        (output_dir/'coverage.json').write_bytes(canonical_bytes(coverage)+b'\n')
        connection.close();connection=None
        manifest={'schema_version':SCHEMA_VERSION,'grid_version':GRID_VERSION,
            'artifact_type':'complete_measured_factor_grid','status':'structurally_frozen',
            'contains_model_results':False,'human_review_status':'pending','eligible_for_confirmatory_evaluation':False,
            'plan':asdict(plan),'expected_counts':expected,'base_binding':binding,
            'tokenizers':tokenizers.descriptor(),'reference_tokenizer':TOKENIZER_SPEC,
            'source_files_sha256':_source_hashes(),'legacy_development_guard':legacy_meta,
            'environment':{'python':platform.python_version(),'sqlite':sqlite3.sqlite_version,'unicode':unicodedata.unidata_version},
            'construction_rows_sha256':logical.hexdigest(),
            'files':[{'path':p.name,'sha256':file_hash(p),'bytes':p.stat().st_size} for p in (db,output_dir/'coverage.json')],
            'validation':{'actual_bpe_counts_populated':True,'all_243_designs_per_query':True,
                'actual_surface_normal_forms':True,'independent_oracles_agree':True,'groups_disjoint':True,
                'all_name_relation_query_crossings':True,'model_calls':0}}
        (output_dir/'manifest.json').write_bytes(canonical_bytes(manifest)+b'\n')
        return manifest
    except BaseException:
        if connection is not None:connection.close()
        shutil.rmtree(output_dir)
        raise


def _manifest_object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('duplicate manifest/metadata JSON key: '+key)
        result[key]=value
    return result


def _manifest_json(value):
    try:return json.loads(value,object_pairs_hook=_manifest_object)
    except (TypeError,UnicodeError,json.JSONDecodeError) as error:
        raise ValueError('invalid complete grid manifest/metadata JSON') from error


def _manifest_hash(value):
    return (isinstance(value,str) and len(value)==64
            and all(character in '0123456789abcdef' for character in value))


def _manifest_hashes(value,basenames=False):
    if not isinstance(value,dict) or not value:return False
    for key,sha in value.items():
        if (not isinstance(key,str) or not key or not _manifest_hash(sha)
                or (basenames and (Path(key).name!=key or not key.endswith('.py')))):
            return False
    return True


def _manifest_file(root,path,input_root=None):
    """Resolve an owned regular file while rejecting every symlink component."""
    path=Path(path).absolute()
    # macOS exposes /var through a system symlink to /private/var. Normalize
    # aliases above the caller's project root while retaining, and checking,
    # every path component inside that root.
    if input_root is not None:
        input_root=Path(input_root).absolute()
        if path.is_relative_to(input_root):path=root/path.relative_to(input_root)
    try:
        resolved=path.resolve(strict=True)
        if not resolved.is_relative_to(root):raise ValueError('grid file escapes project root')
        for candidate in (path,*path.parents):
            if candidate.is_symlink():raise ValueError('grid paths must not contain symlinks')
            if candidate==root:break
        if not resolved.is_file():raise ValueError('grid path must be a regular file')
    except OSError as error:
        raise ValueError('missing or inaccessible complete grid file') from error
    return resolved


def _load_manifest(root,manifest_path,resource_dir=None):
    """Validate portable artifact provenance without loading optional tokenizers.

    File hashes and database metadata bind the complete artifact to its frozen
    plan. Current source code, Unicode runtime and tokenizer resources are
    deliberately checked only by deep replay, so hash-only verification stays
    usable in a later environment without optional tokenizer dependencies.
    """
    input_root=Path(root).absolute();root=input_root.resolve()
    manifest_path=_manifest_file(root,manifest_path,input_root)
    m=_manifest_json(manifest_path.read_bytes())
    fields={'schema_version','grid_version','artifact_type','status','contains_model_results',
        'human_review_status','eligible_for_confirmatory_evaluation','plan','expected_counts',
        'base_binding','tokenizers','reference_tokenizer','source_files_sha256',
        'legacy_development_guard','environment','construction_rows_sha256','files','validation'}
    if (not isinstance(m,dict) or set(m)!=fields
            or type(m['schema_version']) is not int or m['schema_version']!=SCHEMA_VERSION
            or m['grid_version']!=GRID_VERSION
            or m['artifact_type']!='complete_measured_factor_grid'
            or m['status']!='structurally_frozen'
            or m['contains_model_results'] is not False
            or m['human_review_status']!='pending'
            or m['eligible_for_confirmatory_evaluation'] is not False):
        raise ValueError('unsupported complete grid manifest or publication flags')
    validation={'actual_bpe_counts_populated':True,'all_243_designs_per_query':True,
        'actual_surface_normal_forms':True,'independent_oracles_agree':True,'groups_disjoint':True,
        'all_name_relation_query_crossings':True,'model_calls':0}
    if (not isinstance(m['validation'],dict) or set(m['validation'])!=set(validation)
            or any(m['validation'][key] is not True for key in validation if key!='model_calls')
            or type(m['validation']['model_calls']) is not int or m['validation']['model_calls']!=0):
        raise ValueError('complete grid validation flags changed')
    value=m['plan']
    if (not isinstance(value,dict) or set(value)!={'splits','presentation_seed','vocabulary_pilot'}
            or not isinstance(value['splits'],list)
            or any(not isinstance(spec,dict) or set(spec)!={'name','seed','sizes','count'}
                   or not isinstance(spec['sizes'],list) for spec in value['splits'])):
        raise ValueError('invalid complete grid plan fields')
    try:
        plan=_plan_from(value)
        if _json(asdict(plan))!=_json(value) or _json(expected_counts(plan))!=_json(m['expected_counts']):
            raise ValueError('complete grid plan/counts changed')
    except (KeyError,TypeError,AttributeError) as error:
        raise ValueError('invalid complete grid plan') from error
    if (not _manifest_hash(m['construction_rows_sha256'])
            or not _manifest_hashes(m['source_files_sha256'],basenames=True)
            or not isinstance(m['environment'],dict)
            or set(m['environment'])!={'python','sqlite','unicode'}
            or any(not isinstance(v,str) or not v for v in m['environment'].values())
            or not isinstance(m['tokenizers'],dict) or not m['tokenizers']
            or not isinstance(m['reference_tokenizer'],dict) or not m['reference_tokenizer']):
        raise ValueError('invalid complete grid provenance')
    legacy=m['legacy_development_guard']
    if legacy is not None and (not isinstance(legacy,dict)
            or set(legacy)!={'manifest_sha256','orders','role'}
            or not _manifest_hash(legacy['manifest_sha256'])
            or type(legacy['orders']) is not int or legacy['orders']<0
            or legacy['role']!='previously_exposed_development_only'):
        raise ValueError('invalid legacy development guard')

    # The old manifest's stored source hashes are provenance here; comparing
    # them with today's source files belongs to deep replay, not this loader.
    binding=m['base_binding']
    if (not isinstance(binding,dict)
            or set(binding)!={'manifest_path','sha256','source_files_sha256'}
            or binding['manifest_path']!='experiment_dataset_manifest.json'
            or not _manifest_hash(binding['sha256'])
            or not _manifest_hashes(binding['source_files_sha256'])):
        raise ValueError('invalid base frozen binding')
    base_path=_manifest_file(root,root/binding['manifest_path'])
    base_manifest=_manifest_json(base_path.read_bytes())
    if (not isinstance(base_manifest,dict) or file_hash(base_path)!=binding['sha256']
            or base_manifest.get('source_files_sha256')!=binding['source_files_sha256']):
        raise ValueError('base frozen binding changed')
    files=m['files']
    if not isinstance(files,list) or len(files)!=2:
        raise ValueError('complete grid requires exactly two frozen files')
    file_names=set()
    for entry in files:
        if (not isinstance(entry,dict) or set(entry)!={'path','sha256','bytes'}
                or not isinstance(entry['path'],str)
                or entry['path'] not in {'grid.sqlite','coverage.json'}
                or entry['path'] in file_names or not _manifest_hash(entry['sha256'])
                or type(entry['bytes']) is not int or entry['bytes']<=0):
            raise ValueError('invalid, duplicate or extra frozen grid file')
        file_names.add(entry['path'])
        path=_manifest_file(root,manifest_path.parent/entry['path'])
        if path.parent!=manifest_path.parent or path.stat().st_size!=entry['bytes'] or file_hash(path)!=entry['sha256']:
            raise ValueError('frozen grid file hash/size mismatch')
    if file_names!={'grid.sqlite','coverage.json'}:
        raise ValueError('missing frozen grid file')
    connection=None
    try:
        connection=_connect(manifest_path.parent/'grid.sqlite',readonly=True)
        metadata_rows=connection.execute('SELECT key,value FROM metadata').fetchall()
        metadata={row[0]:_manifest_json(row[1]) for row in metadata_rows}
        wanted={'version':m['grid_version'],'plan':m['plan'],'tokenizers':m['tokenizers'],
            'reference_tokenizer':m['reference_tokenizer'],'base_binding':m['base_binding']}
        if len(metadata)!=len(metadata_rows) or metadata!=wanted:
            raise ValueError('database metadata differs from frozen manifest')
        return root,m,connection
    except (ValueError,sqlite3.Error,OSError,TypeError) as error:
        if connection is not None:connection.close()
        if isinstance(error,ValueError):raise
        raise ValueError('invalid complete grid database metadata') from error


def _materialized(root,connection,query_id,design_id,tokenizers):
    row=connection.execute('SELECT * FROM queries WHERE id=?',(query_id,)).fetchone()
    design=connection.execute('SELECT * FROM designs WHERE id=?',(design_id,)).fetchone()
    if row is None or design is None:raise ValueError('unknown query/design ID')
    group=connection.execute('SELECT * FROM groups WHERE id=?',(row['group_id'],)).fetchone()
    source=json.loads(group['source_json']);record=_record(source,row)
    validate_record(record)
    design_values=tuple(design[k] for k in ('id','name_control','relation_id','language','condition','factor_axes'))
    component=_component(source,design_values);suffix=_suffix(component,record);prompt=component['prefix']+suffix
    measured=connection.execute('SELECT * FROM measurements WHERE query_id=? AND design_id=?',(query_id,design_id)).fetchone()
    if measured is None:raise ValueError('missing measured prompt')
    counts=tokenizers.measure(prompt);lengths=measure_text(prompt)
    if hashlib.sha256(prompt.encode()).digest()!=measured['prompt_sha256'] or any(counts[name]!=measured[name] for name in counts):raise ValueError('materialized text/hash/model-token mismatch')
    lengths.update({'model_token_count':counts['o200k_base'],'model_tokenizer_status':'measured',
        'model_tokenizer_id':'gpt-4o-2024-08-06/o200k_base/tiktoken-0.12.0'})
    result={'schema_version':2,'dataset_version':GRID_VERSION,'grid_query_id':query_id,'grid_design_id':design_id,
        'latent_instance_id':group['latent_instance_id'],'query_instance_id':record['instance_id'],
        'split':group['split'],'n':group['n'],**{k:record[k] for k in ('query_type','query_distance','query_subject','query_object','query_relation','query_direction','gold_label')},
        **{k:design[k] for k in ('name_control','relation_id','language','condition')},
        'presentation_plan_sha256':component['plan_sha256'],'prompt':prompt,
        'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'lengths':lengths,
        'model_token_counts':counts,'model_tokenizers':MODEL_TO_ENCODING,
        'factor_axes':json.loads(design['factor_axes']),'is_multi_transform':len(json.loads(design['factor_axes']))>1,
        'human_review_status':'pending','eligible_for_confirmatory_evaluation':False}
    result['prompt_id']='grid-'+digest([GRID_VERSION,record['instance_id'],design_id,result['prompt_sha256']])
    return result


def reconstruct_prompt(project_root,db_path,query_id,design_id,resource_dir=None):
    root=Path(project_root).resolve();tokenizers=ModelTokenizers(resource_dir or root/'tokenizers')
    with _connect(db_path,readonly=True) as connection:
        return _materialized(root,connection,query_id,design_id,tokenizers)


def export_grid(project_root,manifest_path,output,split=None,n=None,relation_id=None,limit=None,resource_dir=None):
    root,m,connection=_load_manifest(project_root,manifest_path,resource_dir)
    output=Path(output)
    if output.exists() or output.is_symlink():raise FileExistsError(output)
    if split is not None and split not in ('training','development','pilot','test'):raise ValueError('unknown split')
    if n is not None and (type(n) is not int or n<4):raise ValueError('invalid n')
    if relation_id is not None and relation_id not in RELATION_IDS:raise ValueError('unknown relation')
    if limit is not None and (type(limit) is not int or limit<1):raise ValueError('limit must be positive')
    sql='SELECT m.query_id,m.design_id FROM measurements m JOIN queries q ON q.id=m.query_id JOIN groups g ON g.id=q.group_id JOIN designs d ON d.id=m.design_id WHERE 1=1';args=[]
    for field,value in (('g.split',split),('g.n',n),('d.relation_id',relation_id)):
        if value is not None:sql+=' AND '+field+'=?';args.append(value)
    sql+=' ORDER BY m.query_id,m.design_id'
    if limit is not None:sql+=' LIMIT ?';args.append(limit)
    tokenizers=ModelTokenizers(resource_dir or root/'tokenizers');output.parent.mkdir(parents=True,exist_ok=True)
    count=0;sha=hashlib.sha256();owned=False
    try:
        with output.open('xb') as handle:
            owned=True
            for qid,did in connection.execute(sql,args):
                row=_materialized(root,connection,qid,did,tokenizers);payload=canonical_bytes(row)+b'\n';handle.write(payload);sha.update(payload);count+=1
    except BaseException:
        if owned:output.unlink(missing_ok=True)
        raise
    finally:connection.close()
    return {'records':count,'sha256':sha.hexdigest(),'bytes':output.stat().st_size}


def verify_grid(project_root,manifest_path,deep=True,resource_dir=None,workers=1):
    from .parallel import iter_groups,MAX_WORKERS
    if type(workers) is not int or not 1<=workers<=MAX_WORKERS:raise ValueError("invalid worker count")
    root,m,c=_load_manifest(project_root,manifest_path,resource_dir)
    try:
        _check_schema(c)
        if c.execute('PRAGMA application_id').fetchone()[0]!=APPLICATION_ID or c.execute('PRAGMA user_version').fetchone()[0]!=401:raise ValueError('database schema identity changed')
        if c.execute('PRAGMA integrity_check').fetchone()[0]!='ok' or c.execute('PRAGMA foreign_key_check').fetchone():raise ValueError('database integrity/foreign key failure')
        plan=_plan_from(m['plan']);expected=expected_counts(plan)
        if expected!=m['expected_counts']:raise ValueError('expected grid counts changed')
        if [tuple(row) for row in c.execute('SELECT * FROM designs ORDER BY id')]!=_design_rows() or [tuple(row) for row in c.execute('SELECT * FROM comparison_rules ORDER BY id')]!=_rules():raise ValueError('design/rule universe changed')
        if c.execute('SELECT count(*) FROM components').fetchone()[0]!=expected['totals']['orders']*243:raise ValueError('component cardinality changed')
        if any(row[0]!=243 for row in c.execute('SELECT count(*) FROM measurements GROUP BY query_id')):raise ValueError('missing design measurement')
        if c.execute('SELECT count(*) FROM measurements').fetchone()[0]!=expected['totals']['prompts']:raise ValueError('measurement cardinality changed')
        _check_prompt_isolation(c)
        coverage=_coverage(c)
        if coverage!=_read_json(Path(manifest_path).parent/'coverage.json'):raise ValueError('coverage cells changed')
        for name,counts in coverage['splits'].items():
            for key in ('orders','queries','prompts','comparison_cells','cells_by_arm'):
                if counts[key]!=expected['splits'][name][key]:raise ValueError('grid is not a complete Cartesian matrix')
        if not deep:return {'valid':True,'deep':False,'prompts':expected['totals']['prompts']}
        if m['base_binding']!=_base_binding(root):raise ValueError('original frozen sources changed')
        tokenizers=ModelTokenizers(resource_dir or root/'tokenizers')
        if m['tokenizers']!=tokenizers.descriptor() or m['reference_tokenizer']!=TOKENIZER_SPEC or m['source_files_sha256']!=_source_hashes():raise ValueError('source/tokenizer environment changed')
        group_id=query_id=0;logical=hashlib.sha256();seen_views={};seen_ids=set()
        legacy,legacy_meta=_legacy_groups(root/'dataset_manifest.json') if (root/'dataset_manifest.json').exists() else ({},None)
        if legacy_meta!=m['legacy_development_guard']:raise ValueError('legacy development guard changed')
        ordinals=Counter();spec_counts={spec.name:spec.count for spec in plan.splits}
        for result in iter_groups(plan,resource_dir or root/'tokenizers',workers):
            group_id=result['group_id'];split=result['split'];n=result['n']
            actual=c.execute('SELECT * FROM groups WHERE id=?',(group_id,)).fetchone()
            if actual is None or tuple(actual)!=result['group_row']:raise ValueError('source group/alias/plan replay mismatch')
            if result['ordering_id'] in seen_ids or 'id:'+result['ordering_id'] in legacy:raise ValueError('cross-split group leak')
            seen_ids.add(result['ordering_id'])
            for view in result['views']:
                if (view in seen_views and seen_views[view]!=split) or 'view:'+view in legacy:raise ValueError('cross-split alias leak')
                seen_views[view]=split
            actual=[tuple(row) for row in c.execute('SELECT * FROM queries WHERE group_id=? ORDER BY id',(group_id,))]
            if actual!=result['query_rows']:raise ValueError('source query/oracle replay mismatch')
            actual=[tuple(row) for row in c.execute('SELECT * FROM components WHERE group_id=? ORDER BY design_id',(group_id,))]
            if actual!=result['component_rows']:raise ValueError('component surface/prefix/plan replay mismatch')
            actual=[tuple(row) for row in c.execute('SELECT * FROM measurements WHERE group_id=? ORDER BY design_id,query_id',(group_id,))]
            if actual!=result['measurement_rows']:raise ValueError('prompt hash/BPE/length replay mismatch')
            for value in result['logical_digests']:logical.update(value)
            query_id=result['query_rows'][-1][0]
            ordinals[(split,n)]+=1
            if ordinals[(split,n)]%10==0 or ordinals[(split,n)]==spec_counts[split]:
                print(f'verified {split} n={n}: {ordinals[(split,n)]}/{spec_counts[split]} orders; surfaces and actual BPE counts',flush=True)
        if logical.hexdigest()!=m['construction_rows_sha256']:raise ValueError('logical measurement row digest changed')
        return {'valid':True,'deep':True,'orders':group_id,'queries':query_id,
            'prompts':expected['totals']['prompts'],'comparison_cells':expected['totals']['comparison_cells'],
            'construction_rows_sha256':logical.hexdigest(),'model_tokenizers':MODEL_TO_ENCODING}
    finally:c.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for command in ('build','verify','export'):
        p=sub.add_parser(command);p.add_argument('--project-root',type=Path,default=Path('.'))
        p.add_argument('--resources',type=Path)
        if command=='build':
            p.add_argument('--output',type=Path,required=True)
            p.add_argument('--config',type=Path,default=Path('configs/matched_dataset.toml'))
        else:p.add_argument('--manifest',type=Path,required=True)
        if command in ('build','verify'):p.add_argument('--workers',type=int,default=1)
        if command=='verify':p.add_argument('--hashes-only',action='store_true')
        if command=='export':
            p.add_argument('--output',type=Path,required=True);p.add_argument('--split');p.add_argument('--n',type=int);p.add_argument('--relation');p.add_argument('--limit',type=int)
    args=parser.parse_args(argv)
    try:
        if args.command=='build':
            m=build_grid(args.project_root,args.output,plan=load_plan(args.config),resource_dir=args.resources,workers=args.workers)
            result={'manifest':str(args.output/'manifest.json'),'counts':m['expected_counts']['totals']}
        elif args.command=='verify':result=verify_grid(args.project_root,args.manifest,not args.hashes_only,args.resources,args.workers)
        else:result=export_grid(args.project_root,args.manifest,args.output,args.split,args.n,args.relation,args.limit,args.resources)
        print(json.dumps(result,ensure_ascii=False))
    except (ValueError,OSError,sqlite3.Error,KeyError,TypeError) as error:parser.exit(2,f'error: {error}\n')


if __name__=='__main__':main()
