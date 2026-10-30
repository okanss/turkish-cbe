"""Full metadata/seed checks plus bounded surface replay of the frozen bank.

Automatic replay cannot repeat a human or AI naturalness assessment. The initial
bounded AI assessment is recorded separately in RESEARCH_AUDIT.md.
"""
from pathlib import Path
import sqlite3, json, hashlib, time, collections, csv, sys
from data_generation.generator import GenerationPlan, generate_record, canonical_bytes, valid_distances, DEFAULT_DISTANCES
from data_generation.validation import validate_record
from experiments.names import name_maps
from complete_grid.dataset_builder import _component, _record, _suffix, _design_rows
from complete_grid.surface_reader import SurfaceReader
from validation.semantic_equivalence import normalize_statement, normalize_query
from renderers.morphology import genitive, ablative
from inference.prompts import make_messages

def run_data_checks(root):
    ROOT = Path(root).resolve()
    from inference.dataset import GridDataset
    with GridDataset(ROOT, 'data/complete_grid/manifest.json'):
        pass
    start = time.monotonic()
    con = sqlite3.connect(f'file:{ROOT}/data/complete_grid/grid.sqlite?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    con.execute('PRAGMA query_only=ON')

    def sha(p):
        h = hashlib.sha256()
        with Path(p).open('rb') as f:
            for b in iter(lambda: f.read(1024 * 1024), b''):
                h.update(b)
        return h.hexdigest()
    report = {'artifact_type': 'independent_data_language_audit', 'contains_model_results': False, 'scope': 'All frozen source groups/queries/metadata; all component normal forms and cross-split prompt hashes; stratified bounded surface replays; AI inspection is not human linguistic approval.', 'dataset_manifest_sha256': sha(ROOT / 'data/complete_grid/manifest.json'), 'checks': {}, 'limitations': []}
    checks = report['checks']
    checks['sqlite_quick_check'] = con.execute('PRAGMA quick_check').fetchone()[0]
    checks['foreign_key_violations'] = len(con.execute('PRAGMA foreign_key_check').fetchall())
    checks['counts'] = {table: con.execute('SELECT COUNT(*) FROM ' + table).fetchone()[0] for table in ['groups', 'queries', 'designs', 'components', 'measurements', 'comparison_rules']}
    print('metadata initial done', flush=True)
    sources = {}
    ranksets = {}
    order_splits = {}
    alias_splits = {}
    alias_count = 0
    order_dups = []
    alias_dups = []
    replay_count = 0
    for g in con.execute('SELECT * FROM groups ORDER BY id'):
        source = json.loads(g['source_json'])
        e = source['endpoint']
        validate_record(e)
        if not (e['ordering_id'] == g['latent_instance_id'] and e['n'] == g['n']):
            raise ValueError('Frozen data audit check failed at original probe line 42')
        if not len(e['latent_order']) == len(set(e['latent_order'])) == e['n']:
            raise ValueError('Frozen data audit check failed at original probe line 43')
        if not {tuple(p) for p in e['premise_edges']} == set(zip(e['latent_order'], e['latent_order'][1:])):
            raise ValueError('Frozen data audit check failed at original probe line 44')
        key = tuple(e['latent_order'])
        if key in order_splits and order_splits[key] != g['split']:
            order_dups.append(g['id'])
        order_splits[key] = g['split']
        for control, maps in source['name_maps'].items():
            for lang, aliases in maps.items():
                key = (control, lang, tuple((aliases[x] for x in e['latent_order'])))
                if key in alias_splits and alias_splits[key] != g['split']:
                    alias_dups.append(g['id'])
                alias_splits[key] = g['split']
                alias_count += 1
        if not source['name_maps'] == name_maps(e):
            raise ValueError('Frozen data audit check failed at original probe line 53')
        n = g['n']
        plan = GenerationPlan(seed=e['seed'], sizes=(n,), examples_per_size=4 * (((200 if g['split'] == 'test' else 50) + 3) // 4), entity_types=('turkish_names',))
        if not generate_record(plan, n, g['sample_index'], 'endpoint') == e:
            raise ValueError('Frozen data audit check failed at original probe line 57')
        replay_count += 1
        sources[g['id']] = source
        ranksets[g['id']] = {x: i for i, x in enumerate(e['latent_order'])}
    checks['source_chains_checked'] = len(sources)
    checks['seed_endpoint_replays'] = replay_count
    checks['cross_split_exact_source_order_duplicates'] = len(order_dups)
    checks['cross_split_aliased_order_duplicates'] = len(alias_dups)
    checks['aliased_orders_checked'] = alias_count
    balance = collections.defaultdict(collections.Counter)
    qtypes = collections.Counter()
    queryreplays = 0
    query_failures = []
    for q in con.execute('SELECT q.*,g.split,g.n,g.sample_index FROM queries q JOIN groups g ON g.id=q.group_id ORDER BY q.id'):
        e = sources[q['group_id']]['endpoint']
        ranks = ranksets[q['group_id']]
        i, j = (ranks[q['query_subject']], ranks[q['query_object']])
        gold = i < j if q['query_relation'] == 'before' else i > j
        if not (gold == bool(q['gold_label']) and abs(i - j) == q['query_distance']):
            raise ValueError('Frozen data audit check failed at original probe line 67')
        if not q['query_direction'] == ('forward' if i < j else 'reverse'):
            raise ValueError('Frozen data audit check failed at original probe line 68')
        if q['query_type'] == 'endpoint':
            if not {i, j} == {0, q['n'] - 1}:
                raise ValueError('Frozen data audit check failed at original probe line 69')
        elif not (1 <= i <= q['n'] - 2 and 1 <= j <= q['n'] - 2):
            raise ValueError('Frozen data audit check failed at original probe line 70')
        key = (q['split'], q['n'], q['query_type'], q['query_distance'] if q['query_type'] == 'distance_controlled' else None)
        balance[key][q['query_direction'] + '|' + str(gold)] += 1
        qtypes[q['query_type']] += 1
        n = q['n']
        plan = GenerationPlan(seed=e['seed'], sizes=(n,), examples_per_size=4 * (((200 if q['split'] == 'test' else 50) + 3) // 4), entity_types=('turkish_names',))
        record = generate_record(plan, n, q['sample_index'], q['query_type'], q['query_distance'] if q['query_type'] == 'distance_controlled' else None)
        if not _record(sources[q['group_id']], q) == record:
            raise ValueError('Frozen data audit check failed at original probe line 75')
        queryreplays += 1
    checks['query_rank_gold_distance_replays'] = queryreplays
    checks['query_types'] = dict(qtypes)
    checks['balanced_strata_checked'] = len(balance)
    checks['balance_max_cell_difference'] = max((max(v.values()) - min(v.values()) for v in balance.values()))
    checks['test_every_cell_50'] = all((set(v.values()) == {50} and len(v) == 4 for k, v in balance.items() if k[0] == 'test'))
    checks['non_test_cells_12_or_13'] = all((set(v.values()) == {12, 13} and len(v) == 4 for k, v in balance.items() if k[0] != 'test'))
    checks['balance'] = [{'split': k[0], 'n': k[1], 'query_type': k[2], 'controlled_distance': k[3], 'cells': dict(sorted(v.items()))} for k, v in sorted(balance.items(), key=lambda x: str(x[0]))]
    print('sources/queries replay done', flush=True)
    expected_normal = {gid: hashlib.sha256(canonical_bytes(sorted((tuple(x) for x in s['endpoint']['premise_edges'])))).digest() for gid, s in sources.items()}
    component_count = 0
    for row in con.execute('SELECT group_id,normal_form_sha256 FROM components'):
        if not row['normal_form_sha256'] == expected_normal[row['group_id']]:
            raise ValueError('Frozen data audit check failed at original probe line 86')
        component_count += 1
    checks['component_normal_forms_verified'] = component_count
    checks['query_design_cardinality_errors'] = con.execute('SELECT COUNT(*) FROM (SELECT query_id,count(*) k FROM measurements GROUP BY query_id HAVING k!=243)').fetchone()[0]
    checks['bilingual_presentation_plan_mismatches'] = con.execute("SELECT count(*) FROM components a JOIN designs da ON da.id=a.design_id JOIN designs db ON db.name_control=da.name_control AND db.relation_id=da.relation_id AND db.condition=da.condition AND db.language='tr' JOIN components b ON b.group_id=a.group_id AND b.design_id=db.id WHERE da.language='en' AND a.plan_sha256!=b.plan_sha256").fetchone()[0]
    checks['cross_split_visible_prompt_hash_overlap'] = con.execute('SELECT count(*) FROM (SELECT m.prompt_sha256 FROM measurements m JOIN groups g ON g.id=m.group_id GROUP BY m.prompt_sha256 HAVING min(g.split)!=max(g.split))').fetchone()[0]
    print('all db metadata/isolation done', flush=True)
    selected = [row[0] for row in con.execute('SELECT MIN(id) FROM groups GROUP BY split,n ORDER BY split,n')]
    designs = _design_rows()
    clause_count = 0
    component_replays = 0
    question_count = 0
    surface_sha = hashlib.sha256()
    prompt_mismatches = 0
    for gid in selected:
        s = sources[gid]
        cache = {}
        for design in designs:
            comp = _component(s, design, cache)
            component_replays += 1
            clause_count += s['endpoint']['n'] - 1
            stored = con.execute('SELECT prefix_sha256 FROM components WHERE group_id=? AND design_id=?', (gid, design[0])).fetchone()[0]
            if not hashlib.sha256(comp['prefix'].encode()).digest() == stored:
                raise ValueError('Frozen data audit check failed at original probe line 102')
            for typ in ['endpoint', 'internal', 'distance_controlled']:
                q = con.execute('SELECT * FROM queries WHERE group_id=? AND query_type=? ORDER BY id LIMIT 1', (gid, typ)).fetchone()
                record = _record(s, q)
                text = comp['prefix'] + _suffix(comp, record)
                actual = hashlib.sha256(text.encode()).digest()
                stored = con.execute('SELECT prompt_sha256 FROM measurements WHERE query_id=? AND design_id=?', (q['id'], design[0])).fetchone()[0]
                if not actual == stored:
                    raise ValueError('Frozen data audit check failed at original probe line 109')
                surface_sha.update(actual)
                question_count += 1
    checks['stratified_surface_source_groups'] = len(selected)
    checks['stratified_actual_surface_components_replayed'] = component_replays
    checks['stratified_actual_surface_clauses_reparsed'] = clause_count
    checks['stratified_actual_prompt_hashes_replayed'] = question_count
    checks['stratified_replay_sha256'] = surface_sha.hexdigest()
    print('stratified actual surface/hash replay done', flush=True)
    review_path = ROOT / 'data/validation/turkish_samples.csv'
    rows = list(csv.DictReader(review_path.open(newline='')))
    review_manifest = json.loads((review_path.parent / 'turkish_samples_manifest.json').read_bytes())
    if not sha(review_path) == review_manifest['csv_sha256']:
        raise ValueError('Frozen data audit check failed at original probe line 119')
    review_clauses = 0
    for row in rows:
        line = next((l for l in row['canonical_prompt'].splitlines() if l.startswith('Varlıklar: ')))
        names = tuple(line[len('Varlıklar: '):-1].split(', '))
        source_edges = [normalize_statement(text, names, row['relation_id']) for text in row['canonical_premises'].splitlines()]
        edited_edges = [normalize_statement(text, names, row['relation_id']) for text in row['rendered_premises'].splitlines()]
        if not (set(source_edges) == set(edited_edges) and len(source_edges) == len(edited_edges)):
            raise ValueError('Frozen data audit check failed at original probe line 127')
        if not normalize_query(row['query'], names, row['relation_id']) == normalize_query(row['canonical_query'], names, row['relation_id']):
            raise ValueError('Frozen data audit check failed at original probe line 128')
        review_clauses += len(edited_edges)
        if not row['human_review_status'] == 'pending':
            raise ValueError('Frozen data audit check failed at original probe line 130')
        if not all((not row[k] for k in row if k.startswith('reviewer'))):
            raise ValueError('Frozen data audit check failed at original probe line 131')
    checks['linguistic_review_csv'] = {'rows': len(rows), 'source_contexts': len(set((r['ordering_id'] for r in rows))), 'relations': dict(collections.Counter((r['relation_id'] for r in rows))), 'variants': dict(collections.Counter((r['variant'] for r in rows))), 'all_surface_premises_reparsed': review_clauses, 'human_reviews_completed': 0, 'human_status': 'pending', 'AI_screen': {'inspected_rows': 216, 'inspection_unit': 'first premise + complete query for all P0/P2/P3/P4/P5/P6 rows; P1 inspected computationally', 'status': 'no definite grammar error detected in bounded AI screen; marked P3 postverbal subjects and spatial-P4 position descriptions need native human naturalness judgments', 'human_approval': False}, 'same_renderer_and_morphology_shared_by_parser': True}
    fixtures = {'Ayşe': ("Ayşe'nin", "Ayşe'den"), 'Bora': ("Bora'nın", "Bora'dan"), 'İpek': ("İpek'in", "İpek'ten"), 'Çetin': ("Çetin'in", "Çetin'den"), 'Yavuz': ("Yavuz'un", "Yavuz'dan"), 'Gül': ("Gül'ün", "Gül'den"), 'Öztürk': ("Öztürk'ün", "Öztürk'ten"), 'Koç': ("Koç'un", "Koç'tan"), 'E0003': ("E0003'ün", "E0003'ten"), 'E0004': ("E0004'ün", "E0004'ten"), 'E0006': ("E0006'nın", "E0006'dan"), 'E0009': ("E0009'un", "E0009'dan")}
    for name, expected in fixtures.items():
        if not (genitive(name), ablative(name)) == expected:
            raise ValueError('Frozen data audit check failed at original probe line 140')
    checks['independent_suffix_gold_fixtures'] = len(fixtures)
    report['limitations'] = ['No native human linguistic approval is claimed. Existing 252 rows cover 36 source contexts at n=4/8 and remain unannotated.', 'Closed parser shares suffix routines with renderer; independently specified 12 suffix fixtures mitigate but do not remove shared-system error risk.', 'Exact ID/order/text isolation excludes duplicate instances; all strict total-order chains are abstractly isomorphic by design. This is not distribution-free out-of-family generalization.', 'Endpoint n-1 distance is a latent graph distance, not a demonstrated lower bound on model computational reasoning depth. Degree-based endpoint shortcuts remain possible.', 'Language and presentation lengths/tokenization remain design confounds, and P4/P5 are multi-feature interventions.', 'Large evaluation now requires two-reviewer, dataset-bound linguistic evidence; no such real approval is supplied.']
    report['elapsed_seconds'] = round(time.monotonic() - start, 3)
    report['checks']['linguistic_review_csv']['AI_screen'] = {'status': 'not performed by this automatic replay', 'human_approval': False, 'initial_audit_screen_reference': 'RESEARCH_AUDIT.md: bounded AI inspection of 216 rows; not a human review'}
    shortcuts=0
    for source in sources.values():
        item=source['endpoint']
        if endpoint_shortcut(item['premise_edges'],item['query_subject'],item['query_object'],item['query_relation']) != item['gold_label']:
            raise ValueError('degree-only endpoint shortcut mismatch')
        shortcuts+=1
    checks['degree_only_endpoint_shortcuts_verified']=shortcuts
    con.close()
    return report


def endpoint_shortcut(premise_edges, subject, obj, relation):
    """Degree-only endpoint answer; no transitive-path traversal or LLM."""
    if relation not in {"before","after"}:raise ValueError("strict-order relation required")
    subjects={a for a,b in premise_edges};objects={b for a,b in premise_edges}
    first,last=subjects-objects,objects-subjects
    if len(first)!=1 or len(last)!=1:raise ValueError("a directed chain is required")
    first,last=next(iter(first)),next(iter(last))
    if {subject,obj}!={first,last}:return None
    return (subject==first) if relation=="before" else (subject==last)
