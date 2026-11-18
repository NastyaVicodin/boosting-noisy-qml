"""Freeze the 21 missing selected-teacher PEC210 trajectories; no sampling."""
from common import *
import collections
import numpy as np

HISTORICAL = ROOT / 'artifacts/strict_v5_pec210_depth_noise_explore_v1_20261022T204310Z'
BASE = ROOT / 'artifacts/strict_v5_lowshot_depth_noise_3x3_dual_v1_20261020T1030Z'
REF_FIX = ROOT / 'artifacts/strict_v5_lowshot_depth_noise_3x3_ref_fix_v2_20261020T104700Z'
PROPOSAL = ROOT / 'artifacts/strict_v5_pec_2v3_runtime_estimate_20261029T132235Z/PROPOSED_NEW_SLOTS.csv'
TARGET_TABLE = ROOT / 'artifacts/strict_v5_section_7_3_evidence_20261024T193535Z/SECTION_7_3_TARGETS.csv'
EXPECTED_PROPOSAL_SHA = 'fbf60eb1e02417b6cca9538ef53a6fd69f0986f96df4419882e028712b4dcf11'
CELLS = {(1, .02, 210), (2, .02, 210), (3, .005, 210)}
SOURCE_TEACHERS = {1: (1, 2, 4), 2: (2, 3, 4), 3: (2, 3, 4)}
MATCH_FIELDS = (
    'teacher_path', 'teacher_sha256', 'train_split', 'train_split_sha256',
    'validation_split', 'validation_split_sha256', 'initial_path',
    'initial_file_sha256', 'initial_array_sha256', 'orders_path',
    'orders_sha256', 'orders_array_sha256', 'epsilon_mar', 'minimum_correct',
    'a_star', 'margin_target_id', 'margin_target_registry',
    'margin_target_registry_sha256', 'classification_target_id',
    'classification_target_registry', 'classification_target_registry_sha256',
)

def array_hash(value):
    a = np.ascontiguousarray(np.asarray(value))
    return canonical(dict(dtype=str(a.dtype), shape=list(a.shape),
                          bytes_sha256=hashlib.sha256(a.tobytes()).hexdigest()))

assert digest(PROPOSAL) == EXPECTED_PROPOSAL_SHA
with PROPOSAL.open(newline='') as stream:
    proposed = list(csv.DictReader(stream))
selected = [
    v for v in proposed
    if (int(v['depth']), float(v['noise']), int(v['pec_shots'])) in CELLS
]
assert len(selected) == 21
assert collections.Counter(
    (int(v['depth']), float(v['noise']), int(v['pec_shots'])) for v in selected
) == {cell: 7 for cell in CELLS}
assert len({(int(v['depth']), float(v['noise']), int(v['source_teacher']),
             int(v['repetition'])) for v in selected}) == 21
assert len({v['proposed_pec_pair_id'] for v in selected}) == 21

baseline = jsonl(BASE / 'manifest/runs.jsonl')
corrected = jsonl(REF_FIX / 'manifest/runs.jsonl')
refs = {r['pair_id']: r for r in baseline if r['method'] == 'reference'}
fixed_refs = {r['pair_id']: r for r in corrected if r['method'] == 'reference'}
with TARGET_TABLE.open(newline='') as stream:
    targets = {
        (int(v['depth']), int(v['source_teacher'])): v
        for v in csv.DictReader(stream)
    }

rows = []
lineage = []
for entry in sorted(selected, key=lambda v: (
    int(v['depth']), float(v['noise']), int(v['source_teacher']),
    int(v['repetition'])
)):
    depth = int(entry['depth'])
    noise = float(entry['noise'])
    teacher = int(entry['source_teacher'])
    repetition = int(entry['repetition'])
    paper_teacher = int(entry['paper_teacher'])
    assert SOURCE_TEACHERS[depth][paper_teacher] == teacher
    pair = entry['comparison_pair_id']
    assert pair == f'd{depth}_t{teacher:02d}_r{repetition:02d}'
    assert pair in refs and pair in fixed_refs
    ref, fixed = refs[pair], fixed_refs[pair]
    assert all(ref[key] == fixed[key] for key in MATCH_FIELDS)
    target = targets[(depth, teacher)]
    assert ref['epsilon_mar'] == float(entry['margin_target']) == float(target['margin_threshold_original'])
    assert ref['minimum_correct'] == int(entry['classification_minimum_correct']) == int(target['minimum_correct_executed'])
    assert ref['train_split_sha256'] == entry['train_split_sha256']
    assert ref['validation_split_sha256'] == entry['validation_split_sha256']
    assert ref['initial_array_sha256'] == entry['initial_array_sha256']
    assert ref['orders_array_sha256'] == entry['orders_array_sha256']
    assert str(Path(ref['initial_path']).resolve()) == str(Path(entry['initial_path']).resolve())
    assert str(Path(ref['orders_path']).resolve()) == str(Path(entry['orders_path']).resolve())
    for key, hash_key in (
        ('teacher_path', 'teacher_sha256'),
        ('train_split', 'train_split_sha256'),
        ('validation_split', 'validation_split_sha256'),
        ('initial_path', 'initial_file_sha256'),
        ('orders_path', 'orders_sha256'),
        ('margin_target_registry', 'margin_target_registry_sha256'),
        ('classification_target_registry', 'classification_target_registry_sha256'),
    ):
        assert '/test/' not in ref[key] and 'test_prediction' not in ref[key]
        assert digest(ref[key]) == ref[hash_key], (pair, key)
    initial = np.load(ref['initial_path'], allow_pickle=False)
    orders = np.load(ref['orders_path'], allow_pickle=False)
    assert initial.shape == (4 * depth + 3, 4)
    assert orders.shape == (500, 1536)
    assert array_hash(initial) == ref['initial_array_sha256']
    assert array_hash(orders) == ref['orders_array_sha256']
    run_id = f'd{depth}_t{teacher:02d}_r{repetition:02d}_p{str(noise).replace(".", "p")}_pec210_expand'
    row = dict(ref)
    row.update(
        index=len(rows), run_id=run_id, kind='train', mode='pec_dual',
        method='explicit_sampled_pec', pair_id=entry['proposed_pec_pair_id'],
        comparison_pair_id=pair, paper_teacher=paper_teacher,
        N_fit=210, N_monitor=210, N_confirm=210, N_diagnostic=210,
        N_edge=0, N_gradient=0, noise_strength_1q=noise,
        noise_strength_2q=noise, mu=.1, batch_size=64,
        monitor_every_updates=24, max_epochs=500, max_updates=12000,
        patience_epochs=100, patience_min_delta=0., restore_best=True,
        gradient_convention='exact_ideal_mean_vjp_at_signed_PEC_score',
        online_budget_raw_shots=549144980,
        primary_endpoint='margin_and_classification',
        target_id=ref['margin_target_id'],
        target_registry=ref['classification_target_registry'],
        reference_run_id=fixed['run_id'],
        stream_definition=(
            'Frozen historical PEC210 source namespace; allocation-specific '
            'proposed pair_id and role/event root; independent PEC draws'
        ),
    )
    rows.append(row)
    lineage.append(dict(index=row['index'], run_id=run_id,
        proposal_pair_id=row['pair_id'], comparison_pair_id=pair,
        reference_run_id=fixed['run_id'], initial_file_sha256=ref['initial_file_sha256'],
        orders_sha256=ref['orders_sha256']))

assert len(rows) == 21 and len({r['run_id'] for r in rows}) == 21
with (OUT / 'manifest/runs.jsonl').open('x') as stream:
    for row in rows:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
new_json(OUT / 'manifest/lineage.json', dict(
    schema='strict_v5.pec_figure6.explore21.lineage.v1', utc=now(),
    historical_source=str(HISTORICAL), proposal=source_record(PROPOSAL),
    base_manifest=source_record(BASE / 'manifest/runs.jsonl'),
    corrected_reference_manifest=source_record(REF_FIX / 'manifest/runs.jsonl'),
    target_table=source_record(TARGET_TABLE), rows=lineage,
))
new_json(OUT / 'manifest/protocol.json', dict(
    schema='strict_v5.pec_figure6.explore21.v1', utc=now(),
    authorization='2026-10-30 user authorization: complete three PEC trajectories for each of three selected teachers in all eight Figure 6 configurations.',
    run_count=21, selected_cells=[dict(depth=d, noise=p, raw_pec_shots=n,
        missing_trajectories=7) for d,p,n in sorted(CELLS)],
    unchanged_science=True, source_campaign=str(HISTORICAL),
    frozen_source_namespace=NS,
    shot_allocations=dict(fit=210, monitor=210, confirm=210, diagnostic=210,
                          edge=0, gradient=0),
    gradient='Sampled explicit PEC forward scores; exact ideal-mean simulator VJP; zero acquired gradient shots.',
    objective='Unweighted minibatch arithmetic mean PsiG(y*sampled_PEC_score,0.5)',
    optimizer='clip_by_global_norm(1), Adam(lr=.01,b1=.9,b2=.999,eps=1e-8,eps_root=0)',
    batch_size=64, updates_per_epoch=24,
    patience=dict(epochs=100, min_delta=0, no_warmup=True,
                  strict_lower_best_epoch_mean=True, restore_best=True),
    max_epochs=500, max_updates=12000, online_budget_raw_shots=549144980,
    target_policy='Independently monitor and confirm each target on same trajectory; stop when both confirmed, patience, epoch cap, or original online cap.',
    accounting='Algorithm=fit+edge+measured-gradient; monitor and confirmation separate; completed acquisitions count toward online cap; raw PEC shots are not multiplied by gamma squared.',
    scheduler_plan=dict(partition='cpu-skylake', cpus_per_task=1,
        memory_gb_per_task=8, wall_hours=96, scientific_wall_clock_stop=False),
    safety=dict(test_access=False, automatic_retry=False,
                historical_files_edited=False),
))
old = load_json(HISTORICAL / 'manifest/runtime.json')
runtime = dict(utc=now(), executable=old['executable'],
               versions=old['versions'], modules={})
for name in old['modules']:
    path = OUT / 'frozen_src' / Path(*name.split('.')).with_suffix('.py')
    if name == 'numpy_compat':
        path = OUT / 'frozen_src/numpy_compat.py'
    assert path.exists(), path
    runtime['modules'][name] = source_record(path)
new_json(OUT / 'manifest/runtime.json', runtime)
paths = set((OUT / 'source').glob('*.py')) | set((OUT / 'source').glob('*.sh'))
paths |= set((OUT / 'frozen_src').rglob('*.py'))
paths |= {
    OUT / 'manifest/runs.jsonl', OUT / 'manifest/lineage.json',
    OUT / 'manifest/protocol.json', OUT / 'manifest/runtime.json',
    PROPOSAL, TARGET_TABLE,
    BASE / 'manifest/runs.jsonl', REF_FIX / 'manifest/runs.jsonl',
}
for row in rows:
    for key in (
        'teacher_path', 'train_split', 'validation_split',
        'initial_path', 'orders_path', 'margin_target_registry',
        'classification_target_registry'
    ):
        paths.add(Path(row[key]))
assert all('/test/' not in str(path) and 'test_prediction' not in path.name
           for path in paths)
new_json(OUT / 'manifest/frozen_execution.json', dict(
    utc=now(), protocol_sha256=digest(OUT / 'manifest/protocol.json'),
    files=[source_record(path) for path in sorted(paths)],
    policy='Freeze scientific source, data, targets, seeds, manifest, and runtime before any submission.',
))
print(json.dumps(dict(status='PREPARED', rows=len(rows),
    protocol_sha256=digest(OUT / 'manifest/protocol.json'),
    frozen_sha256=digest(OUT / 'manifest/frozen_execution.json'),
    measurements_drawn=0), sort_keys=True))
