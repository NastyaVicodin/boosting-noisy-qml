"""Static launch guard for the 21 missing PEC210 trajectories; no sampling."""
from common import *
import collections
import numpy as np

HISTORICAL = ROOT / 'artifacts/strict_v5_pec210_depth_noise_explore_v1_20261022T204310Z'
BASE = ROOT / 'artifacts/strict_v5_lowshot_depth_noise_3x3_dual_v1_20261020T1030Z'
PILOT = ROOT / 'artifacts/strict_v5_lower_depth_noise_pec_pilot_v1_20261021T092811Z'
PROPOSAL = ROOT / 'artifacts/strict_v5_pec_2v3_runtime_estimate_20261029T132235Z/PROPOSED_NEW_SLOTS.csv'
TARGET_TABLE = ROOT / 'artifacts/strict_v5_section_7_3_evidence_20261024T193535Z/SECTION_7_3_TARGETS.csv'
CELLS = {(1, .02, 210), (2, .02, 210), (3, .005, 210)}
SOURCE_TEACHERS = {1: (1, 2, 4), 2: (2, 3, 4), 3: (2, 3, 4)}

def array_hash(value):
    a = np.ascontiguousarray(np.asarray(value))
    return canonical(dict(dtype=str(a.dtype), shape=list(a.shape),
                          bytes_sha256=hashlib.sha256(a.tobytes()).hexdigest()))

def preflight():
    frozen = check_frozen()
    protocol = load_json(OUT / 'manifest/protocol.json')
    rows = jsonl(OUT / 'manifest/runs.jsonl')
    assert frozen['protocol_sha256'] == digest(OUT / 'manifest/protocol.json')
    assert len(rows) == protocol['run_count'] == 21
    assert [r['index'] for r in rows] == list(range(21))
    assert len({r['run_id'] for r in rows}) == 21
    assert len({r['pair_id'] for r in rows}) == 21
    assert collections.Counter(
        (r['depth'], r['noise_strength_1q'], r['N_fit']) for r in rows
    ) == {cell: 7 for cell in CELLS}
    assert not protocol['safety']['test_access']
    targets = {
        (int(v['depth']), int(v['source_teacher'])): v
        for v in csv.DictReader(TARGET_TABLE.open(newline=''))
    }
    reference = {
        r['pair_id']: r for r in jsonl(BASE / 'manifest/runs.jsonl')
        if r['method'] == 'reference'
    }
    existing_pec60 = {
        (r['teacher'], r['repetition']): r
        for r in jsonl(PILOT / 'manifest/runs.jsonl')
        if r['method'] == 'explicit_sampled_pec'
        and r['depth'] == 3 and r['N_fit'] == 60
    }
    proposed = list(csv.DictReader(PROPOSAL.open(newline='')))
    assert len({r['proposed_pec_pair_id'] for r in proposed}) == len(proposed)
    proposed_pec60 = {
        (int(r['source_teacher']), int(r['repetition'])): r
        for r in proposed
        if int(r['depth']) == 3 and float(r['noise']) == .005
        and int(r['pec_shots']) == 60
    }
    for row in rows:
        assert SOURCE_TEACHERS[row['depth']][row['paper_teacher']] == row['teacher']
        assert row['kind'] == 'train' and row['mode'] == 'pec_dual'
        assert row['method'] == 'explicit_sampled_pec'
        assert row['N_fit'] == row['N_monitor'] == row['N_confirm'] == row['N_diagnostic'] == 210
        assert row['N_gradient'] == row['N_edge'] == 0
        assert row['noise_strength_1q'] == row['noise_strength_2q']
        assert row['batch_size'] == 64 and row['monitor_every_updates'] == 24
        assert row['patience_epochs'] == 100 and row['patience_min_delta'] == 0
        assert row['restore_best'] and row['max_epochs'] == 500
        assert row['max_updates'] == 12000
        assert row['online_budget_raw_shots'] == 549144980
        assert row['gradient_convention'] == 'exact_ideal_mean_vjp_at_signed_PEC_score'
        assert row['pair_id'] not in {
            r['pair_id'] for r in jsonl(HISTORICAL / 'manifest/runs.jsonl')
        }
        target = targets[(row['depth'], row['teacher'])]
        assert row['epsilon_mar'] == float(target['margin_threshold_original'])
        assert row['minimum_correct'] == int(target['minimum_correct_executed'])
        for key, hash_key in (
            ('teacher_path', 'teacher_sha256'),
            ('train_split', 'train_split_sha256'),
            ('validation_split', 'validation_split_sha256'),
            ('initial_path', 'initial_file_sha256'),
            ('orders_path', 'orders_sha256'),
            ('margin_target_registry', 'margin_target_registry_sha256'),
            ('classification_target_registry', 'classification_target_registry_sha256'),
        ):
            assert '/test/' not in row[key] and 'test_prediction' not in row[key]
            assert digest(row[key]) == row[hash_key], (row['run_id'], key)
        initial = np.load(row['initial_path'], allow_pickle=False)
        orders = np.load(row['orders_path'], allow_pickle=False)
        assert initial.shape == (4 * row['depth'] + 3, 4)
        assert orders.shape == (500, 1536)
        assert array_hash(initial) == row['initial_array_sha256']
        assert array_hash(orders) == row['orders_array_sha256']
        pair = row['comparison_pair_id']
        assert reference[pair]['initial_array_sha256'] == row['initial_array_sha256']
        assert reference[pair]['orders_array_sha256'] == row['orders_array_sha256']
        identities = [
            reference[f'd{row["depth"]}_t{row["teacher"]:02d}_r{rep:02d}']
            for rep in range(3)
        ]
        assert len({v['initial_array_sha256'] for v in identities}) == 3
        assert len({v['orders_array_sha256'] for v in identities}) == 3
        if row['depth'] == 3 and row['noise_strength_1q'] == .005:
            key = (row['teacher'], row['repetition'])
            comparator = existing_pec60.get(key)
            if comparator is not None:
                assert comparator['initial_array_sha256'] == row['initial_array_sha256']
                assert comparator['orders_array_sha256'] == row['orders_array_sha256']
            else:
                planned = proposed_pec60[key]
                assert planned['initial_array_sha256'] == row['initial_array_sha256']
                assert planned['orders_array_sha256'] == row['orders_array_sha256']
    for name in ('audit.py', 'lineage.py', 'patience.py', 'reference_kernel.py',
                 'train_runner.py', 'pec_measurements.py', 'pec_kernel.py',
                 'endpoints.py', 'environment.sh'):
        assert digest(OUT / 'source' / name) == digest(HISTORICAL / 'source' / name), name
    return dict(status='PASS', rows=len(rows), frozen_files=len(frozen['files']),
                protocol_sha256=digest(OUT / 'manifest/protocol.json'),
                frozen_sha256=digest(OUT / 'manifest/frozen_execution.json'),
                measurements_drawn=0, test_access=False)

if __name__ == '__main__':
    print(json.dumps(preflight(), sort_keys=True))
