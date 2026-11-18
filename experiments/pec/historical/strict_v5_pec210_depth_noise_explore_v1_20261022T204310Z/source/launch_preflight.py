"""Verify the frozen ten-run PEC210 campaign and its saved-data pairing."""
from common import *
import ast,importlib,importlib.metadata
import numpy as np
from reference_kernel import array_hash
from pec_kernel import protocol

CELLS=[(1,1,.02),(2,2,.02),(3,3,.005),(4,0,.01),(5,1,.005)]
MIN_CORRECT={(1,1):380,(2,2):374,(3,3):319,(4,0):320,(5,1):299}

def preflight():
    frozen=check_frozen();p=load_json(OUT/'manifest/protocol.json');rows=jsonl(OUT/'manifest/runs.jsonl')
    assert frozen['protocol_sha256']==digest(OUT/'manifest/protocol.json')
    assert p['user_authorized'] and p['run_count']==len(rows)==10 and not p['safety']['test_access']
    assert [r['index'] for r in rows]==list(range(10)) and len({r['run_id'] for r in rows})==10
    assert p['patience']['epochs']==100 and p['patience']['min_delta']==0 and p['patience']['restore_best']
    assert p['max_epochs']==500 and p['max_updates']==12000
    expected=[(d,t,rep,noise) for d,t,noise in CELLS for rep in (0,1)]
    assert [(r['depth'],r['teacher'],r['repetition'],r['noise_strength_1q']) for r in rows]==expected
    base=Path(p['baseline_manifest']);refs={r['pair_id']:r for r in jsonl(base) if r['method']=='reference'}
    for row in rows:
        assert row['kind']=='train' and row['method']=='explicit_sampled_pec' and row['mode']=='pec_dual'
        assert row['N_fit']==row['N_monitor']==row['N_confirm']==row['N_diagnostic']==210
        assert row['N_gradient']==row['N_edge']==0
        assert row['noise_strength_1q']==row['noise_strength_2q']
        assert row['patience_epochs']==100 and row['patience_min_delta']==0 and row['restore_best']
        assert row['max_epochs']==500 and row['max_updates']==12000 and row['monitor_every_updates']==24
        assert row['minimum_correct']==MIN_CORRECT[(row['depth'],row['teacher'])] and row['mu']==.1
        assert row['gradient_convention']=='exact_ideal_mean_vjp_at_signed_PEC_score'
        assert row['online_budget_raw_shots']==549144980
        ref=refs[row['pair_id']]
        for key in ['teacher_path','teacher_sha256','train_split','train_split_sha256',
                    'validation_split','validation_split_sha256','initial_path','initial_file_sha256',
                    'initial_array_sha256','orders_path','orders_sha256','orders_array_sha256',
                    'epsilon_mar','minimum_correct','a_star','margin_target_id','margin_target_registry',
                    'margin_target_registry_sha256','classification_target_id','classification_target_registry',
                    'classification_target_registry_sha256']:
            assert row[key]==ref[key],(row['pair_id'],key)
        for key in ['train_split','validation_split','initial_path','orders_path']:
            assert '/test/' not in row[key] and 'test_prediction' not in row[key]
        assert digest(row['initial_path'])==row['initial_file_sha256']
        assert digest(row['orders_path'])==row['orders_sha256']
        initial=np.load(row['initial_path'],allow_pickle=False)
        orders=np.load(row['orders_path'],allow_pickle=False)
        assert initial.shape==(4*row['depth']+3,4) and orders.shape==(500,1536)
        assert array_hash(initial)==row['initial_array_sha256'] and array_hash(orders)==row['orders_array_sha256']
        assert digest(row['teacher_path'])==row['teacher_sha256']
        assert digest(row['train_split'])==row['train_split_sha256']
        assert digest(row['validation_split'])==row['validation_split_sha256']
        q=protocol(210,row['depth'],row['noise_strength_1q'],row['noise_strength_2q'])
        assert q.raw_samples_per_expectation==210 and q.gate_counts==(16*row['depth']+12,4*row['depth'])
    pilot=Path(p['source_campaign'])
    for name in ['audit.py','lineage.py','patience.py','reference_kernel.py',
                 'train_runner.py','pec_measurements.py']:
        assert digest(OUT/'source'/name)==digest(pilot/'source'/name),name
    for path in (OUT/'source').glob('*.py'):ast.parse(path.read_text())
    runtime=load_json(OUT/'manifest/runtime.json')
    for name,version in runtime['versions'].items():assert importlib.metadata.version(name)==version
    for name,record in runtime['modules'].items():
        module=importlib.import_module(name)
        assert Path(module.__file__).resolve().is_relative_to(FROZEN.resolve())
        assert digest(module.__file__)==record['sha256']
    checks=load_json(OUT/'validation/deterministic_checks.json')
    assert checks['status']=='PASS' and checks['new_measurement_samples']==0
    assert {c['depth'] for c in checks['checks']}==set(range(1,6))
    return dict(utc=now(),status='PASS',run_count=10,files_verified=len(frozen['files']),
                protocol_sha256=digest(OUT/'manifest/protocol.json'),
                frozen_sha256=digest(OUT/'manifest/frozen_execution.json'),test_access=False)

if __name__=='__main__':print(json.dumps(preflight(),sort_keys=True))
