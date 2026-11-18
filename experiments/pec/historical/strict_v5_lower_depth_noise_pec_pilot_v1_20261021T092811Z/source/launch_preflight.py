"""Read-only frozen-input and runtime checks for the eight-job PEC pilot."""
from common import *
import ast,importlib,importlib.metadata,math
import numpy as np
from reference_kernel import array_hash
from pec_kernel import protocol

def preflight():
    frozen=check_frozen();p=load_json(OUT/'manifest/protocol.json');rows=jsonl(OUT/'manifest/runs.jsonl')
    assert frozen['protocol_sha256']==digest(OUT/'manifest/protocol.json')
    assert p['user_authorized'] and p['run_count']==len(rows)==8 and not p['safety']['test_access']
    assert [r['index'] for r in rows]==list(range(8)) and len({r['run_id'] for r in rows})==8
    assert p['patience']['epochs']==100 and p['patience']['min_delta']==0 and p['patience']['restore_best']
    assert p['max_epochs']==500 and p['max_concurrent_new_pec']==4
    assert [(r['depth'],r['teacher'],r['repetition']) for r in rows]==[(3,2,0),(3,2,1),(3,3,0),(3,3,1),(2,3,0),(2,3,1),(2,4,0),(2,4,1)]
    for r in rows:
        n=60 if r['depth']==3 else 100;pnoise=.005 if r['depth']==3 else .01
        assert r['kind']=='train' and r['method']=='explicit_sampled_pec' and r['mode']=='pec_dual'
        assert r['N_fit']==r['N_monitor']==r['N_confirm']==r['N_diagnostic']==n and r['N_gradient']==r['N_edge']==0
        assert r['noise_strength_1q']==r['noise_strength_2q']==pnoise
        assert r['patience_epochs']==100 and r['patience_min_delta']==0 and r['restore_best']
        assert r['max_epochs']==500 and r['max_updates']==12000 and r['monitor_every_updates']==24
        assert r['minimum_correct']==math.ceil(r['a_star']*384) and r['mu']==.1
        assert r['gradient_convention']=='exact_ideal_mean_vjp_at_signed_PEC_score'
        assert r['online_budget_raw_shots']==549144980
        for key in ['train_split','validation_split','initial_path','orders_path']:
            assert '/test/' not in r[key] and 'test_prediction' not in r[key]
        assert digest(r['initial_path'])==r['initial_file_sha256'] and digest(r['orders_path'])==r['orders_sha256']
        initial=np.load(r['initial_path'],allow_pickle=False);orders=np.load(r['orders_path'],allow_pickle=False)
        assert initial.shape==(4*r['depth']+3,4) and orders.shape==(500,1536)
        assert array_hash(initial)==r['initial_array_sha256'] and array_hash(orders)==r['orders_array_sha256']
        assert digest(r['teacher_path'])==r['teacher_sha256']
        assert digest(r['train_split'])==r['train_split_sha256'] and digest(r['validation_split'])==r['validation_split_sha256']
        q=protocol(n,r['depth'],pnoise,pnoise)
        assert q.raw_samples_per_expectation==n and q.gate_counts==(16*r['depth']+12,4*r['depth'])
    for path in (OUT/'source').glob('*.py'):
        ast.parse(path.read_text())
    runtime=load_json(OUT/'manifest/runtime.json')
    for name,version in runtime['versions'].items():assert importlib.metadata.version(name)==version
    for name,record in runtime['modules'].items():
        module=importlib.import_module(name)
        assert Path(module.__file__).resolve().is_relative_to(FROZEN.resolve())
        assert digest(module.__file__)==record['sha256']
    checks=load_json(OUT/'validation/deterministic_checks.json')
    assert checks['status']=='PASS' and checks['new_measurement_samples']==0
    return frozen

if __name__=='__main__':
    frozen=preflight();print(json.dumps(dict(utc=now(),status='PASS',run_count=8,files_verified=len(frozen['files']),protocol_sha256=digest(OUT/'manifest/protocol.json'),test_access=False),sort_keys=True))
