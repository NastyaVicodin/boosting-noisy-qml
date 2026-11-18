from common import *
import ast, importlib, importlib.metadata, math

def preflight():
    frozen=check_frozen();p=load_json(OUT/'manifest/protocol.json');rows=jsonl(OUT/'manifest/runs.jsonl')
    assert p['user_authorized'] and p['run_count']==len(rows)==10 and not p['safety']['test_access']
    assert [r['index'] for r in rows]==list(range(10)) and len({r['run_id'] for r in rows})==10
    assert {(r['teacher'],r['repetition']) for r in rows}=={(t,r) for t in range(5) for r in range(2)}
    assert p['patience']['epochs']==100 and p['patience']['min_delta']==0 and p['patience']['restore_best']
    base=Path(p['parent_campaign']);old={r['pair_id']:r for r in jsonl(base/'manifest/runs.jsonl') if r['N_fit']==210}
    for r in rows:
        assert r['kind']=='train' and r['N_fit']==r['N_monitor']==r['N_confirm']==r['N_diagnostic']==240
        assert r['patience_epochs']==100 and r['patience_min_delta']==0 and r['restore_best']
        assert r['noise_strength_1q']==r['noise_strength_2q']==.01 and r['mu']==.1
        assert r['minimum_correct']==math.ceil(r['a_star']*384)
        assert r['max_updates']==12000 and r['max_epochs']==500 and r['online_budget_raw_shots']==549144980
        assert r['gradient_convention']=='exact_ideal_mean_vjp_at_signed_PEC_score' and r['N_gradient']==r['N_edge']==0
        for key in ['train_split','validation_split','initial_path','orders_path','margin_target_registry','target_registry','epsilon_mar','minimum_correct','initial_array_sha256','orders_array_sha256']:
            assert r[key]==old[r['pair_id']][key],key
        for key in ['train_split','validation_split','initial_path','orders_path']:
            assert '/test/' not in r[key] and 'test_prediction' not in r[key]
    for name in ['train_runner.py','patience.py','audit.py','endpoints.py','lineage.py','reference_kernel.py','pec_measurements.py','common.py']:
        assert digest(OUT/'source'/name)==digest(base/'source'/name),name
    assert (OUT/'source/pec_kernel.py').read_text()==(base/'source/pec_kernel.py').read_text().replace('assert n in (120,180,210)','assert n in (120,180,210,240)')
    for path in list((OUT/'source').glob('*.py'))+list(FROZEN.rglob('*.py')):ast.parse(path.read_text())
    runtime=load_json(OUT/'manifest/runtime.json')
    for name,version in runtime['versions'].items():assert importlib.metadata.version(name)==version
    for name,record in runtime['modules'].items():
        m=importlib.import_module(name);assert Path(m.__file__).resolve().is_relative_to(FROZEN.resolve())
        assert digest(m.__file__)==record['sha256']
    tests=load_json(OUT/'validation/control_checks.json')
    assert all(c['passed'] for c in tests['checks']) and tests['new_measurement_samples']==0
    return dict(utc=now(),status='PASS',run_count=10,files_verified=len(frozen['files']),protocol_sha256=digest(OUT/'manifest/protocol.json'),frozen_sha256=digest(OUT/'manifest/frozen_execution.json'),test_access=False)
if __name__=='__main__':print(json.dumps(preflight(),sort_keys=True))
