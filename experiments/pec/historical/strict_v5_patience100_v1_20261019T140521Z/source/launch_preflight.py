from common import *
import ast,importlib,importlib.metadata,math

def preflight():
    frozen=check_frozen();p=load_json(OUT/'manifest/protocol.json');rows=jsonl(OUT/'manifest/runs.jsonl')
    assert p['user_authorized'] and p['run_count']==len(rows)==30 and not p['safety']['test_access']
    assert [r['index'] for r in rows]==list(range(30)) and len({r['run_id'] for r in rows})==30
    assert p['patience']['epochs']==100 and p['patience']['min_delta']==0 and p['patience']['restore_best']
    for kind,n,count in [('train',120,6),('train',180,8),('reference_review',25,10),('pec_review_continue',120,4),('pec_review_continue',180,2)]:
        selected=[r for r in rows if r['kind']==kind and r['N_fit']==n];assert len(selected)==count
        for r in selected:
            assert r['N_fit']==r['N_monitor']==r['N_confirm']==n
            assert r['patience_epochs']==100 and r['patience_min_delta']==0 and r['restore_best']
            assert r['noise_strength_1q']==r['noise_strength_2q']==(0. if n==25 else .01) and r['mu']==.1
            assert r['minimum_correct']==math.ceil(r['a_star']*384)
            assert r['max_updates']==12000 and r['max_epochs']==500
            for key in ['train_split','validation_split','initial_path','orders_path']:assert '/test/' not in r[key] and 'test_prediction' not in r[key]
    assert not p['algorithm_budget_30M_applied'] and p['online_budget_raw_shots']==549144980
    assert all(r['online_budget_raw_shots']==549144980 for r in rows)
    reviews=load_json(OUT/'validation/reference_review_checks.json')
    assert len(reviews['checks'])==10 and all(c['passed'] for c in reviews['checks']) and reviews['new_measurement_samples']==0
    guards=load_json(OUT/'manifest/guards.json')
    assert [g['source_slurm_task'] for g in guards]==['5254249_0','5254249_1']
    for pth in list((OUT/'source').glob('*.py'))+list(FROZEN.rglob('*.py')):ast.parse(pth.read_text())
    runtime=load_json(OUT/'manifest/runtime.json')
    for name,version in runtime['versions'].items():assert importlib.metadata.version(name)==version
    for name in runtime['modules']:
        module=importlib.import_module(name)
        assert Path(module.__file__).resolve().is_relative_to(FROZEN.resolve())
        assert digest(module.__file__)==runtime['modules'][name]['sha256']
    checks=load_json(OUT/'validation/control_checks.json')
    assert all(c['passed'] for c in checks['checks']) and checks['new_measurement_samples']==0
    return dict(utc=now(),status='PASS',run_count=30,files_verified=len(frozen['files']),protocol_sha256=digest(OUT/'manifest/protocol.json'),frozen_sha256=digest(OUT/'manifest/frozen_execution.json'),test_access=False)
if __name__=='__main__':print(json.dumps(preflight(),sort_keys=True))
