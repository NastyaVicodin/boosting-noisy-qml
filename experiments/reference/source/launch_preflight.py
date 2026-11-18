from common import *
from reference_kernel import array_hash
import numpy as np,sys
rows=jsonl(OUT/'manifest/runs.jsonl')
check_frozen();assert len(rows)==45
assert all(r['method']=='reference' and r['index']==i for i,r in enumerate(rows))
for r in rows:
    assert not (OUT/'runs'/r['run_id']).exists()
    assert r['N_fit']==r['N_monitor']==r['N_confirm']==25
    assert r['patience_epochs']==100 and r['max_epochs']==500
    assert digest(r['initial_path'])==r['initial_file_sha256']
    assert digest(r['orders_path'])==r['orders_sha256']
    assert digest(r['train_split'])==r['train_split_sha256']
    assert digest(r['validation_split'])==r['validation_split_sha256']
    assert '/test/' not in r['train_split'] and '/test/' not in r['validation_split']
    theta=np.load(r['initial_path'],allow_pickle=False)
    assert theta.shape==(4*r['depth']+3,4) and array_hash(theta)==r['initial_array_sha256']
import runner,train_runner
new_json(OUT/'validation/preflight.json',dict(utc=now(),status='PASS',run_count=45,
    protocol_sha256=digest(OUT/'manifest/protocol.json'),
    frozen_sha256=digest(OUT/'manifest/frozen_execution.json'),
    import_preflight='runner and train_runner loaded; no missing PEC helper import',
    failed_v1_reference_runs_reused=False,
    new_measurement_samples=0,test_access=False))
print('REFERENCE_V2_PREFLIGHT_PASS')
