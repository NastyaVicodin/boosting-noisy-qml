"""Read-only executable preflight: no circuit or measurement sampling."""
from common import *
from reference_kernel import array_hash
import collections,math
import numpy as np
import subprocess,sys
def main():
    check_frozen()
    rows=jsonl(OUT/'manifest/runs.jsonl');assert len(rows)==243
    counts=collections.Counter(r['method'] for r in rows)
    assert counts=={'reference':45,'adaboost_r':99,'real_adaboost':99}
    ids=set();pairs={}
    for i,r in enumerate(rows):
        assert r['index']==i and r['run_id'] not in ids
        ids.add(r['run_id'])
        assert r['N_fit']==r['N_monitor']==r['N_confirm']==25
        assert r['N_gradient']==0
        if r['method']=='reference':
            assert r['max_epochs']==500 and r['patience_epochs']==100
        else:
            assert r['max_epochs_per_weak']==500 and r['weak_patience_epochs']==20
        assert r['minimum_correct']==math.ceil(r['a_star']*384)
        assert '/test/' not in r['train_split'] and '/test/' not in r['validation_split']
        assert digest(r['teacher_path'])==r['teacher_sha256']
        assert digest(r['train_split'])==r['train_split_sha256']
        assert digest(r['validation_split'])==r['validation_split_sha256']
        assert digest(r['margin_target_registry'])==r['margin_target_registry_sha256']
        assert digest(r['classification_target_registry'])==r['classification_target_registry_sha256']
        assert not (OUT/'runs'/r['run_id']).exists()
        pairs.setdefault(r['pair_id'],r)
        b=pairs[r['pair_id']]
        assert all(r[k]==b[k] for k in ['initial_file_sha256','orders_sha256',
             'teacher_sha256','train_split_sha256','validation_split_sha256'])
    assert len(pairs)==45
    for r in pairs.values():
        assert digest(r['initial_path'])==r['initial_file_sha256']
        assert digest(r['orders_path'])==r['orders_sha256']
        theta=np.load(r['initial_path'],allow_pickle=False)
        order=np.load(r['orders_path'],allow_pickle=False)
        assert theta.shape==(4*r['depth']+3,4) and array_hash(theta)==r['initial_array_sha256']
        assert order.shape==(500,1536) and array_hash(order)==r['orders_array_sha256']
        assert np.all(np.sort(order,axis=1)==np.arange(1536))
    groups={}
    for group,n in [('reference',45),('core',90),('mid',72),('high',36)]:
        indices=[int(s) for s in (OUT/'manifest'/f'{group}_indices.txt').read_text().splitlines()]
        assert len(indices)==n and len(set(indices))==n
        groups[group]=indices
    assert sorted(sum(groups.values(),[]))==list(range(243))
    assert all(rows[i]['method']=='reference' for i in groups['reference'])
    assert all(rows[i]['method']!='reference' for g in ['core','mid','high'] for i in groups[g])
    result=subprocess.run([sys.executable,'-B',str(OUT/'source/test_control.py')],
        capture_output=True,text=True,check=True)
    assert 'synthetic dual-endpoint accounting PASS' in result.stdout
    new_json(OUT/'validation/preflight.json',dict(utc=now(),status='PASS',
        protocol_sha256=digest(OUT/'manifest/protocol.json'),
        frozen_sha256=digest(OUT/'manifest/frozen_execution.json'),
        run_count=len(rows),pairs=len(pairs),method_counts=dict(counts),
        scheduler_group_counts={g:len(v) for g,v in groups.items()},
        synthetic_control_stdout=result.stdout,
        test_arrays_opened=False,measurement_samples_generated=False))
    print('PRELAUNCH_PREFLIGHT_PASS')
if __name__=='__main__':main()
