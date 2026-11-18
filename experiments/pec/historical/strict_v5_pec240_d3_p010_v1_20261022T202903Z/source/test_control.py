"""Synthetic N240 control/accounting checks; no model evaluation or sampling."""
from endpoints import *
from unittest.mock import patch
import contextlib, io, tempfile, ast
import jax.numpy as jnp
import train_runner as tr
import pec_measurements as pm
import pec_kernel as pk

rows=jsonl(OUT/'manifest/runs.jsonl'); checks=[]
def record(name): checks.append(dict(test=name,passed=True,synthetic_only=True))
base=Path(load_json(OUT/'manifest/protocol.json')['parent_campaign'])
for name in ['train_runner.py','patience.py','audit.py','endpoints.py','lineage.py','reference_kernel.py','pec_measurements.py','common.py']:
    assert digest(OUT/'source'/name)==digest(base/'source'/name),name
old=(base/'source/pec_kernel.py').read_text()
assert (OUT/'source/pec_kernel.py').read_text()==old.replace('assert n in (120,180,210)','assert n in (120,180,210,240)')
assert pk.protocol(240).raw_samples_per_expectation==240
record('same_training_gradients_optimizer_stopping_and_accounting_only_allow_N240')
initial=np.zeros((15,4),np.float32); orders=np.tile(np.arange(1536,dtype=np.int32),(500,1))
def split(row,role):
    n=1536 if role=='train_split' else 384
    return np.zeros((n,4),np.float32),np.ones(n,np.float32)
for case in ['nonhit','restored_hit','early_class','both_early','epoch_cap']:
    with tempfile.TemporaryDirectory(dir=OUT/'validation') as tmp:
        d=Path(tmp);row=dict(rows[0]);led=Ledger(d,200000000);steps=[0];calls=[]
        if case=='epoch_cap':row['max_updates']=4*24
        def update(theta,state,x,y,scores):
            steps[0]+=1;epoch=(steps[0]-1)//24+1
            return theta+1,state,dict(parameter_finite=True,gradient_finite=True,loss={1:1.,2:.9,3:.8}.get(epoch,.85))
        def acquire(theta,x,row,role,event,ledger,**kw):
            eid=ledger.begin(role,len(x),row['N_fit'],event=event,synthetic=True,**kw);ledger.finish(eid)
            return jnp.zeros(len(x)),eid
        def evaluate(theta,x,y,row,role,u,ledger,ep,phase='epoch'):
            calls.append((ep,role,u,phase,float(theta[0,0])))
            eid=ledger.begin(role,len(x),240,endpoint=ep,synthetic=True);ledger.finish(eid)
            passed=(case=='restored_hit' and phase=='patience100_restored_best') or (case=='early_class' and ep=='classification') or (case=='both_early' and u>=48)
            return dict(passes=passed,acquisition_ids=[eid])
        with patch.object(tr,'load_split',split),patch.object(tr,'pec_acquisition',acquire),patch.object(tr,'pec_update',update),patch.object(tr,'pec_evaluate',evaluate),contextlib.redirect_stdout(io.StringIO()):
            result=tr.run_train(row,d,led,initial,orders)
        stop=2 if case=='both_early' else 4 if case=='epoch_cap' else 103
        assert result['updates_completed']==steps[0]==stop*24
        assert result['algorithm_raw_shots']==led.totals()['fit']==stop*1536*240
        if case=='both_early': assert result['stop_reason']=='both_targets_confirmed'
        else:
            assert result['stop_reason']==('epoch_cap' if case=='epoch_cap' else 'patience100')
            assert result['best_parameters_restored'] and result['best_epoch']==3
            restored=[v for v in calls if v[3]=='patience100_restored_best']
            assert restored and all(v[4]==72 and v[2]==stop*24 for v in restored)
        if case=='early_class':
            assert result['first_hits']['classification']['algorithm_raw_shots']==1536*240
            assert len([v for v in calls if v[0]=='classification'])==2
        if case=='restored_hit':
            assert len(result['first_hits'])==2
            assert all(v['algorithm_raw_shots']==103*1536*240 for v in result['first_hits'].values())
        record('N240_'+case+'_preserves_all_fit_cost_and_independent_first_hits')
with tempfile.TemporaryDirectory(dir=OUT/'validation') as tmp:
    led=Ledger(Path(tmp),1000000);seen=[]
    def predict(theta,x,root,stream,n):
        seen.append(n);return jnp.ones(len(x)),dict(raw_physical_shots=len(x)*n,simulator_propagations=1,padded_simulator_propagations=1)
    with patch.object(pm,'predict',predict):pm.acquisition(jnp.asarray(initial),np.zeros((64,4)),rows[0],'fit','u1',led)
    assert seen==[240] and led.totals()['fit']==64*240
    led.budget=led.online()+384*240-1;before=led.counter
    try:pm.evaluate(jnp.asarray(initial),np.zeros((384,4)),np.ones(384),rows[0],'confirm',24,led,'classification')
    except BudgetReached:pass
    else:raise AssertionError('Partial confirmation group allowed')
    assert led.counter==before
record('raw_shots_passed_once_and_confirmation_budget_checked_before_acquisition')
new_json(OUT/'validation/control_checks.json',dict(utc=now(),checks=checks,new_measurement_samples=0,new_model_optimizer_updates=0,fixed_parameter_PEC_validation_repeated=False))
print(json.dumps(dict(status='PASS',checks=len(checks),new_measurement_samples=0)))
