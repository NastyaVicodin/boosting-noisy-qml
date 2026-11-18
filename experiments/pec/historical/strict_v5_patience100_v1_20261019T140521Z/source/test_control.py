"""Synthetic tests only: no quantum sampling, datasets or training computation."""
from endpoints import *
from unittest.mock import patch
import ast,contextlib,io,tempfile
import train_runner as tr
import review_runner as rr
import pec_measurements as pm
import guard
from patience import Patience100,scan_updates
from boosting_qml.training.convergence import BestLossNoImprovementConfig,BestLossNoImprovementTracker
import jax.numpy as jnp

rows=jsonl(OUT/'manifest/runs.jsonl');checks=[]
def record(name,**kw):checks.append(dict(test=name,passed=True,synthetic_only=True,**kw))
def split(row,role):
    n=1536 if role=='train_split' else 384
    return np.zeros((n,4),np.float32),np.ones(n,np.float32)

for losses in [[1.]*101,[1.,.9]+[.9]*100,[1.,.9,.8]+[.85]*100,[1./i for i in range(1,31)]]:
    new=Patience100();old=BestLossNoImprovementTracker(BestLossNoImprovementConfig(patience=100))
    for epoch,loss in enumerate(losses,1):
        a=new.update(loss,epoch);b=old.update(loss,epoch=epoch)
        for key in ['epoch','improved','best_epoch','epochs_without_improvement','should_stop']:assert a[key]==getattr(b,key)
    if len(losses)==101:assert a['should_stop'] and a['best_epoch']==1
record('strict_best_ties_reset_and_hundred_epoch_boundary_match_boosting')
try:Patience100().update(float('nan'),1)
except FloatingPointError:pass
else:raise AssertionError('Nonfinite loss not rejected')
record('nonfinite_loss_fails_closed')

orders=np.tile(np.arange(1536,dtype=np.int32),(500,1));initial=np.zeros((15,4),np.float32)
for index,case in [(0,'restore_hit'),(6,'nonhit'),(14,'restore_hit'),(0,'early_class'),(14,'both_early'),(6,'epoch_cap')]:
    with tempfile.TemporaryDirectory(dir=OUT/'validation') as tmp:
        d=Path(tmp);row=dict(rows[index]);n=row['N_fit'];ledger=Ledger(d,100000000);calls=[];steps=[0]
        if case=='epoch_cap':row['max_updates']=4*24
        def fake_update(theta,state,x,y,scores):
            steps[0]+=1;epoch=(steps[0]-1)//24+1
            loss={1:1.,2:.9,3:.8}.get(epoch,.85)
            return theta+1,state,dict(parameter_finite=True,gradient_finite=True,loss=loss)
        def fake_acquire(theta,x,row,role,event,led,**kw):
            eid=led.begin(role,len(x),n,event=event,synthetic=True,**kw);led.finish(eid)
            return jnp.zeros(len(x)),eid
        def fake_evaluate(theta,x,y,row,role,u,led,ep,phase='epoch'):
            calls.append((ep,role,u,phase,float(theta[0,0])))
            eid=led.begin(role,len(x),n,endpoint=ep,phase=phase,synthetic=True);led.finish(eid)
            passed=(case=='restore_hit' and phase=='patience100_restored_best') or (case=='early_class' and ep=='classification') or (case=='both_early' and u>=48)
            return dict(passes=passed,acquisition_ids=[eid],accuracy=float(passed),sampled_ramp=float(not passed))
        with patch.object(tr,'load_split',split),patch.object(tr,'pec_acquisition',fake_acquire),patch.object(tr,'pec_update',fake_update),patch.object(tr,'pec_evaluate',fake_evaluate),patch.object(tr,'reference_evaluate',fake_evaluate),patch.object(tr,'make_step',lambda *a:fake_update),patch.object(tr,'stream_keys',lambda *a:None),contextlib.redirect_stdout(io.StringIO()):
            result=tr.run_train(row,d,ledger,initial,orders)
        stop=2 if case=='both_early' else 4 if case=='epoch_cap' else 103
        assert steps[0]==result['updates_completed']==stop*24
        assert ledger.totals()['fit']==result['algorithm_raw_shots']==stop*1536*n
        assert len(list(d.glob('state_u*.npz')))==stop
        if case=='both_early':assert result['stop_reason']=='both_targets_confirmed' and not result['best_parameters_restored']
        else:
            assert result['stop_reason']==('epoch_cap' if case=='epoch_cap' else 'patience100')
            assert result['best_epoch']==3 and result['best_parameters_restored']
            restored=[c for c in calls if c[3]=='patience100_restored_best'];assert restored and all(c[4]==72. for c in restored)
            assert all(c[2]==stop*24 for c in restored)
        if case=='restore_hit':
            assert len(result['first_hits'])==2
            assert all(h['algorithm_raw_shots']==103*1536*n and h['position']['selected_model_epoch']==3 for h in result['first_hits'].values())
        if case=='early_class':
            assert result['first_hits']['classification']['algorithm_raw_shots']==1536*n
            assert len([c for c in calls if c[0]=='classification'])==2
        if case in ['nonhit','epoch_cap']:assert not result['first_hits'] and ledger.totals()['confirm']==0
        record(f'training_N{n}_{case}',stop_epoch=stop,fit_shots=ledger.totals()['fit'])

# Confirm real acquisition entry propagates N exactly once and rejects partial groups.
for index in [0,6]:
    with tempfile.TemporaryDirectory(dir=OUT/'validation') as tmp:
        led=Ledger(Path(tmp),1000000);row=rows[index];seen=[]
        def predict(theta,x,root,stream,n):
            seen.append((len(x),n));return jnp.ones(len(x)),dict(raw_physical_shots=len(x)*n,simulator_propagations=1,padded_simulator_propagations=1)
        with patch.object(pm,'predict',predict):pm.acquisition(jnp.asarray(initial),np.zeros((64,4)),row,'fit','u1',led)
        assert seen==[(64,row['N_fit'])] and led.totals()['fit']==64*row['N_fit']
        led.budget=led.online()+384*row['N_fit']-1;count=led.counter
        try:pm.evaluate(jnp.asarray(initial),np.zeros((384,4)),np.ones(384),row,'confirm',24,led,'classification')
        except BudgetReached:pass
        else:raise AssertionError('Partial group admitted')
        assert led.counter==count
record('raw_PEC_shots_once_and_atomic_endpoint_budget')

with tempfile.TemporaryDirectory(dir=OUT/'validation') as tmp:
    d=Path(tmp);(d/'state_u00504.npz').write_bytes(b'synthetic')
    append_json(d/'endpoint_checks.jsonl',dict(endpoint='classification',position=dict(epoch=2),confirmed=True))
    assert not guard.stopping_epoch_ready(d,21)
    append_json(d/'endpoint_checks.jsonl',dict(endpoint='margin',position=dict(epoch=21),confirmation_attempted=True))
    assert not guard.stopping_epoch_ready(d,21)
    append_json(d/'endpoint_checks.jsonl',dict(endpoint='margin',position=dict(epoch=21),confirmed=False))
    assert guard.stopping_epoch_ready(d,21)
    assert not guard.stopping_epoch_ready(d,22)
record('guard_waits_for_checkpoint_and_completed_final_epoch_checks')

old=ROOT/'artifacts/strict_v5_pec180_boost25_classification_v1_20261019T092910Z/source'
for name in ['pec_kernel.py','reference_kernel.py']:assert digest(old/name)==digest(OUT/'source'/name)
assert endpoint_seed('pec_dual','pair','margin','monitor','u24:chunk0')!=endpoint_seed('pec_dual','pair','margin','monitor','restored_best:u24:chunk0')
record('gradients_unchanged_and_restoration_uses_fresh_seed_namespace')

from lineage import save_state,load_state,source_ledger,verify_reference_overlap
from patience import tracker_from_history
from reference_kernel import optimizer
import optax
# Deterministic classical Adam state round trip; no model evaluation or sampling.
with tempfile.TemporaryDirectory(dir=OUT/'validation') as tmp:
    d=Path(tmp);theta=jnp.asarray(initial);state=optimizer().init(theta)
    for u in range(72):
        delta,state=optimizer().update(jnp.full_like(theta,.1),state,theta);theta=optax.apply_updates(theta,delta)
    cp=d/'state.npz';save_state(cp,theta,state);theta2,state2=load_state(cp,72)
    for a,b in zip(jax.tree_util.tree_leaves(state),jax.tree_util.tree_leaves(state2)):assert np.array_equal(a,b)
    delta1,next1=optimizer().update(jnp.full_like(theta,.2),state,theta)
    delta2,next2=optimizer().update(jnp.full_like(theta2,.2),state2,theta2)
    assert np.array_equal(delta1,delta2)
    try:load_state(cp,71)
    except AssertionError:pass
    else:raise AssertionError('Mismatched optimizer update counter admitted')
record('full_Adam_state_roundtrip_identical_next_classical_update_and_counter_validation')

with tempfile.TemporaryDirectory(dir=OUT/'validation') as tmp:
    d=Path(tmp);row=dict(rows[0],max_updates=5*24);n=row['N_fit'];led=Ledger(d,100000000)
    led.inherited_algorithm_raw_shots=3*1536*n;led.inherited_online_raw_shots=2*3*1536*n
    p=Patience100();history=[p.update(v,i) for i,v in enumerate([1.,.9,.8],1)]
    cp=d/'inherited_best.npz';save_state(cp,jnp.full_like(jnp.asarray(initial),72),optimizer().init(jnp.asarray(initial)))
    seen=[]
    def acquire(theta,x,row,role,event,ledger,**kw):
        seen.append(event);eid=ledger.begin(role,len(x),n,event=event);ledger.finish(eid);return jnp.zeros(len(x)),eid
    def update(theta,state,x,y,scores):return theta+1,state,dict(parameter_finite=True,gradient_finite=True,loss=.9)
    def evaluate(theta,x,y,row,role,u,ledger,ep,phase='epoch'):
        eid=ledger.begin(role,len(x),n,endpoint=ep);ledger.finish(eid)
        return dict(passes=ep=='classification',acquisition_ids=[eid])
    resume=dict(theta=jnp.full_like(jnp.asarray(initial),72),state=optimizer().init(jnp.asarray(initial)),epoch=3,history=history,
        best_theta=jnp.full_like(jnp.asarray(initial),72),best_checkpoint=str(cp))
    with patch.object(tr,'load_split',split),patch.object(tr,'pec_acquisition',acquire),patch.object(tr,'pec_update',update),patch.object(tr,'pec_evaluate',evaluate),contextlib.redirect_stdout(io.StringIO()):
        result=tr.run_train(row,d,led,initial,orders,resume=resume)
    assert seen[0]=='u73' and seen[-1]=='u120' and len(seen)==48
    assert result['algorithm_raw_shots']==5*1536*n and led.totals()['fit']==2*1536*n
    assert result['first_hits']['classification']['algorithm_raw_shots']==4*1536*n
    assert result['best_epoch']==3 and result['patience']['epochs_without_improvement']==2
record('continuation_preserves_global_updates_patience_history_best_checkpoint_and_inherited_costs')

# A source that ran beyond100-cutoff; existing checks reused without sampling.
with tempfile.TemporaryDirectory(dir=OUT/'validation') as tmp:
    root=Path(tmp);old=root/'old';(old/'manifest').mkdir(parents=True);source=old/'runs'/'synthetic';source.mkdir(parents=True)
    (old/'manifest/protocol.json').write_text('{}');(old/'manifest/frozen_execution.json').write_text('{"files":[]}')
    orig=Ledger(source,100000000)
    for u in range(1,24*103+1):
        epoch=(u-1)//24+1;theta=np.full((15,4),epoch,np.float32)
        eid=orig.begin('fit',64,120,event=f'u{u}');orig.finish(eid)
        append_json(source/'updates.jsonl',dict(update=u,loss=1.,theta_sha256=array_hash(theta)))
        if u%24==0:
            with (source/f'state_u{u:05d}.npz').open('xb') as f:np.savez(f,theta=theta)
            append_json(source/'evaluations.jsonl',dict(update=u,monitor=dict(passes=False)))
    prior=root/'prior.jsonl'
    for epoch in range(1,104):append_json(prior,dict(endpoint='classification',position=dict(epoch=epoch,phase='retrospective_epoch'),confirmed=epoch==102,
        confirmation_attempted=epoch==102,monitor=dict(passes=epoch==102),algorithm_raw_shots=epoch*1536*120))
    append_json(prior,dict(endpoint='classification',position=dict(epoch=21,phase='restored_best',selected_model_epoch=1),confirmed=True,monitor=dict(passes=True),algorithm_raw_shots=21*1536*120))
    d=root/'new';d.mkdir();led=Ledger(d,100000000);calls=[]
    row=dict(rows[24],source_campaign=str(old),source_run_id='synthetic',source_protocol_sha256=digest(old/'manifest/protocol.json'),
        source_frozen_sha256=digest(old/'manifest/frozen_execution.json'),prior_classification_sources=[source_record(prior)])
    before={str(p):digest(p) for p in old.rglob('*') if p.is_file()}
    def evaluate(theta,x,y,row,role,u,ledger,ep,phase='epoch'):
        assert phase=='patience100_restored_best' and float(theta[0,0])==1. and u==2424
        eid=ledger.begin(role,len(x),120,endpoint=ep);ledger.finish(eid);calls.append((ep,role))
        return dict(passes=ep=='classification',acquisition_ids=[eid])
    with patch.object(rr,'load_split',split),patch.object(rr,'evaluate',evaluate):result=rr.run_review(row,d,led)
    assert result['stop_epoch']==101 and result['best_epoch']==1
    assert result['algorithm_raw_shots']==101*1536*120 and result['original_completed_fit_outside_reused_prefix']==2*1536*120
    assert len(jsonl(d/'reused_classification_checks.jsonl'))==101 and led.totals()['fit']==0
    assert result['first_hits']['classification']['position']['epoch']==101
    assert before=={str(p):digest(p) for p in old.rglob('*') if p.is_file()}
    assert calls==[('margin','monitor'),('classification','monitor'),('classification','confirm')]
record('retrospective_cutoff_excludes_late_and_patience20_restored_hits_reuses_epoch_checks_preserves_actual_spend')

a=[dict(update=1,post_theta_sha256='a',loss=1.)];b=[dict(update=1,post_theta_sha256='a',loss=1.)]
assert verify_reference_overlap(dict(margin=a,classification=b))==1
try:verify_reference_overlap(dict(margin=a,classification=[dict(update=1,post_theta_sha256='b',loss=1.)]))
except AssertionError:pass
else:raise AssertionError('Nonidentical reference trajectories accepted')
record('reference_merge_requires_identical_shared_parameter_hashes_and_losses')

new_json(OUT/'validation/control_checks.json',dict(utc=now(),checks=checks,new_measurement_samples=0,new_model_optimizer_updates=0,
    synthetic_classical_Adam_roundtrip=True,synthetic_training_steps_only=True,fixed_parameter_PEC_validation_repeated=False))
print(json.dumps(dict(status='PASS',checks=len(checks))))
