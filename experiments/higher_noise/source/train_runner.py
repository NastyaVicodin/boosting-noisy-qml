from endpoints import *
from patience import Patience100,tracker_from_history
from lineage import save_state,load_theta
from pec_measurements import acquisition as pec_acquisition,evaluate as pec_evaluate
from pec_kernel import update as pec_update
from reference_kernel import make_step,make_evaluator,stream_keys,optimizer
import jax.numpy as jnp
import time
REFERENCE_EVALUATORS={}
def reference_evaluator(depth):
    if depth not in REFERENCE_EVALUATORS:
        REFERENCE_EVALUATORS[depth]=make_evaluator(25,depth)
    return REFERENCE_EVALUATORS[depth]

def reference_evaluate(theta,x,y,row,role,u,ledger,endpoint,phase='epoch'):
    n=row['N_monitor'] if role=='monitor' else row['N_confirm'];assert n==25
    ledger.ensure(len(x)*n);root=endpoint_seed('reference_patience100',row['pair_id'],endpoint,phase,role,u)
    keys=stream_keys(root,u,len(x));kernel=reference_evaluator(row['depth']);correct=0;loss=[];mean_loss=[];ids=[]
    for c in range(len(x)//384):
        xx=x[c*384:(c+1)*384];yy=y[c*384:(c+1)*384]
        eid=ledger.begin(role,384,n,endpoint=endpoint,event=f'{phase}:u{u}:chunk{c}',theta_sha256=array_hash(theta),stream_root_seed=root)
        start=time.perf_counter();scores,info=kernel(theta,jnp.asarray(xx),jnp.asarray(yy),keys[c*384:(c+1)*384]);jax.block_until_ready(scores)
        info=host(info);ledger.finish(eid,kernel_seconds=time.perf_counter()-start);assert info['finite']
        ledger.oracle('exact_ideal_mean_sampling_and_reporting',384,acquisition_id=eid,affects_stopping=False)
        ledger.reuse(eid,'endpoint accuracy/ramp metrics');ids.append(eid);correct+=info['correct'];loss.append(info['sampled_ramp_mu0p1']);mean_loss.append(info['mean_ramp_mu0p1'])
    sampled=float(np.mean(loss));return dict(endpoint=endpoint,sampled_ramp=sampled,exact_mean_ramp=float(np.mean(mean_loss)),correct=correct,accuracy=correct/len(x),
        passes=bool(sampled<=row['epsilon_mar'] if endpoint=='margin' else correct>=row['minimum_correct']),acquisition_ids=ids)

def run_train(row,directory,ledger,initial,orders,*,resume=None,tracker=None):
    assert row['patience_epochs']==100 and row['patience_min_delta']==0 and row['restore_best']
    x,y=load_split(row,'train_split');vx,vy=load_split(row,'validation_split');theta=jnp.asarray(initial);state=optimizer().init(theta)
    tracker=tracker or EndpointTracker(directory,ledger);patience=Patience100();best_theta=None;completed=0;epoch_losses=[];reason=None;decision=None;best_cp=None
    if resume:
        theta,state=resume['theta'],resume['state'];completed=resume['epoch']*24
        patience=tracker_from_history(resume['history']);decision=resume['history'][-1]
        best_theta,best_cp=resume['best_theta'],Path(resume['best_checkpoint'])
        assert not decision['should_stop'] and patience.epoch==resume['epoch']
    reference=row['method']=='reference';evaluate=reference_evaluate if reference else pec_evaluate
    reference_step=make_step(25,'historical_clip',row['depth']) if reference else None
    try:
        for u in range(completed+1,row['max_updates']+1):
            epoch,offset=divmod(u-1,24);ix=orders[epoch,64*offset:64*(offset+1)]
            if reference:
                root=row['reference_stream_seeds']['fit_seed'];eid=ledger.begin('fit',64,25,event=f'u{u}',theta_sha256=array_hash(theta),input_indices=ix.tolist(),stream_root_seed=root)
                theta,state,info=reference_step(theta,state,jnp.asarray(x[ix]),jnp.asarray(y[ix]),stream_keys(root,u,64));jax.block_until_ready(theta);info=host(info)
                ledger.finish(eid);ledger.oracle('exact_ideal_mean_historical_clip_vjp',64,update=u)
            else:
                scores,eid=pec_acquisition(theta,x[ix],row,'fit',f'u{u}',ledger,input_indices=ix.tolist())
                theta,state,info=pec_update(theta,state,jnp.asarray(x[ix]),jnp.asarray(y[ix]),scores);jax.block_until_ready(theta);info=host(info)
                ledger.oracle('exact_ideal_mean_vjp',64,update=u)
            assert info['gradient_finite'] and info['parameter_finite'];completed=u
            ledger.reuse(eid,'training loss, gradient and epoch-patience statistic')
            append_json(directory/'updates.jsonl',dict(update=u,epoch=epoch+1,theta_sha256=array_hash(theta),utc=now(),**info));epoch_losses.append(info['loss'])
            if u%24:continue
            decision=patience.update(sum(epoch_losses)/24,epoch+1);epoch_losses=[]
            if decision['improved']:
                best_theta=jnp.array(theta);best_cp=directory/f'state_u{u:05d}.npz'
            append_json(directory/'patience.jsonl',dict(utc=now(),**decision))
            checkpoint=directory/f'state_u{u:05d}.npz'
            save_state(checkpoint,theta,state)
            model=dict(checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),theta_array_sha256=array_hash(theta))
            for ep,xx,yy in [('margin',x,y),('classification',vx,vy)]:
                if ep not in tracker.hits:
                    tracker.check(ep,dict(update=u,epoch=epoch+1,phase='epoch'),model,
                        lambda role,ep=ep,xx=xx,yy=yy:evaluate(theta,xx,yy,row,role,u,ledger,ep))
            print(json.dumps(dict(epoch=epoch+1,best_epoch=patience.best_epoch,bad_epochs=patience.bad_epochs,first_hit_endpoints=list(tracker.hits),algorithm_raw_shots=algorithm_shots(ledger),utc=now())),flush=True)
            if len(tracker.hits)==2:reason='both_targets_confirmed';break
            if decision['should_stop']:reason='patience100';break
            if u==row['max_updates']:reason='epoch_cap';break
        restored=False
        if reason in ['patience100','epoch_cap']:
            assert best_theta is not None;theta=best_theta;restored=True
            assert best_cp is not None
            model=dict(checkpoint=str(best_cp),checkpoint_sha256=digest(best_cp),theta_array_sha256=array_hash(theta))
            for ep,xx,yy in [('margin',x,y),('classification',vx,vy)]:
                if ep not in tracker.hits:
                    tracker.check(ep,dict(update=completed,epoch=completed//24,selected_model_epoch=patience.best_epoch,phase='patience100_restored_best'),model,
                        lambda role,ep=ep,xx=xx,yy=yy:evaluate(theta,xx,yy,row,role,completed,ledger,ep,phase='patience100_restored_best'))
    except BudgetReached as exc:
        reason='online_budget';restored=False;append_json(directory/'budget_stop.jsonl',dict(utc=now(),message=str(exc)))
    return dict(status='BOTH_TARGETS_CONFIRMED' if len(tracker.hits)==2 else 'PATIENCE_STOP_PARTIAL_HIT' if tracker.hits and reason=='patience100' else 'PATIENCE_STOP_NONHIT' if reason=='patience100' else 'STOPPED_PARTIAL_HIT' if tracker.hits else 'STOPPED_NONHIT',
        stop_reason=reason,updates_completed=completed,fit_epochs=completed/24,patience=decision,best_epoch=patience.best_epoch,best_parameters_restored=restored,
        first_hits=tracker.hits,latest_checks=tracker.latest,algorithm_raw_shots=algorithm_shots(ledger),theta_sha256=array_hash(theta),**ledger.receipt())
