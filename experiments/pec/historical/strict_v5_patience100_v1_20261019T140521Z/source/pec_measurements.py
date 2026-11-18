from endpoints import *
from pec_kernel import *
import time

def acquisition(theta,x,row,role,event,ledger,*,endpoint=None,**metadata):
    n=row['N_fit'] if role=='fit' else row['N_monitor'] if role=='monitor' else row['N_confirm']
    root=seed('explicit_pec',row['pair_id'],role,event) if role=='fit' else endpoint_seed(row['mode'],row['pair_id'],endpoint,role,event)
    stream=role if role=='fit' else f'{endpoint}_{role}'
    if row.get('mode')=='pec_retrospective' and endpoint=='margin':
        root=seed('explicit_pec',row['pair_id'],role,event);stream=role
    eid=ledger.begin(role,len(x),n,event=event,endpoint=endpoint,theta_sha256=array_hash(theta),stream_root_seed=root,stream=stream,**metadata)
    start=time.perf_counter();scores,executor=predict(theta,jnp.asarray(x),root,stream,n);jax.block_until_ready(scores)
    assert executor['raw_physical_shots']==len(x)*n
    ledger.finish(eid,executor_ledger=executor,kernel_seconds=time.perf_counter()-start)
    ledger.oracle('exact_noisy_auxiliary_means_for_sampling',executor['simulator_propagations'],acquisition_id=eid,padded_propagations=executor['padded_simulator_propagations'])
    if not np.all(np.isfinite(scores)):raise FloatingPointError('Nonfinite PEC score')
    return scores,eid

@jax.jit
def metrics(theta,x,y,scores):
    mean=ideal_means(theta,x)
    ramp=lambda s:jnp.mean(jnp.clip(1-y*s/.1,0,1))
    return dict(sampled_ramp=ramp(scores),exact_ideal_mean_ramp=ramp(mean),correct=jnp.sum(jnp.where(scores>=0,1,-1)==y),ties=jnp.sum(scores==0),finite=jnp.all(jnp.isfinite(mean)))

def evaluate(theta,x,y,row,role,u,ledger,endpoint,phase="epoch"):
    n=row['N_monitor'] if role=='monitor' else row['N_confirm'];ledger.ensure(len(x)*n)
    chunks=[];ids=[]
    assert len(x)%384==0
    for c in range(len(x)//384):
        xx=x[384*c:384*(c+1)];yy=y[384*c:384*(c+1)]
        scores,eid=acquisition(theta,xx,row,role,(f'u{u}:chunk{c}' if phase=='epoch' else f'{phase}:u{u}:chunk{c}'),ledger,endpoint=endpoint,input_start=c*384,input_stop=(c+1)*384)
        v=host(metrics(theta,jnp.asarray(xx),jnp.asarray(yy),scores));assert v['finite']
        ledger.oracle('exact_ideal_mean_reporting',384,acquisition_id=eid,affects_stopping=False)
        ledger.reuse(eid,'sampled endpoint metrics and reporting');chunks.append(v);ids.append(eid)
    loss=float(np.mean([c['sampled_ramp'] for c in chunks]));correct=int(sum(c['correct'] for c in chunks))
    passes=loss<=row['epsilon_mar'] if endpoint=='margin' else correct>=row['minimum_correct']
    return dict(endpoint=endpoint,passes=bool(passes),sampled_ramp=loss,correct=correct,accuracy=correct/len(x),
        exact_mean_ramp=float(np.mean([c['exact_ideal_mean_ramp'] for c in chunks])),executed_score_ties=sum(c['ties'] for c in chunks),
        target=row['epsilon_mar'] if endpoint=='margin' else row['minimum_correct'],acquisition_ids=ids)

