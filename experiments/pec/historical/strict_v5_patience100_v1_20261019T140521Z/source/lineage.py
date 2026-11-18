"""Checkpoint identity, optimizer restoration and completed acquisition accounting."""
from audit import *
from reference_kernel import optimizer
import jax.numpy as jnp
import re

def complete_rows(path):
    p=Path(path)
    return [json.loads(s) for s in p.read_bytes().splitlines(keepends=True) if s.endswith(b'\n')] if p.exists() else []

def event_update(e):
    match=re.search(r'(?:^|:)u(\d+)',str(e.get('event',e['event_id'])))
    if match:return int(match.group(1))
    if isinstance(e.get('stream_event'),int):return e['stream_event']
    raise ValueError('Unrecognized acquisition update: '+str(e))

def source_ledger(path,n,cut_update=None):
    events=complete_rows(path);intent={};done=set();fit={}
    actual={r:0 for r in Acquisitions.ROLES};prefix_roles=dict(actual)
    for e in events:
        if e['kind']=='acquisition_intent':
            assert e['event_id'] not in intent;intent[e['event_id']]=e
        elif e['kind']=='completed':
            assert e['event_id'] in intent and e['event_id'] not in done;done.add(e['event_id'])
    for eid in done:
        e=intent[eid];actual[e['role']]+=e['raw_shots']
        if e['role']=='diagnostic':continue
        u=event_update(e)
        if cut_update is None or u<=cut_update:prefix_roles[e['role']]+=e['raw_shots']
        if e['role']=='fit':
            assert e['raw_shots']==64*n and u not in fit;fit[u]=e['raw_shots']
    prefix={};total=0
    for expect,u in enumerate(sorted(fit),1):
        assert expect==u;total+=fit[u];prefix[u]=total
    return dict(prefix=prefix,actual=actual,prefix_roles=prefix_roles,unresolved=[e for k,e in intent.items() if k not in done])

def load_theta(source,epoch,updates):
    p=Path(source)/f'state_u{24*epoch:05d}.npz';v=updates[24*epoch-1]
    assert v['update']==24*epoch
    with np.load(p,allow_pickle=False) as z:theta=np.asarray(z['theta'],np.float32)
    assert array_hash(theta)==v.get('theta_sha256',v.get('post_theta_sha256'))
    return jnp.asarray(theta),dict(checkpoint=str(p),checkpoint_sha256=digest(p),theta_array_sha256=array_hash(theta),source_model_epoch=epoch)

def save_state(path,theta,state):
    with Path(path).open('xb') as f:
        np.savez(f,theta=np.asarray(theta),**{f'opt_{i}':np.asarray(v) for i,v in enumerate(jax.tree_util.tree_leaves(state))})

def load_state(path,expected_update):
    with np.load(path,allow_pickle=False) as z:
        theta=jnp.asarray(z['theta']);template=optimizer().init(theta)
        leaves,tree=jax.tree_util.tree_flatten(template)
        assert set(z.files)=={'theta'}|{f'opt_{i}' for i in range(len(leaves))},'Missing full optimizer checkpoint'
        restored=[];counts=[]
        for i,v in enumerate(leaves):
            a=z[f'opt_{i}'];assert a.shape==v.shape and a.dtype==v.dtype
            restored.append(jnp.asarray(a))
            if a.shape==() and np.issubdtype(a.dtype,np.integer):counts.append(int(a))
        assert counts and all(c==expected_update for c in counts),counts
    return theta,jax.tree_util.tree_unflatten(tree,restored)

def checked_orders(row):
    assert digest(row['orders_path'])==row['orders_sha256']
    orders=np.load(row['orders_path'],allow_pickle=False)
    assert array_hash(orders)==row['orders_array_sha256']
    return orders

def verify_reference_overlap(updates):
    a,b=updates['margin'],updates['classification'];overlap=min(len(a),len(b))
    for i,(x,y) in enumerate(zip(a,b),1):
        assert x['update']==y['update']==i
        assert x['post_theta_sha256']==y['post_theta_sha256'] and x['loss']==y['loss']
    return overlap
