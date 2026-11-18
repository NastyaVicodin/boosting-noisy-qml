from common import *
from reference_kernel import array_hash,host
import numpy as np
import jax

class BudgetReached(RuntimeError):pass

class Ledger:
    def __init__(self,directory,budget):
        self.directory=Path(directory);self.budget=budget;self.events={};self.done=set();self.counter=0;self.inherited_algorithm_raw_shots=0;self.inherited_online_raw_shots=0
    def totals(self):
        return {r:sum(v['raw_shots'] for k,v in self.events.items() if k in self.done and v['role']==r) for r in Acquisitions.ROLES}
    def online(self):return self.inherited_online_raw_shots+sum(v for k,v in self.totals().items() if k!='diagnostic')
    def ensure(self,n):
        if self.online()+n>self.budget:raise BudgetReached(f'Online cap {self.budget}: acquired {self.online()}, next acquisition group {n}')
    def begin(self,role,n,shots,**kw):
        if role!='diagnostic':self.ensure(n*shots)
        self.counter+=1;eid=f'{role}:{self.counter:07d}'
        event=dict(event_id=eid,role=role,n_inputs=int(n),shots_per_input=int(shots),raw_shots=int(n*shots),**kw)
        self.events[eid]=event;append_json(self.directory/'acquisitions.jsonl',dict(kind='acquisition_intent',**event));return eid
    def finish(self,eid,**kw):
        assert eid in self.events and eid not in self.done
        self.done.add(eid);append_json(self.directory/'acquisitions.jsonl',dict(kind='completed',event_id=eid,**kw))
    def reuse(self,eid,consumer):
        assert eid in self.done
        append_json(self.directory/'acquisitions.jsonl',dict(kind='classical_reuse',event_id=eid,consumer=consumer,raw_shots=0))
    def oracle(self,kind,n,**kw):append_json(self.directory/'oracle_access.jsonl',dict(kind=kind,n_inputs=int(n),raw_shots=0,**kw))
    def receipt(self):return dict(shot_totals=self.totals(),online_shots=self.online(),inherited_algorithm_raw_shots=self.inherited_algorithm_raw_shots,inherited_online_raw_shots=self.inherited_online_raw_shots,unresolved_acquisition_intents=[self.events[k] for k in self.events if k not in self.done])

def load_split(row,role):
    assert role in ['train_split','validation_split']
    p=Path(row[role]).resolve();assert '/test/' not in str(p) and 'test' not in p.name.lower()
    assert digest(p)==row[role+'_sha256']
    with np.load(p,allow_pickle=False) as z:return np.asarray(z['X'],np.float32),np.asarray(z['y'],np.float32)

def scalar_record(value):
    """Omit all prediction/parameter arrays from JSON summaries; hash array identities."""
    from dataclasses import is_dataclass,asdict
    if is_dataclass(value):value=asdict(value)
    if isinstance(value,dict):return {str(k):scalar_record(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [scalar_record(v) for v in value]
    if isinstance(value,(np.ndarray,jax.Array)):
        if np.ndim(value)==0:return scalar_record(np.asarray(value).item())
        return dict(omitted_array=True,shape=list(value.shape),array_sha256=array_hash(value))
    if isinstance(value,np.generic):return scalar_record(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value
