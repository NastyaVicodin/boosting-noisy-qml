"""Shared append-only IO and explicit acquisition accounting for screen v1."""
from pathlib import Path
import csv, datetime, hashlib, json, os, sys

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[2]
FROZEN = OUT/'frozen_src'
NS = 'strict_v5.pec210_depth_noise_explore.v1.20261022'
sys.path.insert(0, str(FROZEN))

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps([NS,*parts],separators=(',',':')).encode()).digest()[:8],'big') % (2**31-1) or 1
def new_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def new_csv(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def jsonl(path): return [json.loads(s) for s in Path(path).read_text().splitlines() if s]
def load_json(path): return json.loads(Path(path).read_text())
def source_record(path):
    p=Path(path);s=p.stat()
    return dict(path=str(p.resolve()),sha256=digest(p),bytes=s.st_size,mtime_utc=datetime.datetime.fromtimestamp(s.st_mtime,datetime.timezone.utc).isoformat())
def append_json(path, value):
    with Path(path).open('a') as f:f.write(json.dumps(value,sort_keys=True,allow_nan=False)+'\n');f.flush()

class Acquisitions:
    """A repeated acquisition ID is an error; reuse points to its existing event."""
    ROLES={'fit','edge','monitor','confirm','diagnostic','gradient'}
    def __init__(self, path=None): self.events={};self.path=path;self.reuses=[]
    def acquire(self, event_id, role, *, n_inputs, shots, learners=1, **metadata):
        if event_id in self.events:raise ValueError('Duplicate acquisition: '+event_id)
        if role not in self.ROLES or min(n_inputs,shots,learners)<1:raise ValueError('Invalid acquisition')
        event=dict(event_id=event_id,role=role,n_inputs=int(n_inputs),shots_per_input=int(shots),learners=int(learners),raw_shots=int(n_inputs)*int(shots)*int(learners),**metadata)
        self.events[event_id]=event
        if self.path:append_json(self.path,dict(kind='acquisition_intent',**event))
        return event
    def reuse(self,event_id,consumer):
        if event_id not in self.events:raise KeyError(event_id)
        e=dict(kind='classical_reuse',event_id=event_id,consumer=consumer,raw_shots=0);self.reuses.append(e)
        if self.path:append_json(self.path,e)
    def totals(self):
        return {role:sum(e['raw_shots'] for e in self.events.values() if e['role']==role) for role in sorted(self.ROLES)}

def prefix_cost(rounds, selected):
    chosen=rounds[:selected]
    if len(chosen)!=selected or [r['round'] for r in chosen]!=list(range(1,selected+1)):raise ValueError('Incomplete prefix')
    return {role:sum(r.get(role,0) for r in chosen) for role in Acquisitions.ROLES}

def check_frozen():
    m=load_json(OUT/'manifest/frozen_execution.json')
    for r in m['files']:
        if digest(r['path'])!=r['sha256']:raise RuntimeError('Frozen file mismatch: '+r['path'])
    return m
