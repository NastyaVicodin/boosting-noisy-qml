"""Deterministic scheduler index maps; no job submission."""
from common import *
rows=jsonl(OUT/'manifest/runs.jsonl')
groups=dict(reference=[],core=[],mid=[],high=[])
for row in rows:
    i=row['index']
    if row['method']=='reference':groups['reference'].append(i)
    elif row['depth'] in [4,5] and row['noise_strength_1q']==.05:
        groups['high'].append(i)
    elif row['noise_strength_1q']==.005:groups['core'].append(i)
    else:groups['mid'].append(i)
assert {k:len(v) for k,v in groups.items()}==dict(reference=45,core=90,mid=72,high=36)
assert sorted(sum(groups.values(),[]))==list(range(243))
for group,indices in groups.items():
    with (OUT/'manifest'/f'{group}_indices.txt').open('x') as f:
        f.write(''.join(f'{i}\n' for i in indices))
new_json(OUT/'manifest/scheduler_groups.json',dict(utc=now(),
    groups={k:dict(count=len(v),indices_file=str(OUT/'manifest'/f'{k}_indices.txt'))
            for k,v in groups.items()},
    intended_concurrency=dict(reference=4,core=6,mid=4,high=4),
    mid_dependency='afterany entire reference array; ref4+core6+high4=14 initially;core6+mid4+high4=14 after reference',
    PEC_running_cap=6,total_training_cap=20))
print({k:len(v) for k,v in groups.items()})
