"""Reference-only packaging correction; no optimizer or measurement work."""
from common import *
OLD=ROOT/'artifacts/strict_v5_lowshot_depth_noise_3x3_dual_v1_20261020T1030Z'
old=jsonl(OLD/'manifest/runs.jsonl')
rows=[]
for r in old:
    if r['method']!='reference':continue
    row=dict(r);row['index']=len(rows)
    row['source_frozen_protocol_sha256']=digest(OLD/'manifest/protocol.json')
    rows.append(row)
assert len(rows)==45
with (OUT/'manifest/runs.jsonl').open('x') as f:
    for r in rows:f.write(json.dumps(r,sort_keys=True)+'\n')
original=load_json(OLD/'manifest/protocol.json')
original.update(schema='strict_v5.lowshot_depth_noise_3x3.reference_fix.v2',
    prepared_at_utc=now(),status='PREPARED_NOT_LAUNCHED',
    run_count=45,reference_runs=45,boosting_runs=0,
    source_campaign=str(OLD),source_protocol_sha256=digest(OLD/'manifest/protocol.json'),
    correction='The v1 train_runner imported an uncopied PEC helper module before any reference fit. Remove unused PEC imports and run this reference-only package. No acquired measurements in failed v1 reference tasks; same paired inputs, seeds, targets, 25 shots and stopping rules.',
    v1_failed_reference_directory_immutable=True)
new_json(OUT/'manifest/protocol.json',original)
print(dict(reference_runs=len(rows),protocol_sha256=digest(OUT/'manifest/protocol.json')))
