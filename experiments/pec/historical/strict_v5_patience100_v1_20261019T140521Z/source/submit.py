"""Only the replacements/reviews authorized by the user's patience correction."""
from common import *
from launch_preflight import preflight
import subprocess

def submit(label,command,script):
    target=OUT/'manifest'/f'submission_{label}.json';intent=OUT/'manifest'/f'submission_intent_{label}.json'
    if target.exists() or intent.exists():raise RuntimeError('Already attempted; inspect receipt, no automatic retry: '+label)
    checked=preflight()
    new_json(intent,dict(utc=now(),label=label,command=command,script_sha256=digest(script),preflight=checked,user_authorized=True))
    r=subprocess.run(command,capture_output=True,text=True)
    receipt=dict(utc=now(),label=label,command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,script_sha256=digest(script),protocol_sha256=digest(OUT/'manifest/protocol.json'),frozen_sha256=digest(OUT/'manifest/frozen_execution.json'))
    if r.returncode==0:receipt['job_id']=r.stdout.strip().split(';')[0]
    new_json(target,receipt);print(json.dumps(receipt),flush=True)
    if r.returncode:raise RuntimeError('Submission failed; no retry: '+label)
    return receipt['job_id']

if __name__=='__main__':
    preflight();guards=[];jobs=[]
    for i in range(2):
        script=OUT/'launch_guard.sbatch';guards.append(submit(f'guard{i}',['sbatch','--parsable',str(script),str(i)],script))
    for arm in ['pec120','reference25','review120','pec180']:
        script=OUT/f'launch_{arm}.sbatch';command=['sbatch','--parsable']
        if arm=='pec180':command+=['--dependency=afterany:5254249','--kill-on-invalid-dep=yes']
        jobs.append(submit(arm,command+[str(script)],script))
    for i in range(2):
        script=OUT/'launch_review180.sbatch'
        jobs.append(submit(f'review180_{i}',['sbatch','--parsable',f'--array={28+i}',f'--dependency=afterany:{guards[i]}:5254249_{i}','--kill-on-invalid-dep=yes',str(script)],script))

    script=OUT/'launch_report.sbatch'
    submit('report',['sbatch','--parsable','--dependency=afterany:'+':'.join(jobs),'--kill-on-invalid-dep=yes',str(script)],script)
