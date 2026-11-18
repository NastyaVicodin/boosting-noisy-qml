"""Submit exactly the user-approved ten PEC210 trajectories and deadline report."""
from common import *
from launch_preflight import preflight
import subprocess

def submit(label,args):
    script=OUT/f'launch_{"report" if label=="deadline_report" else "pec210"}.sbatch'
    target=OUT/'manifest'/f'submission_{label}.json';intent=OUT/'manifest'/f'submission_intent_{label}.json'
    if target.exists() or intent.exists():raise RuntimeError('Already attempted; inspect scheduler/receipt, no automatic resubmission')
    checked=preflight();command=['sbatch','--parsable',*args,str(script)]
    new_json(intent,dict(utc=now(),command=command,script_sha256=digest(script),preflight=checked,user_authorized=True))
    result=subprocess.run(command,capture_output=True,text=True,env=dict(os.environ,TZ='Europe/Amsterdam'))
    receipt=dict(utc=now(),command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,protocol_sha256=digest(OUT/'manifest/protocol.json'),frozen_sha256=digest(OUT/'manifest/frozen_execution.json'))
    if result.returncode==0:receipt['job_id']=result.stdout.strip().split(';')[0]
    new_json(target,receipt);print(json.dumps(receipt),flush=True)
    if result.returncode:raise RuntimeError('Submission failed; no automatic retry')
    return receipt['job_id']

if __name__=='__main__':
    submit('pec210',[])
    submit('deadline_report',['--begin=2026-10-20T14:00:00'])
