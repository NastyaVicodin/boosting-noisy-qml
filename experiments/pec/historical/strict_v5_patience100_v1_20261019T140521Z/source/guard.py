"""Apply patience to already running immutable PEC180 tasks; no training/sampling."""
from pathlib import Path
import argparse,datetime,hashlib,json,subprocess,time
from patience import scan_updates
D=Path(__file__).resolve().parents[1]
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def records(path):
    return [json.loads(s) for s in path.read_bytes().splitlines(keepends=True) if s.endswith(b'\n')] if path.exists() else []
def write(path,obj):
    with path.open('x') as f:json.dump(obj,f,indent=2)
def append(path,obj):
    with path.open('a') as f:f.write(json.dumps(obj)+'\n')

def stopping_epoch_ready(source,cut):
    cp=source/f'state_u{cut*24:05d}.npz'
    checks=records(source/'endpoint_checks.jsonl')
    done={v['endpoint'] for v in checks if v.get('confirmed') and v['position']['epoch']<=cut}
    checked={v['endpoint'] for v in checks if v['position']['epoch']==cut and 'confirmed' in v}
    return cp.exists() and (done|checked)=={'margin','classification'}

def guard(pair_index):
    row=json.loads((D/'manifest/guards.json').read_text())[pair_index]
    source=Path(row['source_run']);out=D/'guards'/row['name'];out.mkdir(parents=True)
    frozen=json.loads((D/'manifest/frozen_execution.json').read_text())
    for f in frozen['files']:assert hashlib.sha256(Path(f['path']).read_bytes()).hexdigest()==f['sha256']
    write(out/'start.json',dict(utc=now(),row=row,user_authorized=True))
    last_epoch=0
    while True:
        history=scan_updates(records(source/'updates.jsonl'))
        decision=history[-1] if history else None
        if decision and decision['epoch']!=last_epoch:
            append(out/'patience.jsonl',dict(utc=now(),**decision));last_epoch=decision['epoch']
        if decision and decision['should_stop']:
            cut=decision['epoch'];cut_update=cut*24;cp=source/f'state_u{cut_update:05d}.npz'
            # Preserve the existing final-epoch check opportunity before cancellation.
            if stopping_epoch_ready(source,cut):
                command=['scancel',row['source_slurm_task']]
                write(out/'cancellation_intent.json',dict(utc=now(),command=command,decision=decision,checkpoint=str(cp),checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest()))
                result=subprocess.run(command,capture_output=True,text=True)
                write(out/'result.json',dict(status='PATIENCE_STOP_REQUESTED',utc=now(),decision=decision,command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,
                    note='Review reconciles actual completed shots and any polling/batch overshoot separately from cutoff cost.'))
                if result.returncode:raise RuntimeError('Cancellation failed; inspect receipt')
                return
        # Detect a natural target/cap/budget/error end without keeping a guard alive.
        q=subprocess.run(['squeue','-h','-j',row['source_slurm_task'],'-o','%T'],capture_output=True,text=True,check=True)
        states=[s.strip() for s in q.stdout.splitlines() if s.strip()]
        if not states:
            write(out/'result.json',dict(status='SOURCE_ALREADY_TERMINAL',utc=now(),decision=decision,source_result_present=(source/'result.json').exists(),source_error_present=(source/'error.json').exists()))
            return
        time.sleep(15)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--index',type=int,choices=[0,1],required=True);guard(parser.parse_args().index)
