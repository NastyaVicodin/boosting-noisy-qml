from audit import *
import argparse,platform,socket,time,traceback
def run(index,protocol_hash):
    assert protocol_hash==digest(OUT/'manifest/protocol.json');check_frozen()
    row=jsonl(OUT/'manifest/runs.jsonl')[index];assert row['index']==index
    directory=OUT/'runs'/row['run_id'];directory.mkdir();ledger=Ledger(directory,row['online_budget_raw_shots']);start=time.perf_counter()
    try:
        new_json(directory/'start.json',dict(utc=now(),run=row,protocol_sha256=protocol_hash,hostname=socket.gethostname(),python=platform.python_version(),test_access=False))
        if row['kind']=='reference_review':
            from review_runner import run_reference_review
            result=run_reference_review(row,directory,ledger)
        elif row['kind']=='pec_review_continue':
            from review_runner import run_review
            result=run_review(row,directory,ledger)
        else:
            assert digest(row['initial_path'])==row['initial_file_sha256'] and digest(row['orders_path'])==row['orders_sha256']
            initial=np.load(row['initial_path'],allow_pickle=False);orders=np.load(row['orders_path'],allow_pickle=False)
            assert array_hash(initial)==row['initial_array_sha256'] and array_hash(orders)==row['orders_array_sha256']
            from train_runner import run_train
            result=run_train(row,directory,ledger,initial,orders)
        new_json(directory/'result.json',dict(run_id=row['run_id'],utc=now(),wall_seconds=time.perf_counter()-start,test_access=False,simulator_assisted=True,**result))
    except BaseException as exc:
        new_json(directory/'error.json',dict(status='ERROR',utc=now(),run_id=row['run_id'],error=repr(exc),traceback=traceback.format_exc(),no_automatic_retry=True,**ledger.receipt()));raise
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--index',type=int,choices=range(30),required=True);parser.add_argument('--protocol-sha256',required=True)
    args=parser.parse_args();run(args.index,args.protocol_sha256)
