"""Build the approved pilot manifest, then freeze it after deterministic checks."""
from common import *
import argparse,importlib.metadata

PROPOSAL=ROOT/'artifacts/strict_v5_lower_noise_depth_pec_proposal_20261021T091501Z/proposal.json'
BASE=ROOT/'artifacts/strict_v5_lowshot_depth_noise_3x3_dual_v1_20261020T1030Z'
SOURCE=ROOT/'artifacts/strict_v5_patience100_v1_20261019T140521Z'

def build():
    proposal=load_json(PROPOSAL)
    assert proposal['status']=='PROPOSED_NOT_SUBMITTED' and proposal['job_count']==8
    base_rows=jsonl(BASE/'manifest/runs.jsonl')
    rows=[]
    for index,item in enumerate(proposal['jobs']):
        ref=next(r for r in base_rows if r['method']=='reference' and r['pair_id']==item['pair_id'])
        assert ref['depth']==item['depth'] and ref['teacher']==item['teacher'] and ref['repetition']==item['repetition']
        for key in ['teacher_sha256','train_split_sha256','validation_split_sha256','initial_array_sha256','orders_array_sha256','epsilon_mar','minimum_correct','margin_target_id','classification_target_id']:
            assert ref[key]==item[key],(key,item['pair_id'])
        row=dict(ref)
        row.update(index=index,run_id=item['run_id'],kind='train',mode='pec_dual',method='explicit_sampled_pec',
                   N_fit=item['N_fit'],N_monitor=item['N_monitor'],N_confirm=item['N_confirm'],
                   N_diagnostic=item['N_diagnostic'],N_edge=0,N_gradient=0,
                   noise_strength_1q=item['noise'],noise_strength_2q=item['noise'],
                   mu=.1,batch_size=64,monitor_every_updates=24,max_epochs=500,max_updates=12000,
                   patience_epochs=100,patience_min_delta=0.,restore_best=True,
                   gradient_convention='exact_ideal_mean_vjp_at_signed_PEC_score',
                   online_budget_raw_shots=549144980,primary_endpoint='margin_and_classification',
                   target_id=item['margin_target_id'],
                   target_registry=ref['classification_target_registry'],
                   stream_definition='source/common.py seed(namespace,pair,method/role/event); no shared draws with baselines')
        row.pop('patience_min_delta',None)
        row['patience_min_delta']=0.
        rows.append(row)
    with (OUT/'manifest/runs.jsonl').open('x') as f:
        for row in rows:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    protocol=dict(schema='strict_v5.lower_depth_noise_pec_pilot.v1',user_authorized=True,run_count=8,
                  target_policy='Frozen sampled training margin and matched validation classification; independent confirmations on one trajectory; stop when both confirmed, patience100, 500 epochs or online cap.',
                  gradients='Sampled explicit PEC forward scores with exact ideal-mean VJP; simulator assistance recorded, no measured gradient shots.',
                  max_epochs=500,max_concurrent_new_pec=4,online_budget_raw_shots=549144980,
                  patience=dict(epochs=100,min_delta=0.,restore_best=True,metric='Mean sampled fitting loss over each completed epoch, checked every epoch, no warmup'),
                  shot_allocations=dict(depth3_p005=60,depth2_p01=100,fit_monitor_confirm_equal=True),
                  resource_policy='Algorithm shots are fit+edge+measured gradient. Monitor, confirmation and diagnostics tracked separately.',
                  safety=dict(test_access=False,automatic_retry=False,fixed_parameter_PEC_validation_repeated=False),
                  source_campaign=str(SOURCE),proposal=str(PROPOSAL),utc=now())
    new_json(OUT/'manifest/protocol.json',protocol)
    new_json(OUT/'manifest/baseline_lineage.json',dict(utc=now(),proposal_sha256=digest(PROPOSAL),
             existing_results=[dict(pair_id=j['pair_id'],reference_path=j['reference_result_path'],reference_sha256=j['reference_result_sha256'],boosting=j['boosting_baselines']) for j in proposal['jobs']]))
    old=load_json(SOURCE/'manifest/runtime.json')
    runtime=dict(utc=now(),executable=old['executable'],versions=old['versions'],modules={})
    for name in old['modules']:
        module_path=OUT/'frozen_src'/Path(*name.split('.')).with_suffix('.py')
        if name=='numpy_compat':module_path=OUT/'frozen_src/numpy_compat.py'
        assert module_path.exists(),module_path
        runtime['modules'][name]=source_record(module_path)
    new_json(OUT/'manifest/runtime.json',runtime)
    print(json.dumps(dict(status='BUILT',run_count=len(rows),protocol_sha256=digest(OUT/'manifest/protocol.json'))))

def freeze():
    rows=jsonl(OUT/'manifest/runs.jsonl');assert len(rows)==8
    checks=load_json(OUT/'validation/deterministic_checks.json')
    assert checks['status']=='PASS' and checks['new_measurement_samples']==0
    paths={PROPOSAL,OUT/'manifest/runs.jsonl',OUT/'manifest/protocol.json',OUT/'manifest/runtime.json',
           OUT/'manifest/baseline_lineage.json',OUT/'validation/deterministic_checks.json'}
    paths.update((OUT/'source').glob('*'))
    paths.update((OUT/'frozen_src').rglob('*.py'))
    for row in rows:
        paths.update(Path(row[k]) for k in ['teacher_path','train_split','validation_split','initial_path','orders_path','margin_target_registry','classification_target_registry'])
    for item in load_json(OUT/'manifest/baseline_lineage.json')['existing_results']:
        paths.add(Path(item['reference_path']))
        paths.update(Path(b['result_path']) for b in item['boosting'])
    files=[source_record(p) for p in sorted(paths) if p.is_file()]
    assert len(files)==len(paths),(len(files),len(paths))
    new_json(OUT/'manifest/frozen_execution.json',dict(utc=now(),policy='immutable_source_and_inputs',protocol_sha256=digest(OUT/'manifest/protocol.json'),files=files))
    print(json.dumps(dict(status='FROZEN',files=len(files),frozen_sha256=digest(OUT/'manifest/frozen_execution.json'))))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['build','freeze']);args=parser.parse_args()
    (build if args.mode=='build' else freeze)()
