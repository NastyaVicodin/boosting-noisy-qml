"""Build the user-approved ten-run PEC210 depth/noise pilot; no measurements."""
from common import *

BASE = ROOT/'artifacts/strict_v5_lowshot_depth_noise_3x3_dual_v1_20261020T1030Z'
REF_FIX = ROOT/'artifacts/strict_v5_lowshot_depth_noise_3x3_ref_fix_v2_20261020T104700Z'
P01_P02 = ROOT/'artifacts/strict_v5_fixed_noise_p01_p02_dual_v1_20261020T132536Z'
PILOT = ROOT/'artifacts/strict_v5_lower_depth_noise_pec_pilot_v1_20261021T092811Z'
SNAPSHOT = ROOT/'artifacts/strict_v5_existing_record_stop_cost_reporting_correction_20261022T195235Z/teacher_gate_snapshot.csv'
SELECTION = [(1,1,.02),(2,2,.02),(3,3,.005),(4,0,.01),(5,1,.005)]

base_rows = jsonl(BASE/'manifest/runs.jsonl')
fixed_rows = jsonl(P01_P02/'manifest/runs.jsonl')
ref_fix_rows = jsonl(REF_FIX/'manifest/runs.jsonl')
rows=[];lineage=[]
for depth,teacher,p in SELECTION:
    for repetition in (0,1):
        pair=f'd{depth}_t{teacher:02d}_r{repetition:02d}'
        refs=[r for r in base_rows if r['method']=='reference' and r['pair_id']==pair]
        corrected=[r for r in ref_fix_rows if r['method']=='reference' and r['pair_id']==pair]
        assert len(refs)==len(corrected)==1
        ref,ref_fix=refs[0],corrected[0]
        assert ref['depth']==depth and ref['teacher']==teacher and ref['repetition']==repetition
        for key in ['teacher_path','teacher_sha256','train_split','train_split_sha256',
                    'validation_split','validation_split_sha256','initial_path','initial_file_sha256',
                    'initial_array_sha256','orders_path','orders_sha256','orders_array_sha256',
                    'epsilon_mar','minimum_correct','a_star','margin_target_id','margin_target_registry',
                    'margin_target_registry_sha256','classification_target_id','classification_target_registry',
                    'classification_target_registry_sha256']:
            assert ref[key]==ref_fix[key],(pair,key)
        source_rows=base_rows if p==.005 else fixed_rows
        boosting=[]
        for method in ['adaboost_r','real_adaboost']:
            matches=[r for r in source_rows if r['method']==method and r['pair_id']==pair
                     and r['noise_strength_1q']==r['noise_strength_2q']==p]
            assert len(matches)==1,(pair,p,method)
            b=matches[0]
            for key in ['teacher_sha256','train_split_sha256','validation_split_sha256',
                        'initial_array_sha256','orders_array_sha256','epsilon_mar','minimum_correct',
                        'margin_target_registry_sha256','classification_target_registry_sha256']:
                assert b[key]==ref[key],(pair,p,method,key)
            boosting.append(dict(method=method,run_id=b['run_id'],manifest=str(BASE/'manifest/runs.jsonl' if p==.005 else P01_P02/'manifest/runs.jsonl')))
        row=dict(ref)
        row.update(index=len(rows),run_id=f'{pair}_p{str(p).replace(".","p")}_pec210_explore',
                   kind='train',mode='pec_dual',method='explicit_sampled_pec',
                   N_fit=210,N_monitor=210,N_confirm=210,N_diagnostic=210,N_edge=0,N_gradient=0,
                   noise_strength_1q=p,noise_strength_2q=p,mu=.1,batch_size=64,
                   monitor_every_updates=24,max_epochs=500,max_updates=12000,
                   patience_epochs=100,patience_min_delta=0.,restore_best=True,
                   gradient_convention='exact_ideal_mean_vjp_at_signed_PEC_score',
                   online_budget_raw_shots=549144980,primary_endpoint='margin_and_classification',
                   target_id=ref['margin_target_id'],target_registry=ref['classification_target_registry'],
                   reference_run_id=ref_fix['run_id'],
                   stream_definition='New strict_v5.pec210_depth_noise_explore.v1.20261022 namespace; event-root seeds indexed by pair_id and role; independent of prior PEC60/100 draws')
        assert row['minimum_correct']=={(1,1):380,(2,2):374,(3,3):319,(4,0):320,(5,1):299}[(depth,teacher)]
        rows.append(row)
        lineage.append(dict(pair_id=pair,noise=p,reference_run_id=ref_fix['run_id'],
                            reference_manifest=str(REF_FIX/'manifest/runs.jsonl'),boosting=boosting))
assert len(rows)==10 and len({r['run_id'] for r in rows})==10
with (OUT/'manifest/runs.jsonl').open('x') as stream:
    for row in rows:stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
new_json(OUT/'manifest/baseline_lineage.json',dict(utc=now(),selection=SELECTION,
    base_manifest=str(BASE/'manifest/runs.jsonl'),base_manifest_sha256=digest(BASE/'manifest/runs.jsonl'),
    corrected_reference_manifest=str(REF_FIX/'manifest/runs.jsonl'),corrected_reference_manifest_sha256=digest(REF_FIX/'manifest/runs.jsonl'),
    p01_p02_manifest=str(P01_P02/'manifest/runs.jsonl'),p01_p02_manifest_sha256=digest(P01_P02/'manifest/runs.jsonl'),
    corrected_teacher_snapshot=str(SNAPSHOT),corrected_teacher_snapshot_sha256=digest(SNAPSHOT),pairs=lineage))
new_json(OUT/'manifest/protocol.json',dict(schema='strict_v5.pec210_depth_noise_explore.v1',utc=now(),
    authorization='User approved the proposed five teacher/noise pairs, two repetitions each, 2026-10-22.',
    user_authorized=True,run_count=10,selected_cells=[dict(depth=d,teacher=t,noise=p,repetitions=[0,1]) for d,t,p in SELECTION],
    selection_policy='Outcome-informed exploratory subset from the frozen 25-shot grid; not a representative cell-gate or success-rate estimate.',
    shot_allocations=dict(fit=210,monitor=210,confirm=210,diagnostic=210,edge=0,gradient=0),
    target_policy='Same frozen sampled training-margin and matched validation-classification targets; independent confirmations on one trajectory; stop when both confirmed, patience100, 500 epochs or existing online cap.',
    gradients='Sampled explicit PEC forward scores with exact ideal-mean VJP; simulator assistance recorded, no measured gradient shots.',
    max_epochs=500,max_updates=12000,online_budget_raw_shots=549144980,algorithm_budget_30M_applied=False,
    patience=dict(epochs=100,min_delta=0.,restore_best=True,metric='Mean sampled fitting loss over each completed epoch, checked every epoch, no warmup'),
    resource_policy='Algorithm shots are fitting+once-only edge+measured gradients. Monitoring, confirmation and diagnostics tracked separately.',
    scheduler=dict(array='0-9%2',dependency='afterany:5306944',memory_gb=8,cpus=1,wall_hours=96,
                   walltime_is_queue_limit_not_scientific_stopping=True,max_total_training_concurrency=20),
    safety=dict(test_access=False,automatic_retry=False,fixed_parameter_PEC_validation_repeated=False,
                original_campaigns_immutable=True,unresolved_if_scheduler_walltime_reached=True),
    source_campaign=str(PILOT),baseline_manifest=str(BASE/'manifest/runs.jsonl')))

old=load_json(PILOT/'manifest/runtime.json')
runtime=dict(utc=now(),executable=old['executable'],versions=old['versions'],modules={})
for name in old['modules']:
    module_path=OUT/'frozen_src'/Path(*name.split('.')).with_suffix('.py')
    if name=='numpy_compat':module_path=OUT/'frozen_src/numpy_compat.py'
    assert module_path.exists(),module_path
    runtime['modules'][name]=source_record(module_path)
new_json(OUT/'manifest/runtime.json',runtime)
print(json.dumps(dict(status='BUILT',run_count=len(rows),protocol_sha256=digest(OUT/'manifest/protocol.json'))))
