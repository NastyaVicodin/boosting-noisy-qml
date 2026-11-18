"""Freeze ten missing PEC60/PEC100 trajectories; no circuits or draws."""
from __future__ import annotations

from common import *
import copy
import importlib.metadata
import math

PILOT = ROOT / 'artifacts/strict_v5_lower_depth_noise_pec_pilot_v1_20261021T092811Z'
BASE = ROOT / 'artifacts/strict_v5_lowshot_depth_noise_3x3_dual_v1_20261020T1030Z'
PLAN = ROOT / 'artifacts/strict_v5_pec_2v3_runtime_estimate_20261029T132235Z/PROPOSED_NEW_SLOTS.csv'
SOURCE_NAMES = ('common.py','audit.py','endpoints.py','lineage.py','patience.py',
                'reference_kernel.py','train_runner.py','pec_measurements.py',
                'pec_kernel.py','environment.sh','runner.py')


def build():
    assert not (OUT/'manifest/runs.jsonl').exists()
    assert not list((OUT/'runs').iterdir())
    planned = [p for p in csv.DictReader(PLAN.open(newline='')) if
               (int(p['depth']),float(p['noise']),int(p['pec_shots']))
               in {(3,.005,60),(2,.01,100)}]
    assert len(planned)==10
    planned.sort(key=lambda p:(int(p['pec_shots']),int(p['paper_teacher']),int(p['repetition'])))
    old = jsonl(PILOT/'manifest/runs.jsonl')
    refs = {(r['depth'],r['teacher'],r['repetition']):r for r in
            jsonl(BASE/'manifest/runs.jsonl') if r['method']=='reference'}
    templates = {(3,60):next(r for r in old if r['depth']==3),
                 (2,100):next(r for r in old if r['depth']==2)}
    rows=[]
    for index,p in enumerate(planned):
        d,n,t,rep = (int(p['depth']),int(p['pec_shots']),
                     int(p['source_teacher']),int(p['repetition']))
        paper=int(p['paper_teacher']); noise=float(p['noise'])
        assert t==(2,3,4)[paper]
        ref=refs[d,t,rep]; template=templates[d,n]
        assert ref['pair_id']==p['comparison_pair_id']
        assert ref['train_split_sha256']==p['train_split_sha256']
        assert ref['validation_split_sha256']==p['validation_split_sha256']
        assert ref['initial_array_sha256']==p['initial_array_sha256']
        assert ref['orders_array_sha256']==p['orders_array_sha256']
        assert ref['epsilon_mar']==float(p['margin_target'])
        assert ref['minimum_correct']==int(p['classification_minimum_correct'])
        assert ref['initial_path']==p['initial_path']
        assert ref['orders_path']==p['orders_path']
        assert ref['margin_target_registry_sha256']==template['margin_target_registry_sha256']
        assert ref['classification_target_registry_sha256']==template['classification_target_registry_sha256']
        assert ref['margin_target_registry']==template['margin_target_registry']
        assert ref['classification_target_registry']==template['classification_target_registry']
        row=copy.deepcopy(ref)
        for k in ('kind','mode','method','N_diagnostic','N_edge','N_gradient','mu',
                  'batch_size','monitor_every_updates','max_epochs','max_updates',
                  'patience_epochs','patience_min_delta','restore_best',
                  'gradient_convention','online_budget_raw_shots','primary_endpoint'):
            row[k]=template[k]
        row.update(index=index,kind='train',method='explicit_sampled_pec',
                   mode='pec_dual',run_id=f'd{d}_t{t:02d}_r{rep:02d}_pec{n}_patience100_dual_expand',
                   pair_id=p['proposed_pec_pair_id'],comparison_pair_id=ref['pair_id'],
                   reference_run_id=ref['run_id'],original_pec_template_run_id=template['run_id'],
                   paper_teacher=paper,noise_strength_1q=noise,noise_strength_2q=noise,
                   N_fit=n,N_monitor=n,N_confirm=n,N_diagnostic=n,
                   target_id=ref['margin_target_id'],
                   target_registry=ref['classification_target_registry'],
                   stream_definition='Frozen pilot namespace plus unique allocation-specific PEC pair_id; roots indexed by role and event')
        assert row['N_edge']==row['N_gradient']==0
        assert row['patience_epochs']==100 and row['patience_min_delta']==0
        assert row['max_epochs']==500 and row['max_updates']==12000
        assert row['online_budget_raw_shots']==549144980
        assert row['minimum_correct']==math.ceil(row['a_star']*384)
        for k,hashkey in (('train_split','train_split_sha256'),
                          ('validation_split','validation_split_sha256'),
                          ('teacher_path','teacher_sha256'),
                          ('initial_path','initial_file_sha256'),
                          ('orders_path','orders_sha256')):
            assert '/test/' not in row[k] and digest(row[k])==row[hashkey]
        rows.append(row)
    assert len(rows)==len({r['run_id'] for r in rows})==len({r['pair_id'] for r in rows})==10
    assert {(r['depth'],r['N_fit']) for r in rows}=={(3,60),(2,100)}
    for d,n in ((3,60),(2,100)):
        assert len([r for r in rows if r['depth']==d and r['N_fit']==n])==5
    for d in (2,3):
        for t in (2,3,4):
            rr=[refs[d,t,k] for k in (0,1,2)]
            assert len({r['initial_array_sha256'] for r in rr})==3
            assert len({r['orders_array_sha256'] for r in rr})==3
    with (OUT/'manifest/runs.jsonl').open('x') as f:
        for row in rows:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    old_runtime=load_json(PILOT/'manifest/runtime.json')
    modules={}
    for name,record in old_runtime['modules'].items():
        path=OUT/'frozen_src'/Path(*name.split('.')).with_suffix('.py')
        assert digest(path)==record['sha256']
        modules[name]=source_record(path)
    new_json(OUT/'manifest/runtime.json',dict(executable=old_runtime['executable'],
             versions=old_runtime['versions'],modules=modules,utc=now()))
    copied=[]
    for name in SOURCE_NAMES:
        origin=PILOT/'source'/name; target=OUT/'source'/name
        if name in ('common.py','runner.py'):
            assert name=='common.py' and target.read_text()==origin.read_text().replace('ROOT = OUT.parents[1]','ROOT = OUT.parents[2]') or name=='runner.py' and target.read_text()==origin.read_text().replace('choices=range(8)','choices=range(10)')
        else:
            assert digest(origin)==digest(target)
        copied.append(dict(source=str(origin),source_sha256=digest(origin),destination=str(target),destination_sha256=digest(target)))
    new_json(OUT/'manifest/lineage.json',dict(utc=now(),source_campaign=str(PILOT),
        source_protocol_sha256=digest(PILOT/'manifest/protocol.json'),
        source_frozen_sha256=digest(PILOT/'manifest/frozen_execution.json'),
        reference_campaign=str(BASE),reference_manifest_sha256=digest(BASE/'manifest/runs.jsonl'),
        proposed_slots=str(PLAN),proposed_slots_sha256=digest(PLAN),copied_files=copied,
        wrapper_delta='Path root adjusted for nested artifact; runner index range 8 to 10. Scientific kernel, optimizer, stopping and estimator unchanged.',
        historical_results_modified=False,new_measurement_samples=0))
    protocol=dict(schema='strict_v5.figure6_pilot10_extension.v1',utc=now(),
       user_authorized=True,authorization='User 2026-10-30 authorized three PEC repetitions for each selected teacher in eight Figure 6 configurations.',
       run_count=10,selected_configurations=[{'depth':3,'noise':.005,'N_PEC':60},{'depth':2,'noise':.01,'N_PEC':100}],
       paper_source_teachers={'depth2':[2,3,4],'depth3':[2,3,4]},
       source_campaign=str(PILOT),reference_campaign=str(BASE),proposal_file=str(PLAN),proposal_sha256=digest(PLAN),
       patience={'epochs':100,'min_delta':0,'restore_best':True,'no_warmup':True,'metric':'Arithmetic mean of 24 sampled smooth-hinge minibatch losses after each completed epoch'},
       max_epochs=500,max_updates=12000,online_budget_raw_shots=549144980,
       target_policy='Frozen unrounded training-margin threshold and integer validation-correct cutoff; independent confirmation for each target; stop after both hits or existing scientific limit.',
       gradients='Simulator-supplied ideal-mean VJP at sampled PEC forward score; no measured gradient shots.',
       accounting='Raw algorithm fitting shots plus any once-only edge or measured-gradient shots; monitoring, confirmation and diagnostics separate; no gamma-squared remultiplication.',
       pairing='Use saved N25 reference initial/order arrays for each same-task repetition; unique allocation-specific PEC pair IDs separate new measurement streams; historical arrays unchanged.',
       scheduler={'cpu_per_task':1,'memory_gb_per_task':8,'wall_hours':96,'max_total_training_concurrency':20,'walltime_is_not_scientific_stop':True},
       safety={'test_access':False,'automatic_retry':False,'historical_files_edited':False,'pec120_stays_held':True})
    new_json(OUT/'manifest/protocol.json',protocol)
    paths={OUT/'manifest/runs.jsonl',OUT/'manifest/protocol.json',OUT/'manifest/runtime.json',
           OUT/'manifest/lineage.json',PLAN,PILOT/'manifest/protocol.json',
           PILOT/'manifest/frozen_execution.json',BASE/'manifest/runs.jsonl'}
    paths.update((OUT/'source').glob('*.py'))
    paths.update((OUT/'source').glob('*.sh'))
    paths.update((OUT/'frozen_src').rglob('*.py'))
    for row in rows:
        paths.update(Path(row[k]) for k in ('train_split','validation_split','teacher_path',
                     'initial_path','orders_path','margin_target_registry','classification_target_registry'))
    assert all('/test/' not in str(p) and 'test_prediction' not in p.name for p in paths)
    new_json(OUT/'manifest/frozen_execution.json',dict(utc=now(),
             protocol_sha256=digest(OUT/'manifest/protocol.json'),
             files=[source_record(p) for p in sorted(paths)],
             policy='Immutable after submission; only new receipts and acquired run records may be appended.'))
    print(json.dumps({'status':'FROZEN','run_count':10,'protocol_sha256':digest(OUT/'manifest/protocol.json'),
                      'frozen_sha256':digest(OUT/'manifest/frozen_execution.json')},sort_keys=True))


if __name__=='__main__':build()
