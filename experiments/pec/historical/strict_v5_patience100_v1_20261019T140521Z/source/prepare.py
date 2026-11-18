from common import *
D=ROOT/'artifacts/strict_v5_patience20_correction_v1_20261019T110545Z'
C=ROOT/'artifacts/strict_v5_pec180_boost25_classification_v1_20261019T092910Z'
O=ROOT/'artifacts/strict_v5_lowshot_margin_methods_v1_20261016T185703Z'
MR=ROOT/'artifacts/strict_v5_reference_margin_n25_v1_20261016T172426Z'
CR=ROOT/'artifacts/strict_v5_reference_n25_v1_20261016T164710Z'
rows=jsonl(D/'manifest/runs.jsonl')
for r in rows:
    r['run_id']=r['run_id'].replace('patience20','patience100');r['patience_epochs']=100
    if r['index']<14:
        r['kind']='train'
        if r['index']<6:r['superseded_patience20_run']=str(D/'runs'/r['run_id'].replace('patience100','patience20'))
    elif r['index']<24:
        r['kind']='reference_review';r['stream_definition']='Reuse original matched fitting and endpoint streams; fresh restored-best checks use reference_patience100 endpoint/role/phase streams.';pair=r['pair_id'];r['run_id']=pair+'_reference25_patience100_review'
        r['reference_sources']={ep:dict(campaign=str(root),run_id=pair+suffix,protocol_sha256=digest(root/'manifest/protocol.json'),frozen_sha256=digest(root/'manifest/frozen_execution.json')) for ep,root,suffix in [('margin',MR,'_n25_margin_historical_clip'),('classification',CR,'_n25_historical_clip')]}
    else:
        r['kind']='pec_review_continue';r['prior_classification_sources']=[]
        if r['N_fit']==120:
            for p in [C/'runs'/(r['pair_id']+'_pec120_retrospective_classification')/'endpoint_checks.jsonl',D/'runs'/(r['pair_id']+'_pec120_patience20_review')/'endpoint_checks.jsonl']:
                if p.exists():r['prior_classification_sources'].append(source_record(p))
            r['guard_result']=None
        else:r['guard_result']=str(OUT/'guards'/(r['pair_id']+'_pec180')/'result.json')
        r.pop('existing_classification_review',None)
with (OUT/'manifest/runs.jsonl').open('x') as f:
    for r in rows:f.write(json.dumps(r,sort_keys=True)+'\n')
new_json(OUT/'manifest/guards.json',load_json(D/'manifest/guards.json'))
new_json(OUT/'manifest/protocol.json',dict(schema='strict_v5.patience100.v1',utc=now(),user_authorized=True,run_count=30,
    patience=dict(epochs=100,min_delta=0,metric='Mean of24 sampled smooth-hinge fitting minibatch losses; no warmup; everyepoch',restore_best=True),max_epochs=500,
    algorithm_budget_30M_applied=False,online_budget_raw_shots=549144980,shot_allocations=[25,120,180],
    source_campaigns=dict(previous=str(D),pec180=str(C),original_pec120=str(O),reference_margin=str(MR),reference_classification=str(CR)),
    execution='14new PEC runs;10reference reviews reusing existing trajectories;4PEC120 prefix reviews with continuation from full optimizer checkpoints when patience100 has not fired;2guarded PEC180 reviews.2live PEC180 sources continue under new external100guards.',
    replacement_reason='The two active patience20 PEC120 executables hardcode20 and save theta only. They are superseded by new paired100runs; their actual spend is preserved separately. No pretending an Adam restart is exact continuation.',
    reference='Verify overlapping original reference fitting theta hashes/losses. Retain original first confirmations <=100cutoff. Recheck only unresolved endpoints at restored best. No reference retraining required.',
    accounting='Canonical policy path counts inherited full fitting prefix once plus new fitting. Actual historical expenditure, abandoned20runs, beyond-cutoff work and repeated partial-epoch acquisitions separate. Monitoring/confirmation/diagnostics excluded from algorithm shots but never erased. Inherited online cost included in cap.',
    continuation='OriginalPEC120 theta and Adam leaves resumed at last full epoch with original global update indices, fitting seeds and data orders. Newly repeated partial-epoch acquisitions charged as new work; prior partial spend separate.',
    target_policy='Frozen targets. Separate first confirmations; stop when both confirmed, patience100,500epochs or existing online cap. At patience/cap fresh final monitor/confirm-on-pass on restored best for unresolved targets.',
    sampling='Same per-input fit/monitor/confirm shots within each arm. Reuse existing original margin/classification checks. New final checks use patience100 phase seed. Sampled-score success only.',
    gradients='Unchanged exact ideal VJP for reference/PEC and historical noisy VJP boosting; simulator assistance disclosed.',
    scheduler=dict(pec120_new_concurrency=2,pec180_new_concurrency=2,historical_pec_review_or_continuation_concurrency=2,reference_review_concurrency=2,pec120_wall_hours=48,pec180_wall_hours=96,reference_wall_hours=4,pec_review_wall_hours=48,pec180_training_dependency='afterany:5254249'),
    safety=dict(test_access=False,original_frozen_files_edited=False,automatic_retry=False,fixed_parameter_PEC_validation_repeated=False)))
with (OUT/'STOPPING_POLICY.md').open('x') as f:f.write((ROOT/'AGENTS.md').read_text())
print(json.dumps(dict(rows=len(rows),protocol_sha256=digest(OUT/'manifest/protocol.json'))))
