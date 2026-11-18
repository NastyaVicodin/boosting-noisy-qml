"""Prepare a paired PEC240 campaign from the immutable PEC210 manifest; no sampling."""
from common import *

BASE = ROOT/'artifacts/strict_v5_pec210_overnight_v1_20261019T211408Z'
old = jsonl(BASE/'manifest/runs.jsonl')
assert len(old) == 10
assert {(r['teacher'], r['repetition']) for r in old} == {(t, rep) for t in range(5) for rep in range(2)}
new = []
for i, original in enumerate(old):
    assert original['index'] == i and original['kind'] == 'train'
    assert original['depth'] == 3 and original['noise_strength_1q'] == original['noise_strength_2q'] == .01
    assert original['N_fit'] == original['N_monitor'] == original['N_confirm'] == original['N_diagnostic'] == 210
    assert original['patience_epochs'] == 100 and original['max_epochs'] == 500
    assert original['online_budget_raw_shots'] == 549144980
    row = dict(original)
    row['run_id'] = original['run_id'].replace('_pec210_', '_pec240_')
    assert row['run_id'] != original['run_id']
    for role in ['N_fit', 'N_monitor', 'N_confirm', 'N_diagnostic']:
        row[role] = 240
    assert {key for key in row if row[key] != original[key]} == {'run_id','N_fit','N_monitor','N_confirm','N_diagnostic'}
    new.append(row)
with (OUT/'manifest/runs.jsonl').open('x') as stream:
    for row in new: stream.write(json.dumps(row, sort_keys=True)+'\n')

new_json(OUT/'manifest/protocol.json', dict(
    schema='strict_v5.pec240_d3_p010.v1', utc=now(), user_authorized=True,
    authorization='User requested submission of depth-3 p=0.01 explicit PEC at 240 shots if absent, 2026-10-22.',
    parent_campaign=str(BASE), parent_manifest_sha256=digest(BASE/'manifest/runs.jsonl'),
    parent_protocol_sha256=digest(BASE/'manifest/protocol.json'),
    run_count=10, pairing='The PEC210 five teachers x two repetitions, identical initial parameters, minibatch orders, train/validation splits and frozen targets. Shot estimators at a different N are not claimed to share a complete stochastic stream.',
    shots=dict(fit=240,monitor=240,confirm=240,diagnostic=240,edge=0,gradient=0),
    max_epochs=500, max_updates=12000, online_budget_raw_shots=549144980,
    algorithm_budget_30M_applied=False,
    patience=dict(epochs=100,min_delta=0,restore_best=True,metric='Mean sampled smooth-hinge minibatch fitting loss, checked after every completed epoch without warmup'),
    stopping='First confirmation of both margin and classification, or fitting-loss patience100, 500 epochs, or existing online-shot budget. Check restored best parameters independently for unresolved endpoints at fitting stop.',
    target_policy='Unchanged frozen sampled training-margin and validation-classification thresholds, monitored and confirmed independently on the same trajectory.',
    gradients='Historical exact ideal-mean VJP at sampled signed PEC forward score; no measured gradient shots. Simulator-assisted training.',
    accounting='Algorithm shots are fitting plus once-only edge plus measured gradients. Monitoring, confirmation and diagnostics remain separate and count toward online budget when acquired.',
    scheduler=dict(array='0-9%2', cpu_per_task=1, memory_gb_per_task=8, wall_hours=96, max_concurrent_training_tasks_across_current_campaigns=20, walltime_is_queue_limit_not_scientific_stop=True),
    safety=dict(test_access=False,automatic_retry=False,fixed_parameter_PEC_validation_repeated=False,historical_files_edited=False,pec120_remains_held=True)
))
new_json(OUT/'manifest/lineage.json', dict(utc=now(), parent_campaign=str(BASE), parent_manifest_sha256=digest(BASE/'manifest/runs.jsonl'), parent_frozen_execution_sha256=digest(BASE/'manifest/frozen_execution.json'),
    allowed_changes=['run_id','N_fit','N_monitor','N_confirm','N_diagnostic','pec_kernel.protocol accepts 240'],
    pairing='teacher, repetition, initialization, minibatch order, targets and optimizer frozen; shot draws are fresh at 240',
    new_measurement_samples_during_preparation=0))
print(json.dumps(dict(campaign=str(OUT),rows=len(new),protocol_sha256=digest(OUT/'manifest/protocol.json'),parent_manifest_sha256=digest(BASE/'manifest/runs.jsonl'))))
