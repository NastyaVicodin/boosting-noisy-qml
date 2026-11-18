"""Prepare the approved paired PEC210 arm; no sampling or scheduler actions."""
from pathlib import Path
import ast, datetime, hashlib, json, shutil

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
BASE = ROOT/'artifacts/strict_v5_patience100_v1_20261019T140521Z'
PROPOSAL = ROOT/'artifacts/strict_v5_overnight_pec210_proposal_20261019T210843Z'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, value):
    with Path(p).open('x') as f: json.dump(value, f, indent=2, sort_keys=True); f.write('\n')
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

# Validate the existing frozen implementation before copying it.
frozen = json.loads((BASE/'manifest/frozen_execution.json').read_text())
for entry in frozen['files']:
    path = Path(entry['path'])
    assert '/datasets/test/' not in str(path) and 'test_prediction' not in path.name
    assert sha(path) == entry['sha256'], path
copied = []
for name in ['common.py', 'audit.py', 'endpoints.py', 'lineage.py', 'patience.py',
             'reference_kernel.py', 'train_runner.py', 'pec_measurements.py',
             'pec_kernel.py', 'runner.py', 'environment.sh']:
    src = BASE/'source'/name; dst = OUT/'source'/name
    assert not dst.exists(); shutil.copyfile(src, dst)
    copied.append(dict(source=str(src), destination=str(dst), original_sha256=sha(src)))
for src in sorted((BASE/'frozen_src').rglob('*.py')):
    dst = OUT/'frozen_src'/src.relative_to(BASE/'frozen_src')
    dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, dst)
    copied.append(dict(source=str(src), destination=str(dst), original_sha256=sha(src)))
p = OUT/'source/pec_kernel.py'; text = p.read_text()
assert text.count('assert n in (120,180)') == 1
p.write_text(text.replace('assert n in (120,180)', 'assert n in (120,180,210)'))
p = OUT/'source/runner.py'; text = p.read_text()
assert text.count('choices=range(30)') == 1
p.write_text(text.replace('choices=range(30)', 'choices=range(10)'))
for entry in copied: entry['new_sha256'] = sha(entry['destination'])

rows = [json.loads(s) for s in (PROPOSAL/'proposed_runs.jsonl').read_text().splitlines()]
base_rows = {r['pair_id']:r for r in map(json.loads, (BASE/'manifest/runs.jsonl').read_text().splitlines()) if r['N_fit']==180}
allowed_changes = {'index','kind','run_id','N_fit','N_monitor','N_confirm','N_diagnostic'}
removed = {'guard_result','prior_classification_sources','supersedes_job'}
for i, row in enumerate(rows):
    original = base_rows[row['pair_id']]
    assert row['index']==i and row['kind']=='train'
    assert row['N_fit']==row['N_monitor']==row['N_confirm']==210
    for k,v in original.items():
        if k not in allowed_changes|removed and not k.startswith('source_'): assert row[k]==v, k
    assert row['max_epochs']==500 and row['patience_epochs']==100
assert len(rows)==10 and {(r['teacher'],r['repetition']) for r in rows}=={(t,r) for t in range(5) for r in range(2)}
with (OUT/'manifest/runs.jsonl').open('x') as f:
    for row in rows: f.write(json.dumps(row, sort_keys=True)+'\n')
save(OUT/'manifest/lineage.json', dict(utc=now(), base_campaign=str(BASE),
    base_protocol_sha256=sha(BASE/'manifest/protocol.json'),
    base_frozen_sha256=sha(BASE/'manifest/frozen_execution.json'),
    approved_proposal=str(PROPOSAL/'proposal.json'), proposal_sha256=sha(PROPOSAL/'proposal.json'),
    copied_files=copied, estimator_change='Allow N=210; ideal-mean VJP and optimizer unchanged.',
    pairing='Same ten teachers/initializations/minibatch orders and existing stream namespace. N differs; estimator draws are not claimed identical across N.',
    new_measurement_samples=0, fixed_parameter_validation_repeated=False))
save(OUT/'manifest/protocol.json', dict(schema='strict_v5.pec210_overnight.v1', utc=now(),
    user_authorized=True, authorization='User: yes, do this; accepted overnight PEC180+PEC210 proposal.',
    run_count=10, shot_allocations=[210], max_epochs=500, online_budget_raw_shots=549144980,
    algorithm_budget_30M_applied=False,
    patience=dict(epochs=100, min_delta=0, restore_best=True, metric='Mean of24 sampled smooth-hinge fitting minibatch losses; no warmup; everyepoch'),
    target_policy='Same frozen sampled training-margin and validation-classification targets; separate independent confirmations. Stop when both confirmed, patience100,500epochs or existing online cap. Fresh restored-best checks for unresolved endpoints at patience/epoch stop.',
    gradients='Exact ideal-mean VJP at signed PEC forward score; historical reference/boosting conventions unchanged. Simulator-assisted, no measured gradient acquisitions.',
    accounting='Count completed physical-shot acquisitions once. Algorithm=fitting+edge+measured-gradient; monitoring, confirmation, diagnostics separate. No gamma-squared remultiplication.',
    monitoring='Every completed epoch until each endpoint confirms; training/monitoring/confirmation each210 shots per input.',
    source_campaign=str(BASE), source_proposal=str(PROPOSAL),
    scheduler=dict(pec210_concurrency=10, cpu_per_task=1, memory_gb_per_task=8, wall_hours=96,
                   pec180_array='5256572', pec180_array_throttle=8, original_pec180_tasks=['5254249_0','5254249_1'],
                   max_total_PEC_training_concurrency=20, pec120='remain paused; no resume'),
    report=dict(deadline_local='2026-10-20T14:00:00+02:00', deadline_utc='2026-10-20T12:00:00+00:00',
                deadline_is_snapshot_not_training_stop=True,
                interpretation='Unfinished runs right-censored, not failures; lowest tested successful allocation per pair is not a universal minimum.'),
    safety=dict(test_access=False, automatic_retry=False, fixed_parameter_PEC_validation_repeated=False, original_frozen_files_edited=False)))
print(json.dumps(dict(campaign=str(OUT), rows=len(rows), copied_files=len(copied), new_measurement_samples=0)))
