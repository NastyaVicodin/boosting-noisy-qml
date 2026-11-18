"""Static source, input, target and stream checks; no circuit evaluation."""
from common import *
from reference_kernel import array_hash
import ast
import importlib
import importlib.metadata
import math
import numpy as np


def preflight():
    frozen=check_frozen()
    protocol=load_json(OUT/'manifest/protocol.json')
    rows=jsonl(OUT/'manifest/runs.jsonl')
    assert frozen['protocol_sha256']==digest(OUT/'manifest/protocol.json')
    assert protocol['user_authorized'] and not protocol['safety']['test_access']
    assert protocol['run_count']==len(rows)==10
    assert [r['index'] for r in rows]==list(range(10))
    assert len({r['run_id'] for r in rows})==len({r['pair_id'] for r in rows})==10
    assert protocol['online_budget_raw_shots']==549144980
    assert protocol['max_epochs']==500 and protocol['max_updates']==12000
    assert protocol['patience']['epochs']==100 and protocol['patience']['min_delta']==0
    assert protocol['patience']['restore_best'] and protocol['patience']['no_warmup']

    proposal=list(csv.DictReader(Path(protocol['proposal_file']).open(newline='')))
    assert digest(protocol['proposal_file'])==protocol['proposal_sha256']
    selected={p['proposed_pec_pair_id']:p for p in proposal if
              (int(p['depth']),float(p['noise']),int(p['pec_shots']))
              in {(3,.005,60),(2,.01,100)}}
    assert len(selected)==10
    refs={(r['depth'],r['teacher'],r['repetition']):r for r in
          jsonl(Path(protocol['reference_campaign'])/'manifest/runs.jsonl')
          if r['method']=='reference'}
    source=Path(protocol['source_campaign'])
    originals=jsonl(source/'manifest/runs.jsonl')
    for row in rows:
        d,n,t,rep=(row['depth'],row['N_fit'],row['teacher'],row['repetition'])
        p=selected[row['pair_id']]
        ref=refs[d,t,rep]
        assert row['kind']=='train' and row['mode']=='pec_dual'
        assert row['method']=='explicit_sampled_pec'
        assert (d,row['noise_strength_1q'],n) in {(3,.005,60),(2,.01,100)}
        assert row['noise_strength_2q']==row['noise_strength_1q']
        assert row['N_fit']==row['N_monitor']==row['N_confirm']==row['N_diagnostic']==n
        assert row['N_edge']==row['N_gradient']==0
        assert row['paper_teacher']==int(p['paper_teacher'])
        assert t==(2,3,4)[row['paper_teacher']]
        assert row['comparison_pair_id']==ref['pair_id']==p['comparison_pair_id']
        assert row['reference_run_id']==ref['run_id']
        assert row['patience_epochs']==100 and row['patience_min_delta']==0 and row['restore_best']
        assert row['max_epochs']==500 and row['max_updates']==12000
        assert row['mu']==.1 and row['batch_size']==64 and row['monitor_every_updates']==24
        assert row['gradient_convention']=='exact_ideal_mean_vjp_at_signed_PEC_score'
        assert row['online_budget_raw_shots']==549144980
        assert row['minimum_correct']==ref['minimum_correct']==int(p['classification_minimum_correct'])
        assert row['minimum_correct']==math.ceil(row['a_star']*384)
        assert row['epsilon_mar']==ref['epsilon_mar']==float(p['margin_target'])
        assert row['margin_target_id']==ref['margin_target_id']
        assert row['classification_target_id']==ref['classification_target_id']
        for k in ('train_split','validation_split','teacher_path','initial_path','orders_path',
                  'train_split_sha256','validation_split_sha256','teacher_sha256',
                  'initial_file_sha256','initial_array_sha256','orders_sha256','orders_array_sha256',
                  'margin_target_registry','margin_target_registry_sha256',
                  'classification_target_registry','classification_target_registry_sha256'):
            assert row[k]==ref[k]
        assert row['initial_array_sha256']==p['initial_array_sha256']
        assert row['orders_array_sha256']==p['orders_array_sha256']
        assert row['train_split_sha256']==p['train_split_sha256']
        assert row['validation_split_sha256']==p['validation_split_sha256']
        assert row['original_pec_template_run_id'] in {r['run_id'] for r in originals}
        for k,h in (('train_split','train_split_sha256'),('validation_split','validation_split_sha256'),
                    ('teacher_path','teacher_sha256'),('initial_path','initial_file_sha256'),
                    ('orders_path','orders_sha256'),('margin_target_registry','margin_target_registry_sha256'),
                    ('classification_target_registry','classification_target_registry_sha256')):
            assert '/test/' not in row[k] and 'test_prediction' not in row[k]
            assert digest(row[k])==row[h]
        initial=np.load(row['initial_path'],allow_pickle=False)
        orders=np.load(row['orders_path'],allow_pickle=False)
        assert initial.shape==(4*d+3,4) and orders.shape==(500,1536)
        assert array_hash(initial)==row['initial_array_sha256']
        assert array_hash(orders)==row['orders_array_sha256']
    for d,n in ((3,60),(2,100)):
        assert len([r for r in rows if (r['depth'],r['N_fit'])==(d,n)])==5
        for t in (2,3,4):
            rr=[refs[d,t,k] for k in (0,1,2)]
            assert len({r['initial_array_sha256'] for r in rr})==3
            assert len({r['orders_array_sha256'] for r in rr})==3
    for original in originals:
        ref=refs[original['depth'],original['teacher'],original['repetition']]
        assert original['initial_array_sha256']==ref['initial_array_sha256']
        assert original['orders_array_sha256']==ref['orders_array_sha256']
    assert len({seed('explicit_pec',r['pair_id'],'fit','u1') for r in rows})==10
    assert len({seed('explicit_pec',p['proposed_pec_pair_id'],'fit','u1') for p in proposal})==40
    # At d3/p=.005, new PEC60 and PEC210 slots use the same saved arrays.
    for row in rows:
        if row['depth']==3:
            counterparts=[p for p in proposal if int(p['depth'])==3 and
                          float(p['noise'])==.005 and int(p['pec_shots'])==210 and
                          int(p['source_teacher'])==row['teacher'] and
                          int(p['repetition'])==row['repetition']]
            assert len(counterparts)==1
            assert counterparts[0]['initial_array_sha256']==row['initial_array_sha256']
            assert counterparts[0]['orders_array_sha256']==row['orders_array_sha256']
    for path in list((OUT/'source').glob('*.py'))+list((OUT/'frozen_src').rglob('*.py')):
        ast.parse(path.read_text())
    for name in ('audit.py','endpoints.py','lineage.py','patience.py','reference_kernel.py',
                 'train_runner.py','pec_measurements.py','pec_kernel.py','environment.sh'):
        assert digest(OUT/'source'/name)==digest(source/'source'/name)
    assert (OUT/'source/common.py').read_text()==(source/'source/common.py').read_text().replace('ROOT = OUT.parents[1]','ROOT = OUT.parents[2]')
    assert (OUT/'source/runner.py').read_text()==(source/'source/runner.py').read_text().replace('choices=range(8)','choices=range(10)')
    runtime=load_json(OUT/'manifest/runtime.json')
    assert sys.executable==runtime['executable']
    for name,version in runtime['versions'].items():
        assert importlib.metadata.version(name)==version
    for name,record in runtime['modules'].items():
        module=importlib.import_module(name)
        assert Path(module.__file__).resolve()==Path(record['path']).resolve()
        assert digest(module.__file__)==record['sha256']
    return dict(status='PASS',utc=now(),run_count=10,
                protocol_sha256=digest(OUT/'manifest/protocol.json'),
                frozen_sha256=digest(OUT/'manifest/frozen_execution.json'),
                files_verified=len(frozen['files']),circuit_evaluations=0,
                measurement_draws=0,test_access=False)


if __name__=='__main__':print(json.dumps(preflight(),sort_keys=True))
