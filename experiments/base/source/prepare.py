"""Prepare immutable 25-shot, three-by-three dual-endpoint depth/noise matrix."""
from common import *
from reference_kernel import array_hash
from boosting_qml.pqc.ansatz import initialize_reuploading_params
import jax
import numpy as np
import math

GRID={1:[.005,.04],2:[.005,.03],3:[.005,.01,.02],4:[.005,.05],5:[.005,.05]}
TEACHERS={1:[1,2,4],2:[2,3,4],3:[2,3,4],4:[0,1,4],5:[0,1,2]}
OLD=ROOT/'artifacts/strict_v5_stored_teacher_amplification_review_20261019T224008Z/all_teacher_noise_cells.csv'
PLAN=ROOT/'artifacts/strict_v5_lowshot_depth_noise_3x3_plan_20261020T102700Z/PLAN.md'
CLASS=ROOT/'artifacts/strict_v5_zero_physical_noise_downstream_reanalysis_v1/classification/corrected_classification_targets.csv'
PILOT=ROOT/'artifacts/strict_v5_iid_high_margin_pilot_v1'
EXP=ROOT/'artifacts/strict_v5_final_depth_expansion_v1'

def data(depth,teacher):
    root=PILOT if depth in [2,3] else EXP
    key=f'depth_{depth}_task_{teacher:02d}'
    margin=root/'calibration/frozen_margin_targets.json'
    epsilon=float(load_json(margin)['epsilon_mar_by_depth_task'][key])
    if depth in [2,3]:
        theta=root/f'teacher_parameters/{key}.npy'
        train=root/f'datasets/train/{key}.npz'
        validation=root/f'datasets/validation/{key}.npz'
    else:
        theta=root/f'teachers/{key}.npy'
        train=root/f'datasets/d{depth}/train/task_{teacher:02d}.npz'
        validation=root/f'datasets/d{depth}/validation/task_{teacher:02d}.npz'
    for path,n in [(train,1536),(validation,384)]:
        assert '/test/' not in str(path)
        with np.load(path,allow_pickle=False) as z:
            assert z['X'].shape==(n,4) and z['y'].shape==(n,)
            assert set(np.unique(z['y']))=={-1.,1.}
    assert np.load(theta,allow_pickle=False).shape==(4*depth+3,4)
    return dict(epsilon_mar=epsilon,margin_target_registry=str(margin),
        margin_target_registry_sha256=digest(margin),teacher_path=str(theta),
        teacher_sha256=digest(theta),train_split=str(train),train_split_sha256=digest(train),
        validation_split=str(validation),validation_split_sha256=digest(validation))

def main():
    assert digest(OLD)=='fd3d49adb7b3ca37b26432c8598b2d49f1391ee68008da2e1b07617a0019c2b2'
    targets={(int(r['depth']),int(r['teacher_task'])):float(r['a_star_corrected'])
             for r in csv.DictReader(CLASS.open())}
    rows=[];pairing=[]
    for depth in sorted(GRID):
        for teacher in TEACHERS[depth]:
            shared=data(depth,teacher)
            a_star=targets[(depth,teacher)]
            assert 0<a_star<=1
            for rep in range(3):
                pair=f'd{depth}_t{teacher:02d}_r{rep:02d}'
                initial_seed=seed('initialization',depth,teacher,rep)
                theta=np.asarray(initialize_reuploading_params(
                    jax.random.PRNGKey(initial_seed),n_qubits=4,n_layers=depth,
                    scheme='block_normal'),dtype=np.float32)
                assert theta.shape==(4*depth+3,4)
                order=np.stack([np.random.default_rng(seed('permutation',depth,teacher,rep,epoch))
                    .permutation(1536).astype(np.int32) for epoch in range(500)])
                initial_path=OUT/f'inputs/{pair}_initial.npy'
                order_path=OUT/f'inputs/{pair}_orders.npy'
                with initial_path.open('xb') as f:np.save(f,theta,allow_pickle=False)
                with order_path.open('xb') as f:np.save(f,order,allow_pickle=False)
                common=dict(pair_id=pair,depth=depth,teacher=teacher,repetition=rep,
                    a_star=a_star,minimum_correct=int(math.ceil(a_star*384)),
                    classification_target_id=f'corrected_classification:d{depth}:t{teacher:02d}',
                    margin_target_id=f'frozen_margin:d{depth}:t{teacher:02d}',
                    classification_target_registry=str(CLASS),
                    classification_target_registry_sha256=digest(CLASS),
                    initialization_seed=initial_seed,initial_path=str(initial_path),
                    initial_file_sha256=digest(initial_path),initial_array_sha256=array_hash(theta),
                    orders_path=str(order_path),orders_sha256=digest(order_path),
                    orders_array_sha256=array_hash(order),**shared)
                pairing.append(common)
                for noise in GRID[depth]:
                    tag=str(noise).replace('.','p')
                    if noise==GRID[depth][0]:
                        rows.append(dict(index=len(rows),run_id=f'{pair}_reference25',
                            method='reference',noise_strength_1q=0.,noise_strength_2q=0.,
                            N_fit=25,N_monitor=25,N_confirm=25,N_gradient=0,
                            max_epochs=500,max_updates=12000,patience_epochs=100,
                            patience_min_delta=0,restore_best=True,
                            reference_stream_seeds={role+'_seed':seed('reference',role,pair)
                                for role in ['fit','monitor','confirm','diagnostic']},
                            online_budget_raw_shots=549144980,**common))
                    for method in ['adaboost_r','real_adaboost']:
                        rows.append(dict(index=len(rows),run_id=f'{pair}_p{tag}_{method}',
                            method=method,noise_strength_1q=noise,noise_strength_2q=noise,
                            N_fit=25,N_edge=25,N_monitor=25,N_confirm=25,N_gradient=0,
                            max_rounds=500,max_epochs_per_weak=500,weak_patience_epochs=20,
                            outer_patience_rounds=20,boost_seed=seed('boost',pair,method),
                            online_budget_raw_shots=549144980,**common))
    assert len(rows)==243 and len({r['run_id'] for r in rows})==len(rows)
    with (OUT/'manifest/runs.jsonl').open('x') as f:
        for r in rows:f.write(json.dumps(r,sort_keys=True)+'\n')
    new_json(OUT/'manifest/protocol.json',dict(
        schema='strict_v5.lowshot_depth_noise_3x3_dual.v1',utc=now(),
        status='PREPARED_NOT_LAUNCHED',user_request='3 teachers x3 repetitions per cell;25 shots; dual endpoints;70/30 sweep/PEC slots',
        grid=GRID,teachers=TEACHERS,methods=['corrected_p0_reference','adaboost_r','real_adaboost'],
        run_count=len(rows),reference_runs=45,boosting_runs=198,
        shots_per_input_by_role=dict(fit=25,edge=25,monitor=25,confirm=25),
        target_rule='At least2 of3 repetitions per teacher; at least2 of3 teachers per cell, separately per method and endpoint',
        margin_target='Frozen full-training empirical normalized ramp at mu=.1',
        classification_target='Frozen corrected matched validation accuracy,384 inputs,score>=0 predicts +1',
        reference_stop='Dual endpoint; each completed epoch monitor unresolved endpoints, independent confirmation on monitor pass; stop at both hits,100 patience epochs strictly lower sampled mean epoch fit loss, or500 cap; restore best and final check unresolved endpoints',
        boost_stop='One shared trajectory per method; cached edge-score margin monitor after each completed weak learner, independent full-prefix margin confirmation on pass; sampled full-prefix validation monitor and independent confirmation; preserve each first hit and continue until both hit, outer20-round exact-mean patience,500 round cap or online cap; weak patience20,500 epoch cap',
        gradient='Historical sampled forward with exact ideal-mean reference VJP and exact noisy-mean boosting VJP; zero measured-gradient shots; simulator-assisted, not hardware training',
        algorithm_cost='Fitting+edge+measured-gradient through each endpoint first hit; monitor/confirm/diagnostic separately; no double charge for cached edge predictions',
        projected_pec='gamma_squared(depth,p) times matched p=0 reference algorithm cost; derived estimator variance-matching projection, no PEC measurement or necessary learning cost',
        source_selection=dict(historical_validation_scalar_csv=source_record(OLD),plan=source_record(PLAN)),
        frozen_targets=[source_record(CLASS),source_record(PILOT/'calibration/frozen_margin_targets.json'),
                        source_record(EXP/'calibration/frozen_margin_targets.json')],
        fixed_pairing_across_methods_and_noise=True,test_access=False,
        scheduler=dict(total_training_slots=20,sweep_slots=14,PEC_slots=6,
                       high_noise_illustration_slots=4,backbone_and_reference_slots=10,
                       no_automatic_retry=True),
        upper_cap_online_raw_shots_per_run=549144980,
        caveat='p=.05 at d4/d5 is outside original main grid; high-noise tasks may not finish by figure deadline; incomplete cohorts remain censored'))
    new_json(OUT/'manifest/pairing_summary.json',dict(utc=now(),pairs=pairing))
    print(json.dumps(dict(runs=len(rows),references=45,boosting=198,protocol_sha256=digest(OUT/'manifest/protocol.json'))))
if __name__=='__main__':main()
