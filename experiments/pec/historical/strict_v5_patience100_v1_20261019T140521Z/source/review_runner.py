"""Reuse existing endpoint acquisitions; continue only complete optimizer states."""
from endpoints import *
from lineage import *
from patience import scan_updates
from pec_measurements import evaluate
import jax.numpy as jnp

def retain(tracker,directory,event):
    ep=event['endpoint']
    if ep not in tracker.hits:
        tracker.hits[ep]=event;new_json(directory/f'first_hit_{ep}.json',event)

def final_check(row,directory,ledger,tracker,source,updates,cut,best,evaluator):
    if len(tracker.hits)==2:return False
    theta,model=load_theta(source,best,updates)
    for ep,split in [('margin','train_split'),('classification','validation_split')]:
        if ep in tracker.hits:continue
        x,y=load_split(row,split)
        tracker.check(ep,dict(update=cut*24,epoch=cut,selected_model_epoch=best,phase='patience100_restored_best'),model,
            lambda role,ep=ep,x=x,y=y:evaluator(theta,x,y,row,role,cut*24,ledger,ep,phase='patience100_restored_best'),
            algorithm_cost=cut*1536*row['N_fit'],algorithm_origin='fitting_through_patience100_or_cap')
    return True

def prior_checks(row,cut):
    prior={};lineage=[]
    for record in row.get('prior_classification_sources',[]):
        path=Path(record['path']);assert digest(path)==record['sha256'];lineage.append(source_record(path))
        for v in complete_rows(path):
            if v.get('endpoint')!='classification' or 'confirmed' not in v:continue
            pos=v['position'];epoch=pos['epoch']
            if epoch>cut or 'restored' in pos.get('phase','') or 'selected_model_epoch' in pos:continue
            if epoch in prior:
                assert prior[epoch]['confirmed']==v['confirmed']
                for key in ['correct','sampled_ramp','passes']:
                    assert prior[epoch]['monitor'].get(key)==v['monitor'].get(key)
            else:prior[epoch]=dict(v,reused_from=record)
    return prior,lineage

def run_review(row,directory,ledger):
    campaign=Path(row['source_campaign']);source=campaign/'runs'/row['source_run_id'];n=row['N_fit']
    assert digest(campaign/'manifest/protocol.json')==row['source_protocol_sha256']
    assert digest(campaign/'manifest/frozen_execution.json')==row['source_frozen_sha256']
    if row.get('guard_result'):assert Path(row['guard_result']).exists()
    updates=complete_rows(source/'updates.jsonl');history=scan_updates(updates);assert history
    decision=history[-1];cut=decision['epoch'];best=decision['best_epoch'];cut_update=cut*24
    account=source_ledger(source/'acquisitions.jsonl',n,cut_update)
    assert account['prefix'][cut_update]==cut*1536*n
    ledger.inherited_algorithm_raw_shots=account['prefix'][cut_update]
    ledger.inherited_online_raw_shots=sum(account['prefix_roles'].values())
    tracker=EndpointTracker(directory,ledger)
    for p in source.glob('first_hit_*.json'):
        v=load_json(p)
        if v['position']['epoch']<=cut and 'restored' not in v['position'].get('phase',''):
            retain(tracker,directory,dict(v,retained_original_evidence=source_record(p)))
    if n==120:
        for v in complete_rows(source/'evaluations.jsonl'):
            if v['update']>cut_update:break
            if v.get('confirm',{}).get('passes',False):
                retain(tracker,directory,dict(endpoint='margin',position=dict(update=v['update'],epoch=v['update']//24,phase='original_epoch'),
                    monitor=v['monitor'],confirmation=v['confirm'],confirmed=True,algorithm_raw_shots=account['prefix'][v['update']],
                    original_source=source_record(source/'evaluations.jsonl')));break
    prior,prior_lineage=prior_checks(row,cut)
    new_json(directory/'historical_status.json',dict(utc=now(),source_run=str(source),available_full_epochs=len(updates)//24,
        policy_prefix_epoch=cut,best_epoch=best,patience_decision=decision,first_hits=tracker.hits,
        existing_classification_coverage_epochs=sorted(prior),actual_original_shot_totals=account['actual'],
        canonical_prefix_algorithm_raw_shots=account['prefix'][cut_update],review_incomplete=True))
    inventory=[]
    if n==120:
        vx,vy=load_split(row,'validation_split')
        for epoch in range(1,cut+1):
            if 'classification' in tracker.hits:break
            if epoch in prior:
                event=dict(prior[epoch],classical_reuse_of_existing_retrospective_evaluation=True,new_raw_shots=0)
                ledger.inherited_online_raw_shots+=384*n*(1+bool(event.get('confirmation_attempted',bool(event.get('confirmation')))))
                append_json(directory/'reused_classification_checks.jsonl',event)
                assert event['algorithm_raw_shots']==account['prefix'][epoch*24]
                tracker.latest['classification']=event
                if event['confirmed']:retain(tracker,directory,event)
            else:
                theta,model=load_theta(source,epoch,updates);inventory.append(model)
                event=tracker.check('classification',dict(update=epoch*24,epoch=epoch,phase='retrospective_epoch'),model,
                    lambda role:evaluate(theta,vx,vy,row,role,epoch*24,ledger,'classification'),
                    algorithm_cost=account['prefix'][epoch*24],algorithm_origin='historical_fitting_prefix')
            append_json(directory/'review_progress.jsonl',dict(epoch=epoch,utc=now(),classification_confirmed='classification' in tracker.hits,
                new_online_shots=ledger.online()-ledger.inherited_online_raw_shots))
    new_json(directory/'source_lineage.json',dict(source_run=str(source),updates=source_record(source/'updates.jsonl'),
        ledger=source_record(source/'acquisitions.jsonl'),consulted_checkpoints=inventory,patience_history=history,prior_classification_sources=prior_lineage))
    actual_metadata=dict(actual_original_shot_totals=account['actual'],original_unresolved_intents=account['unresolved'],
        original_completed_fit_outside_reused_prefix=account['actual']['fit']-account['prefix'][cut_update],
        accounting_note='Canonical fitting prefix counted once. Actual original expenditure and completed partial/beyond-cutoff work retained separately; new fitting is additionally charged.')
    if len(tracker.hits)<2 and not decision['should_stop'] and cut<row['max_epochs']:
        # Only the old PEC120 campaign has full Adam leaves. No theta-only restart.
        assert n==120 and not row.get('guard_result')
        theta,model=load_theta(source,cut,updates);theta2,state=load_state(model['checkpoint'],cut_update)
        assert array_hash(theta)==array_hash(theta2)
        best_theta,best_model=load_theta(source,best,updates)
        new_json(directory/'continuation_start.json',dict(utc=now(),global_update=cut_update,checkpoint=model,best_checkpoint=best_model,
            inherited_algorithm_raw_shots=ledger.inherited_algorithm_raw_shots,inherited_online_raw_shots=ledger.inherited_online_raw_shots,
            actual_original_metadata=actual_metadata,optimizer_restored=True,fit_seed_and_global_indices_preserved=True))
        from train_runner import run_train
        result=run_train(row,directory,ledger,theta,checked_orders(row),tracker=tracker,
            resume=dict(theta=theta,state=state,epoch=cut,history=history,best_theta=best_theta,best_checkpoint=best_model['checkpoint']))
        return dict(**result,source_kind='exact_optimizer_continuation_after_retrospective_endpoint_review',
            continued_from_epoch=cut,new_fit_raw_shots=ledger.totals()['fit'],**actual_metadata)
    restored=final_check(row,directory,ledger,tracker,source,updates,cut,best,evaluate)
    return dict(status='REVIEW_BOTH_TARGETS_CONFIRMED' if len(tracker.hits)==2 else 'REVIEW_PARTIAL_HIT' if tracker.hits else 'REVIEW_NONHIT',
        source_kind='retrospective_patience100_correction',stop_epoch=cut,fit_epochs=cut,stop_reason='both_targets_confirmed' if len(tracker.hits)==2 and not restored else 'patience100' if decision['should_stop'] else 'epoch_cap',
        best_epoch=best,patience=decision,first_hits=tracker.hits,latest_checks=tracker.latest,best_parameters_restored=restored,
        cutoff_algorithm_raw_shots=account['prefix'][cut_update],algorithm_raw_shots=account['prefix'][cut_update],new_fit_raw_shots=0,
        **actual_metadata,**ledger.receipt())

def run_reference_review(row,directory,ledger):
    sources={};updates={};results={};lineage={};accounts={}
    for ep,binding in row['reference_sources'].items():
        campaign=Path(binding['campaign']);source=campaign/'runs'/binding['run_id']
        assert digest(campaign/'manifest/protocol.json')==binding['protocol_sha256']
        assert digest(campaign/'manifest/frozen_execution.json')==binding['frozen_sha256']
        sources[ep]=source;updates[ep]=complete_rows(source/'updates.jsonl');results[ep]=load_json(source/'result.json')
        lineage[ep]={name:source_record(source/name) for name in ['updates.jsonl','result.json','acquisitions.jsonl']}
    overlap=verify_reference_overlap(updates)
    longest=max(updates,key=lambda ep:len(updates[ep]));history=scan_updates(updates[longest]);decision=history[-1];cut=decision['epoch']
    tracker=EndpointTracker(directory,ledger)
    for ep,result in results.items():
        v=result.get('selected')
        if v and v['update']<=cut*24:
            assert v['monitor']['passes'] and v['confirmation']['passes']
            retain(tracker,directory,dict(endpoint=ep,position=dict(update=v['update'],epoch=v['update']//24,phase='original_epoch'),
                monitor=v['monitor'],confirmation=v['confirmation'],confirmed=True,algorithm_raw_shots=v['update']*64*25,
                retained_original_evidence=lineage[ep]['result.json']))
    if len(tracker.hits)==2:cut=max(v['position']['epoch'] for v in tracker.hits.values());decision=history[cut-1]
    ledger.inherited_algorithm_raw_shots=cut*1536*25
    ledger.inherited_online_raw_shots=ledger.inherited_algorithm_raw_shots
    for ep,source in sources.items():
        accounts[ep]=source_ledger(source/'acquisitions.jsonl',25,cut*24)
        ledger.inherited_online_raw_shots+=sum(accounts[ep]['prefix_roles'][r] for r in ['monitor','confirm'])
    from train_runner import reference_evaluate
    restored=final_check(row,directory,ledger,tracker,sources[longest],updates[longest],cut,decision['best_epoch'],reference_evaluate)
    new_json(directory/'source_lineage.json',dict(sources=lineage,identical_overlapping_updates=overlap,
        shared_fields=['post_theta_sha256','loss'],chosen_fit_source=longest,patience_history=history[:cut]))
    return dict(status='REVIEW_BOTH_TARGETS_CONFIRMED' if len(tracker.hits)==2 else 'REVIEW_PARTIAL_HIT' if tracker.hits else 'REVIEW_NONHIT',
        source_kind='matched_historical_reference_trajectory_reuse',stop_epoch=cut,fit_epochs=cut,best_epoch=decision['best_epoch'],patience=decision,
        stop_reason='patience100' if decision['should_stop'] else 'both_targets_confirmed' if len(tracker.hits)==2 else 'epoch_cap',
        first_hits=tracker.hits,latest_checks=tracker.latest,best_parameters_restored=restored,new_fit_raw_shots=0,
        algorithm_raw_shots=ledger.inherited_algorithm_raw_shots,
        actual_historical_source_shot_totals={ep:a['actual'] for ep,a in accounts.items()},
        accounting_note='Derived single paired training trajectory, verified identical overlapping hashes/losses. Actual two historical training cohorts preserved separately; no retraining.',**ledger.receipt())
