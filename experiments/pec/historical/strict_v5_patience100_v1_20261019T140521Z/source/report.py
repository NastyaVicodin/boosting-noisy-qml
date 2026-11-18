"""Read-only result aggregation; writes new timestamped reports, never source artifacts."""
from common import *
from patience import scan_updates
import argparse

def rows_at(path):
    p=Path(path)
    return [json.loads(s) for s in p.read_bytes().splitlines(keepends=True) if s.endswith(b'\n')] if p.exists() else []

def acquired(path):
    intent={};done=set()
    for e in rows_at(path):
        if e['kind']=='acquisition_intent':assert e['event_id'] not in intent;intent[e['event_id']]=e
        elif e['kind']=='completed':assert e['event_id'] in intent and e['event_id'] not in done;done.add(e['event_id'])
    totals={r:sum(intent[k]['raw_shots'] for k in done if intent[k]['role']==r) for r in Acquisitions.ROLES}
    return dict(totals=totals,unresolved=[v for k,v in intent.items() if k not in done])

def report(final=False):
    timestamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    prefix=OUT/'analysis'/f'{"FINAL" if final else "CURRENT"}_COMPARISON_{timestamp}'
    protocol=load_json(OUT/'manifest/protocol.json');sources=protocol['source_campaigns'];rows=jsonl(OUT/'manifest/runs.jsonl')
    records=[];evidence={};actual=[]
    def bind(path):
        p=Path(path)
        if p.exists():evidence[str(p)]=source_record(p)
        return str(p)
    for row in rows:
        directory=OUT/'runs'/row['run_id'];rp=directory/'result.json';result=load_json(rp) if rp.exists() else {}
        if rp.exists():bind(rp)
        progress=rows_at(directory/'updates.jsonl');newcost=acquired(directory/'acquisitions.jsonl')['totals']['fit']
        base=0;epoch=0;source=None;coverage={};hits={}
        if row['kind']=='pec_review_continue':
            source=Path(row['source_campaign'])/'runs'/row['source_run_id'];updates=rows_at(source/'updates.jsonl');history=scan_updates(updates)
            if history:epoch=history[-1]['epoch'];base=epoch*1536*row['N_fit']
            bind(source/'updates.jsonl');bind(source/'acquisitions.jsonl')
            for ep in ['margin','classification']:
                f=source/f'first_hit_{ep}.json'
                if f.exists():
                    v=load_json(f);bind(f)
                    if v['position']['epoch']<=epoch:hits[ep]=v
            if row['N_fit']==120:
                vals=[v for v in rows_at(source/'evaluations.jsonl') if v['update']<=epoch*24];bind(source/'evaluations.jsonl')
                coverage['margin']=max([v['update']//24 for v in vals],default=0)
                for v in vals:
                    if v.get('confirm',{}).get('passes'):
                        hits['margin']=dict(position=dict(epoch=v['update']//24),algorithm_raw_shots=v['update']*64*120,confirmation=v['confirm']);break
                covered=set()
                for bound in row.get('prior_classification_sources',[]):
                    p=Path(bound['path']);assert digest(p)==bound['sha256'];bind(p)
                    for v in rows_at(p):
                        pos=v.get('position',{})
                        if v.get('endpoint')=='classification' and 'confirmed' in v and pos.get('epoch',501)<=epoch and 'restored' not in pos.get('phase',''):
                            covered.add(pos['epoch'])
                            if v['confirmed']:hits['classification']=v
                coverage['classification']=max((e for e in covered if all(i in covered for i in range(1,e+1))),default=0)
        if row['kind']=='reference_review' and not result:
            us={ep:rows_at(Path(b['campaign'])/'runs'/b['run_id']/'updates.jsonl') for ep,b in row['reference_sources'].items()}
            history=scan_updates(max(us.values(),key=len));epoch=history[-1]['epoch'];base=epoch*1536*25
            for ep,b in row['reference_sources'].items():
                p=Path(b['campaign'])/'runs'/b['run_id']/'result.json';v=load_json(p);bind(p)
                if v.get('selected') and v['selected']['update']//24<=epoch:
                    sel=v['selected'];hits[ep]=dict(position=dict(epoch=sel['update']//24),algorithm_raw_shots=sel['update']*64*25,confirmation=sel['confirmation'])
        if progress:epoch=progress[-1]['update']/24;bind(directory/'updates.jsonl')
        for ep in ['margin','classification']:
            p=directory/f'first_hit_{ep}.json'
            if p.exists():hits[ep]=load_json(p);bind(p)
        checks=rows_at(directory/'endpoint_checks.jsonl')+rows_at(directory/'reused_classification_checks.jsonl')
        for ep in ['margin','classification']:
            observed=[v['position']['epoch'] for v in checks if v.get('endpoint')==ep and 'confirmed' in v]
            if observed:coverage[ep]=max(coverage.get(ep,0),max(observed))
        cost=result.get('algorithm_raw_shots',base+newcost);epoch=result.get('fit_epochs',epoch)
        for ep in ['margin','classification']:
            hit=result.get('first_hits',hits).get(ep)
            state='HIT' if hit else 'NONHIT' if result else 'ERROR' if (directory/'error.json').exists() else 'INCOMPLETE'
            records.append(dict(pair=row['pair_id'],method='Reference25' if row['N_fit']==25 else f"PEC{row['N_fit']}",endpoint=ep,
                status=state,run_complete=bool(result),epoch=hit['position']['epoch'] if hit else epoch,algorithm_raw_shots=hit['algorithm_raw_shots'] if hit else cost,
                confirmation_metric=(hit.get('confirmation',{}).get('sampled_ramp',hit.get('confirmation',{}).get('sampled_ramp_mu0p1')) if ep=='margin' else hit.get('confirmation',{}).get('accuracy',hit.get('confirmation',{}).get('correct',0)/384)) if hit else None,
                checked_through_epoch=coverage.get(ep),run_directory=str(directory),source_kind=result.get('source_kind',row['kind']),stop_reason=result.get('stop_reason'),
                note='First-hit cost if hit; otherwise cost through stop/current canonical prefix. Incomplete is not failure.'))
    for ep,key in [('margin','original_pec120'),('classification','pec180')]:
        camp=Path(sources[key])
        for row in jsonl(camp/'manifest/runs.jsonl'):
            if row['method'] not in ['adaboost_r','real_adaboost']:continue
            directory=camp/'runs'/row['run_id'];p=directory/'result.json';v=load_json(p);bind(p)
            if ep=='margin':
                totals=v['all_attempts_before_final_validation']['shot_totals'];cost=sum(totals[k] for k in ['fit','edge','gradient'])
                hit=v['status']=='MARGIN_TARGET_CONFIRMED';metric=v['stopping_summary']['hit_train_margin_loss']
            else:
                event=v['first_hits']['classification'];totals=event['workflow_receipt_at_check']['shot_totals'];cost=event['algorithm_raw_shots'];hit=True;metric=event['confirmation']['accuracy']
            records.append(dict(pair=row['pair_id'],method='Ada25' if row['method']=='adaboost_r' else 'Real25',endpoint=ep,status='HIT' if hit else 'NONHIT',
                run_complete=True,epoch=totals['fit']/(1536*25),algorithm_raw_shots=cost,confirmation_metric=metric,round=v['selected_round'],
                run_directory=str(directory),source_kind='completed_original_boosting_cohort',note='Cumulative fitting epochs over all attempted weak learners; algorithm cost includes edge acquisitions.'))
    # Actual expenditure is kept separate; every physical acquisition ledger counted once by absolute path.
    ledger_paths={}
    for label,camp in [('original_PEC120',Path(sources['original_pec120'])),('original_PEC180_and_classification_reviews',Path(sources['pec180'])),
                       ('superseded_patience20_PEC',Path(sources['previous'])),('current_patience100_PEC',OUT)]:
        for row in jsonl(camp/'manifest/runs.jsonl'):
            if row.get('method')!='explicit_sampled_pec':continue
            p=camp/'runs'/row['run_id']/'acquisitions.jsonl'
            if p.exists():ledger_paths[str(p.resolve())]=(label,row,p)
    for label,row,p in ledger_paths.values():
        a=acquired(p);bind(p);actual.append(dict(cohort=label,run_id=row['run_id'],pair=row['pair_id'],N=row['N_fit'],**a['totals'],unresolved_intents=len(a['unresolved']),ledger=str(p)))
    aggregate=[]
    for ep in ['margin','classification']:
        for method in ['Reference25','PEC120','PEC180','Ada25','Real25']:
            cohort=[r for r in records if r['endpoint']==ep and r['method']==method];assert len(cohort)==10
            aggregate.append(dict(endpoint=ep,method=method,hits=sum(r['status']=='HIT' for r in cohort),nonhits=sum(r['status']=='NONHIT' for r in cohort),
                incomplete=sum(r['status'] in ['INCOMPLETE','ERROR'] for r in cohort),run_complete=sum(r['run_complete'] for r in cohort),
                first_hit_or_stop_or_current_algorithm_raw_shots=sum(r['algorithm_raw_shots'] for r in cohort)))
    payload=dict(utc=now(),final_scheduler_collection=final,all_current_runs_complete=all((OUT/'runs'/r['run_id']/'result.json').exists() for r in rows),
        protocol=source_record(OUT/'manifest/protocol.json'),frozen=source_record(OUT/'manifest/frozen_execution.json'),records=records,aggregate=aggregate,
        actual_PEC_acquisition_totals=actual,evidence=list(evidence.values()),test_access=False)
    new_json(str(prefix)+'.json',payload);new_csv(str(prefix)+'.csv',records);new_csv(str(prefix)+'_actual_PEC_spend.csv',actual)
    lines=[f'# Patience100 comparison — {payload["utc"]}', '',
        'Reference and PEC:100 consecutive completed epochs without a strictly lower mean sampled fitting loss;500-epoch cap and existing online cap. Boosting:20 per weak learner. Targets and gradient conventions unchanged.', '',
        '**Algorithm shots only:** fitting + edge + measured-gradient acquisitions (gradient acquisitions are zero). Monitoring, confirmation and diagnostics excluded. PEC120 means120 raw physical shots per expectation estimate; no additional gamma-squared multiplier.', '',
        'HIT means independently confirmed sampled-score success. NONHIT is a completed unresolved target. INCOMPLETE includes queued runs and unfinished retrospective checks; it is not a failure. Reference reuses matched historical fitting trajectories after verifying shared parameter hashes and losses.', '',
        'Cell format: status; cumulative fitting epochs; algorithm shots in millions. For hits, epochs/cost are at first confirmation. For nonhits, at stopping. For incomplete rows, current canonical fitting prefix; classification checks may lag that prefix. Boosting epochs sum all weak-learner fitting.', '']
    for ep in ['margin','classification']:
        lines += [f'## {ep.capitalize()}', '', '| Pair | Reference25 | PEC120 | Ada25 | Real25 |','|---|---|---|---|---|']
        for t in range(5):
            for rep in range(2):
                pair=f'd3_t{t:02d}_r{rep:02d}';cells=[]
                for method in ['Reference25','PEC120','Ada25','Real25']:
                    r=next(r for r in records if r['pair']==pair and r['method']==method and r['endpoint']==ep)
                    cells.append(f"{r['status']}; {r['epoch']:g}; {r['algorithm_raw_shots']/1e6:.3f}")
                lines.append('| '+pair+' | '+' | '.join(cells)+' |')
        lines+=['','| Method | Confirmed hits | Completed nonhits | Incomplete targets | Sum of first-hit/stop/current algorithm shots, M |','|---|---:|---:|---:|---:|']
        for a in aggregate:
            if a['endpoint']==ep:lines.append(f"| {a['method']} | {a['hits']}/10 | {a['nonhits']} | {a['incomplete']} | {a['first_hit_or_stop_or_current_algorithm_raw_shots']/1e6:.3f} |")
        lines+=['']
    lines+=['## Interpretation and accounting','',
        'Margin and classification tables are separate endpoint comparisons. Do not add their costs: reference/PEC dual-target paths share training. Boosting tables come from separate margin and classification cohorts.', '',
        'Historical work after a derived patience cutoff remains actual expenditure. Replacement runs and repeated partial epochs also remain expenditure; they are not silently removed. The companion actual_PEC_spend CSV lists every original, superseded and current PEC ledger separately. Reference historical cohorts also involved duplicated real work; their merged comparison is a verified derived single-trajectory policy cost.', '',
        'All training remains simulator-assisted: sampled forward scores with exact simulator derivatives. These are not measurement-only hardware-training costs or deterministic-mean guarantees.', '',
        'PEC180 is included in the aggregate and CSV; the requested per-pair tables focus on PEC120. Full provenance, manifest hashes, timestamps, confirmation values and classification coverage are in the companion JSON.', '',
        f'All current run results present: **{payload["all_current_runs_complete"]}**. Scheduler-terminal errors/timeouts, if any, remain incomplete. No automatic retry.', '',
        f'Protocol SHA256: `{payload["protocol"]["sha256"]}`. Frozen manifest SHA256: `{payload["frozen"]["sha256"]}`.']
    with Path(str(prefix)+'.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(report=str(prefix)+'.md',aggregate=aggregate,all_current_runs_complete=payload['all_current_runs_complete'])))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');report(parser.parse_args().final)
