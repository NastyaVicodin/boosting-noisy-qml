"""Deterministic scalar-only comparison, including incomplete overnight runs."""
from common import *
import subprocess

def report():
    protocol=load_json(OUT/'manifest/protocol.json');base=Path(protocol['source_campaign'])
    # Reuse the frozen, audited merger of reference and historical PEC120 paths.
    proc=subprocess.run([sys.executable,'-B',str(base/'source/report.py')],capture_output=True,text=True,check=True)
    result=json.loads(proc.stdout.strip().splitlines()[-1]);base_path=Path(result['report']).with_suffix('.json')
    payload=load_json(base_path);records=payload['records'];actual=payload['actual_PEC_acquisition_totals']
    evidence=payload['evidence']+[source_record(base_path)];cache={}
    def read_rows(path):
        path=Path(path)
        if not path.exists():return []
        key=str(path)
        if key not in cache:
            data=path.read_bytes()
            cache[key]=[json.loads(s) for s in data.splitlines(keepends=True) if s.endswith(b'\n')]
            evidence.append(dict(path=key,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
        return cache[key]
    def acquired(path):
        intent={};done=set()
        for x in read_rows(path):
            if x['kind']=='acquisition_intent':assert x['event_id'] not in intent;intent[x['event_id']]=x
            if x['kind']=='completed':assert x['event_id'] in intent and x['event_id'] not in done;done.add(x['event_id'])
        totals={r:sum(intent[k]['raw_shots'] for k in done if intent[k]['role']==r) for r in Acquisitions.ROLES}
        return totals,len(set(intent)-done)
    base_rows=jsonl(base/'manifest/runs.jsonl')
    for r in records:
        if r['method']=='PEC120':r['campaign_activity']='paused';r['note']+=' PEC120 remains paused; missing retrospective checks are unresolved.'
        if r['method']=='PEC180':
            row=next(x for x in base_rows if x['pair_id']==r['pair'] and x['N_fit']==180)
            d=Path(row['source_campaign'])/'runs'/row['source_run_id'] if 'source_campaign' in row else base/'runs'/row['run_id']
            checks=read_rows(d/'endpoint_checks.jsonl')
            covered=[v['position']['epoch'] for v in checks if v.get('endpoint')==r['endpoint'] and 'confirmed' in v and v['position']['epoch']<=r['epoch']]
            r['checked_through_epoch']=max(covered,default=r.get('checked_through_epoch') or 0)
    for row in jsonl(OUT/'manifest/runs.jsonl'):
        d=OUT/'runs'/row['run_id'];totals,pending=acquired(d/'acquisitions.jsonl')
        rp=d/'result.json';result=load_json(rp) if rp.exists() else {}
        if rp.exists():evidence.append(source_record(rp))
        updates=read_rows(d/'updates.jsonl');checks=read_rows(d/'endpoint_checks.jsonl')
        current=sum(totals[k] for k in ['fit','edge','gradient'])
        epoch=updates[-1]['update']/24 if updates else 0
        for endpoint in ['margin','classification']:
            hp=d/f'first_hit_{endpoint}.json';hit=load_json(hp) if hp.exists() else None
            if hp.exists():evidence.append(source_record(hp));assert hit['confirmed']
            coverage=max([v['position']['epoch'] for v in checks if v.get('endpoint')==endpoint and 'confirmed' in v],default=0)
            state='HIT' if hit else 'NONHIT' if result else 'ERROR' if (d/'error.json').exists() else 'INCOMPLETE'
            records.append(dict(pair=row['pair_id'],method='PEC210',endpoint=endpoint,status=state,run_complete=bool(result),
                epoch=hit['position']['epoch'] if hit else epoch,
                algorithm_raw_shots=hit['algorithm_raw_shots'] if hit else current,
                confirmation_metric=(hit['confirmation']['sampled_ramp'] if endpoint=='margin' else hit['confirmation']['accuracy']) if hit else None,
                checked_through_epoch=coverage,run_directory=str(d),source_kind='fresh_paired_N210',stop_reason=result.get('stop_reason'),
                note='First confirmed-hit cost; otherwise acquired cost through current/terminal prefix. Incomplete is not failure.'))
        actual.append(dict(cohort='overnight_PEC210',run_id=row['run_id'],pair=row['pair_id'],N=210,**totals,
                           unresolved_intents=pending,ledger=str(d/'acquisitions.jsonl')))
    methods=['Reference25','PEC120','PEC180','PEC210','Ada25','Real25'];aggregate=[];budgets=[]
    for endpoint in ['margin','classification']:
        for method in methods:
            cohort=[r for r in records if r['endpoint']==endpoint and r['method']==method];assert len(cohort)==10
            aggregate.append(dict(endpoint=endpoint,method=method,hits=sum(r['status']=='HIT' for r in cohort),
                terminal_nonhits=sum(r['status']=='NONHIT' for r in cohort),unresolved=sum(r['status'] in ['INCOMPLETE','ERROR'] for r in cohort),
                first_hit_or_stop_or_current_algorithm_raw_shots=sum(r['algorithm_raw_shots'] for r in cohort)))
            for budget in [5000000,10000000,20000000,40000000,80000000]:
                states=[]
                for r in cohort:
                    if r['status']=='HIT':state='hit' if r['algorithm_raw_shots']<=budget else 'no_confirmed_hit_within_budget'
                    elif r['status']=='NONHIT':state='no_confirmed_hit_under_stopping_policy'
                    elif method in ['PEC180','PEC210'] and (r.get('checked_through_epoch') or 0)*1536*int(method[3:])>=budget:state='no_confirmed_hit_within_budget'
                    else:state='unresolved'
                    states.append(state)
                budgets.append(dict(endpoint=endpoint,method=method,algorithm_shot_budget=budget,hits=states.count('hit'),
                    observed_nonhits=sum(s.startswith('no_confirmed') for s in states),unresolved=states.count('unresolved')))
    prefix=OUT/'analysis'/('OVERNIGHT_COMPARISON_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    value=dict(utc=now(),records=records,aggregate=aggregate,common_algorithm_budgets=budgets,
        actual_PEC_acquisition_totals=actual,evidence=evidence,base_report=str(base_path),
        protocol=source_record(OUT/'manifest/protocol.json'),frozen=source_record(OUT/'manifest/frozen_execution.json'),
        test_access=False,new_measurement_samples=0,minimum_claim='Per-pair lowest tested successful allocation only; not a universal threshold.',
        note='Reference and PEC use exact ideal simulator gradients; boosting noisy-mean gradients. Algorithm costs exclude monitoring/confirmation/diagnostics. Shared PEC endpoint fitting is not added twice.')
    new_json(str(prefix)+'.json',value);new_csv(str(prefix)+'.csv',records)
    new_csv(str(prefix)+'_common_budgets.csv',budgets);new_csv(str(prefix)+'_actual_PEC_spend.csv',actual)
    lines=[f'# Overnight PEC comparison — {value["utc"]}','',
        'Algorithm shots = fitting + edge + measured gradient (zero). Monitoring, confirmation and diagnostics are separate. Margin and classification use distinct frozen targets; do not add their shared PEC fitting costs.',
        '', 'HIT means independent sampled-score confirmation. NONHIT means terminal under the stopping policy. INCOMPLETE means unresolved, including paused PEC120. This snapshot does not establish a universal minimum shot allocation.','']
    for endpoint in ['margin','classification']:
        lines += [f'## {endpoint.capitalize()}','','| Method | Confirmed hits | Terminal nonhits | Unresolved | First-hit/stop/current shots, M |','|---|---:|---:|---:|---:|']
        for a in aggregate:
            if a['endpoint']==endpoint:lines.append(f'| {a["method"]} | {a["hits"]}/10 | {a["terminal_nonhits"]} | {a["unresolved"]} | {a["first_hit_or_stop_or_current_algorithm_raw_shots"]/1e6:.3f} |')
        lines+=['','Cells: status; epoch; algorithm shots in millions. Boosting epochs sum all weak learners.','','| Pair | '+' | '.join(methods)+' |','|---|'+'---|'*len(methods)]
        for pair in sorted({r['pair'] for r in records}):
            cells=[]
            for method in methods:
                r=next(r for r in records if r['pair']==pair and r['method']==method and r['endpoint']==endpoint)
                cells.append(f'{r["status"]}; {r["epoch"]:g}; {r["algorithm_raw_shots"]/1e6:.3f}')
            lines.append('| '+pair+' | '+' | '.join(cells)+' |')
        lines+=['']
    lines+=['The common_budgets CSV reports success at shared algorithm-shot budgets, with insufficiently observed prefixes unresolved. The actual_PEC_spend CSV retains all acquired costs, including superseded work. A successful allocation on one pair does not establish cohort reliability or hardware-training cost.','',f'Protocol SHA256: `{value["protocol"]["sha256"]}`. Frozen SHA256: `{value["frozen"]["sha256"]}`.']
    with Path(str(prefix)+'.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(report=str(prefix)+'.md',aggregate=aggregate)))
if __name__=='__main__':report()
