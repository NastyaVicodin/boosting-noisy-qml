"""Endpoint events and immutable first-hit snapshots, separate from fit work."""
from audit import *

def algorithm_shots(ledger):
    t=ledger.totals();return ledger.inherited_algorithm_raw_shots+sum(t[k] for k in ['fit','edge','gradient'])

def endpoint_seed(*parts):
    return seed('lowshot_depth_noise_3x3_dual.v1',*parts)

class EndpointTracker:
    def __init__(self,directory,ledger):
        self.directory=Path(directory);self.ledger=ledger;self.hits={};self.latest={}
    def endpoint_cost(self,endpoint):
        counts={r:0 for r in Acquisitions.ROLES}
        for eid,e in self.ledger.events.items():
            if eid in self.ledger.done and e.get('endpoint')==endpoint:counts[e['role']]+=e['raw_shots']
        return counts
    def check(self,endpoint,position,model_id,evaluate,*,algorithm_cost=None,algorithm_origin='current_fit'):
        if endpoint in self.hits:return self.hits[endpoint]
        event=dict(endpoint=endpoint,position=position,model_id=model_id,utc=now(),confirmation_attempted=False)
        try:
            event['monitor']=evaluate('monitor')
            if event['monitor']['passes']:
                event['confirmation_attempted']=True
                event['confirmation']=evaluate('confirm')
            hit=bool(event.get('confirmation',{}).get('passes',False))
            event['confirmed']=hit
            event['algorithm_raw_shots']=algorithm_shots(self.ledger) if algorithm_cost is None else int(algorithm_cost)
            event['algorithm_origin']=algorithm_origin
            event['endpoint_shot_totals']=self.endpoint_cost(endpoint)
            event['workflow_receipt_at_check']=self.ledger.receipt()
            self.latest[endpoint]=event
            if hit:
                self.hits[endpoint]=event
                new_json(self.directory/f'first_hit_{endpoint}.json',event)
            return event
        except BudgetReached as exc:
            event['interrupted_by_budget']=str(exc);raise
        finally:
            append_json(self.directory/'endpoint_checks.jsonl',event)

def classification_metrics(scores,labels,minimum_correct):
    scores=np.asarray(scores);labels=np.asarray(labels)
    if not np.all(np.isfinite(scores)):raise FloatingPointError('Nonfinite classification scores')
    correct=int(np.sum(np.where(scores>=0,1,-1)==labels))
    return dict(correct=correct,accuracy=correct/len(labels),minimum_correct=int(minimum_correct),passes=correct>=minimum_correct,
                executed_score_ties=int(np.sum(scores==0)))
