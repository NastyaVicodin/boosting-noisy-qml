"""Synthetic bookkeeping test only: no circuit calls, no measurement sampling."""
from audit import *
from boost_runner import DualEndpointAudit,BothReached
from endpoints import EndpointTracker,algorithm_shots
import tempfile
from types import SimpleNamespace

class FakeWeak:
    def __init__(self,ledger,edge_id,score):
        self.ledger=ledger;self.edge_id=edge_id;self.score=score
        self.params_=np.zeros((7,4),np.float32);self.audit_endpoint=None
    def audit_prediction_phase(self,role):
        from contextlib import nullcontext
        self.audit_role=role
        return nullcontext()
    def predict_scores_jax(self,x,rng):
        eid=self.ledger.begin(self.audit_role,len(x),25,endpoint=self.audit_endpoint)
        self.ledger.finish(eid)
        return np.full(len(x),self.score)

with tempfile.TemporaryDirectory() as tmp:
    d=Path(tmp);ledger=Ledger(d,1000000)
    f=ledger.begin('fit',64,25);ledger.finish(f)
    e=ledger.begin('edge',4,25);ledger.finish(e)
    weak=FakeWeak(ledger,e,1.)
    obj=DualEndpointAudit()
    obj.directory=d;obj.ledger=ledger;obj.row=dict(method='adaboost_r',pair_id='synthetic',
        epsilon_mar=.1,minimum_correct=2)
    obj.round_results=[SimpleNamespace(round_index=1,empirical_train_margin_loss=.05)]
    obj.written=0;obj.weak_learners=[weak];obj.weak_learner_signs=[1.]
    obj.boost_coefs=[1.]
    obj.endpoint_arrays=dict(margin=(np.zeros((4,4)),np.ones(4)),
        classification=(np.zeros((2,4)),np.ones(2)))
    obj.tracker=EndpointTracker(d,ledger)
    try:obj._maybe_write_checkpoint(1)
    except BothReached:pass
    else:raise AssertionError('Did not stop after both confirmations')
    assert set(obj.tracker.hits)=={'margin','classification'}
    assert obj.tracker.hits['margin']['algorithm_raw_shots']==64*25+4*25
    assert obj.tracker.hits['classification']['algorithm_raw_shots']==64*25+4*25
    assert ledger.totals()['monitor']==2*25
    assert ledger.totals()['confirm']==(4+2)*25
    assert all(v['endpoint']!='margin' for v in ledger.events.values() if v['role']=='monitor')
    assert len([v for v in ledger.events.values() if v['role']=='edge'])==1
    assert not obj.tracker.hits['margin']['monitor']['newly_acquired_raw_shots']
with tempfile.TemporaryDirectory() as tmp:
    d=Path(tmp);ledger=Ledger(d,1000000)
    fit1=ledger.begin('fit',64,25);ledger.finish(fit1)
    edge1=ledger.begin('edge',4,25);ledger.finish(edge1)
    first=FakeWeak(ledger,edge1,0.)
    obj=DualEndpointAudit();obj.directory=d;obj.ledger=ledger
    obj.row=dict(method='adaboost_r',pair_id='synthetic_order',
        epsilon_mar=.1,minimum_correct=2)
    obj.round_results=[SimpleNamespace(round_index=1,empirical_train_margin_loss=.5)]
    obj.written=0;obj.weak_learners=[first];obj.weak_learner_signs=[1.]
    obj.boost_coefs=[1.]
    obj.endpoint_arrays=dict(margin=(np.zeros((4,4)),np.ones(4)),
        classification=(np.zeros((2,4)),np.ones(2)))
    obj.tracker=EndpointTracker(d,ledger)
    obj._maybe_write_checkpoint(1)
    assert set(obj.tracker.hits)=={'classification'}
    class_cost=obj.tracker.hits['classification']['algorithm_raw_shots']
    fit2=ledger.begin('fit',64,25);ledger.finish(fit2)
    edge2=ledger.begin('edge',4,25);ledger.finish(edge2)
    obj.weak_learners.append(FakeWeak(ledger,edge2,1.))
    obj.weak_learner_signs.append(1.);obj.boost_coefs.append(1.)
    obj.round_results.append(SimpleNamespace(round_index=2,empirical_train_margin_loss=.05))
    class_monitor_before=ledger.totals()['monitor']
    try:obj._maybe_write_checkpoint(2)
    except BothReached:pass
    else:raise AssertionError('Second endpoint did not stop trajectory')
    assert obj.tracker.hits['classification']['algorithm_raw_shots']==class_cost
    assert obj.tracker.hits['margin']['algorithm_raw_shots']>class_cost
    assert ledger.totals()['monitor']==class_monitor_before
print('synthetic dual-endpoint accounting PASS')
