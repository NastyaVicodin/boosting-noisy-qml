"""Frozen boosting/weak training with per-acquisition audit hooks and no test access."""
from audit import *
import contextlib,time
import jax.numpy as jnp
from boosting_qml.weak_learners.pqc_binary import WeightedPQCBinaryWeakLearner
from boosting_qml.boosting.adaboost_r import AdaBoostR
from boosting_qml.boosting.real_adaboost import RealAdaBoost
from boosting_qml.config import BoostingStopTargetConfig
from boosting_qml.types import BinaryDataset

class AuditedWeak(WeightedPQCBinaryWeakLearner):
    def _initialize_params(self,key):
        value=jnp.asarray(self.paired_initial) if self.round_index==1 else super()._initialize_params(key)
        append_json(self.directory/'weak_initializations.jsonl',dict(round=self.round_index,array_sha256=array_hash(value),paired_reference=self.round_index==1))
        return value
    def audit_permutation(self,epoch,n):
        order=self.paired_orders[epoch-1] if self.round_index==1 else np.random.default_rng(seed('weak_permutation',self.pair_id,self.round_index,epoch)).permutation(n)
        assert len(order)==n
        return order
    def audit_before_fit(self,epoch,params,orders,key):
        self.fit_id=self.ledger.begin('fit',int(orders.size),25,round=self.round_index,epoch=epoch,theta_sha256=array_hash(params),permutation_sha256=array_hash(orders),rng_key=np.asarray(key).tolist(),optimizer_updates=24)
    def audit_after_fit(self,epoch,params,losses):
        jax.block_until_ready(params);losses=np.asarray(losses)
        self.ledger.finish(self.fit_id,post_theta_sha256=array_hash(params),epoch_loss=float(losses[0]))
        if not np.all(np.isfinite(params)) or not np.all(np.isfinite(losses)):raise FloatingPointError('Nonfinite weak training')
        self.ledger.reuse(self.fit_id,'weighted epoch loss, weak patience and best-epoch selection')
        self.ledger.oracle('exact_noisy_mean_vjp',1536,round=self.round_index,epoch=epoch)
        append_json(self.directory/'epochs.jsonl',dict(round=self.round_index,epoch=epoch,epoch_loss=float(losses[0]),online_shots=self.ledger.online(),utc=now()))
    def audit_before_fit_diagnostic(self,n,key):
        self.diag_id=self.ledger.begin('diagnostic',n,25,round=self.round_index,purpose='restored weak full-training surrogate diagnostic',rng_key=np.asarray(key).tolist(),theta_sha256=array_hash(self.params_))
    def audit_after_fit_diagnostic(self,loss):
        self.ledger.finish(self.diag_id,weighted_surrogate_loss=loss)
        self.ledger.oracle('exact_noisy_means_for_sampling',1536,acquisition_id=self.diag_id)
    @contextlib.contextmanager
    def audit_prediction_phase(self,role):
        old=self.audit_role;self.audit_role=role
        try:
            with self.prediction_phase('train'):yield
        finally:self.audit_role=old
    def predict_scores_jax(self,x,rng=None):
        eid=self.ledger.begin(self.audit_role,len(x),25,endpoint=getattr(self,'audit_endpoint',None),round_acquired=self.current_round(),learner_round=self.round_index,theta_sha256=array_hash(self.params_),rng_state_before=None if rng is None else rng.bit_generator.state)
        result=super().predict_scores_jax(x,rng=rng);jax.block_until_ready(result)
        self.ledger.finish(eid);self.ledger.oracle('exact_noisy_means_for_sampling',len(x),acquisition_id=eid)
        if not np.all(np.isfinite(result)):raise FloatingPointError('Nonfinite prediction')
        if self.audit_role=='edge':
            self.edge_id=eid;self.ledger.reuse(eid,'edge, weak sign/coefficient, distribution update and cached ensemble margin monitor')
        return result
    def predict_deterministic_mean_scores_jax(self,x):
        result=super().predict_deterministic_mean_scores_jax(x);jax.block_until_ready(result)
        self.ledger.oracle('exact_noisy_mean_for_ensemble_patience_and_selection',len(x),learner_round=self.round_index)
        return result

