"""One boosting trajectory records independent margin and classification first hits."""
from weak_audit import *
from endpoints import EndpointTracker, algorithm_shots, classification_metrics, endpoint_seed
from boosting_qml.losses.margins import normalized_empirical_ramp_loss

class BothReached(Exception): pass

class DualEndpointAudit:
    def evaluate_prefix(self, endpoint, prefix, role):
        x,y=self.endpoint_arrays[endpoint]
        self.ledger.ensure(prefix*len(x)*25)
        scores=np.zeros(len(x),np.float64);ids=[]
        for i,(weak,sign,coef) in enumerate(zip(self.weak_learners[:prefix],
                self.weak_learner_signs[:prefix],self.boost_coefs[:prefix]),1):
            root=endpoint_seed('boost',self.row['pair_id'],self.row['method'],endpoint,prefix,role,i)
            weak.audit_endpoint=endpoint
            with weak.audit_prediction_phase(role):
                values=weak.predict_scores_jax(x,np.random.default_rng(root))
            weak.audit_endpoint=None
            scores+=float(coef)*float(sign)*np.clip(np.asarray(values,np.float64),-1,1)
            eid=f'{role}:{self.ledger.counter:07d}';ids.append(eid)
            self.ledger.reuse(eid,f'{endpoint} sampled score and metric')
        alpha=sum(abs(float(a)) for a in self.boost_coefs[:prefix])
        ramp=normalized_empirical_ramp_loss(y,scores,alpha,.1)
        out=dict(endpoint=endpoint,prefix=prefix,role=role,sampled_normalized_ramp=ramp,
                 acquisition_ids=ids,executed_score_ties=int(np.sum(scores==0)))
        if endpoint=='margin':
            out.update(target=self.row['epsilon_mar'],passes=ramp<=self.row['epsilon_mar'])
        else:out.update(classification_metrics(scores,y,self.row['minimum_correct']))
        return out

    def _maybe_write_checkpoint(self,round_idx,*,reason='round_completed',force=False):
        self.last_stop_reason=reason
        for rr in self.round_results[self.written:]:
            prefix=int(rr.round_index)
            assert prefix==self.written+1==len(self.round_results)
            edge_ids=[]
            for earlier in self.weak_learners[:prefix]:
                assert hasattr(earlier,'edge_id')
                edge_ids.append(earlier.edge_id)
                self.ledger.reuse(earlier.edge_id,f'cached training margin monitor at prefix {prefix}')
            checkpoint=self.directory/f'weak_{prefix:04d}.npz'
            with checkpoint.open('xb') as stream:
                np.savez(stream,theta=np.asarray(self.weak_learners[prefix-1].params_))
            fields=['round_index','edge_t','boost_coef_t','h_t_star',
                    'empirical_train_margin_loss','empirical_train_error',
                    'deterministic_train_margin_loss','ensemble_patience_rounds_since_improvement',
                    'weak_learner_epochs_run','weak_learner_converged','rng_stream_seeds']
            record={k:scalar_record(getattr(rr,k)) for k in fields if hasattr(rr,k)}
            model=dict(prefix=prefix,
                       weak_checkpoints=[str(self.directory/f'weak_{i:04d}.npz')
                                         for i in range(1,prefix+1)],
                       weak_signs=[float(v) for v in self.weak_learner_signs[:prefix]],
                       boost_coefs=[float(v) for v in self.boost_coefs[:prefix]])
            new_json(self.directory/f'round_{prefix:04d}.json',
                     dict(round=record,model=model,algorithm_raw_shots=algorithm_shots(self.ledger),
                          cumulative_acquisition_ledger=self.ledger.receipt(),utc=now()))
            self.written+=1
            cached_ramp=float(rr.empirical_train_margin_loss)
            def margin_evaluate(role):
                if role=='monitor':
                    return dict(endpoint='margin',prefix=prefix,role=role,
                                sampled_normalized_ramp=cached_ramp,target=self.row['epsilon_mar'],
                                passes=cached_ramp<=self.row['epsilon_mar'],
                                reused_edge_acquisition_ids=edge_ids,newly_acquired_raw_shots=0)
                return self.evaluate_prefix('margin',prefix,'confirm')
            margin=self.tracker.check('margin',dict(round=prefix),model,margin_evaluate)
            classification=self.tracker.check('classification',dict(round=prefix),model,
                        lambda role:self.evaluate_prefix('classification',prefix,role))
            print(json.dumps(dict(round=prefix,method=self.row['method'],
                margin_confirmed=margin['confirmed'],
                classification_confirmed=classification['confirmed'],
                margin_monitor=cached_ramp,
                classification_monitor_accuracy=classification['monitor']['accuracy'],
                algorithm_raw_shots=algorithm_shots(self.ledger),utc=now())),flush=True)
            if len(self.tracker.hits)==2:raise BothReached()

class DualAda(DualEndpointAudit,AdaBoostR):pass
class DualReal(DualEndpointAudit,RealAdaBoost):pass

def run_boost(row,directory,ledger,initial,orders):
    x,y=load_split(row,'train_split');vx,vy=load_split(row,'validation_split')
    cls=DualAda if row['method']=='adaboost_r' else DualReal
    booster=cls(n_rounds=500,margin_threshold=.1,clip_edge=1e-12,
                surrogate_sigma=.5,verbose=False,train_margin_confirmation_policy='none',
                ensemble_patience_rounds=20,ensemble_patience_min_delta=1e-5)
    booster.ledger=ledger;booster.directory=directory;booster.row=row
    booster.written=0;booster.last_stop_reason=None
    booster.endpoint_arrays=dict(margin=(x,y),classification=(vx,vy))
    booster.tracker=EndpointTracker(directory,ledger);all_learners=[]
    def factory():
        weak=AuditedWeak(n_qubits=4,n_layers=row['depth'],n_shots_eff=25,n_epochs=500,
                         batch_size=64,scan_training=True,noise_backend='density_matrix',
                         noise_strength_1q=row['noise_strength_1q'],
                         noise_strength_2q=row['noise_strength_2q'],
                         compute_fit_diagnostic=True,updates_per_epoch=24)
        weak.round_index=len(all_learners)+1
        weak.paired_initial=initial;weak.paired_orders=orders;weak.pair_id=row['pair_id']
        weak.directory=directory;weak.ledger=ledger;weak.audit_role='diagnostic'
        weak.audit_endpoint=None;weak.current_round=lambda:len(all_learners)
        all_learners.append(weak);return weak
    capped=None
    try:
        booster.fit(BinaryDataset(x,y),factory,rng_seed=row['boost_seed'],
                    validation_data=None,test_data=None,stop_target=None)
    except BothReached:pass
    except BudgetReached as exc:capped=str(exc)
    if not capped:
        native=sum(float(w.shot_accounting_summary()['total_raw_shots']) for w in all_learners)
        assert native==sum(ledger.totals().values()),(native,ledger.totals())
    return dict(status='BOTH_TARGETS_CONFIRMED' if len(booster.tracker.hits)==2 else
                'ONLINE_BUDGET_PARTIAL_OR_NONHIT' if capped else 'BOOSTING_PARTIAL_OR_NONHIT',
                first_hits=booster.tracker.hits,latest_checks=booster.tracker.latest,
                accepted_rounds=len(booster.round_results),attempted_rounds=len(all_learners),
                budget_reason=capped,native_stop_reason=booster.last_stop_reason,
                algorithm_raw_shots=algorithm_shots(ledger),**ledger.receipt())
