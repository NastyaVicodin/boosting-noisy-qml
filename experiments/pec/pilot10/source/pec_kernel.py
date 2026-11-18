"""Explicit PEC samples with the historical exact ideal-mean VJP."""
from reference_kernel import *
from functools import lru_cache
from boosting_qml.pec.explicit_gatewise import ExplicitPECProtocol,explicit_pec_predict_batch_grouped,_ideal_scores

@lru_cache(None)
def protocol(n,depth,p1,p2):
    assert (int(depth),int(n),float(p1),float(p2)) in ((3,60,.005,.005),(2,100,.01,.01))
    return ExplicitPECProtocol(n_qubits=4,n_layers=int(depth),noise_strength_1q=float(p1),noise_strength_2q=float(p2),raw_samples_per_expectation=int(n),chunk_size=256)

def predict(theta,x,root,stream,n,depth,p1,p2):
    return explicit_pec_predict_batch_grouped(theta,x,protocol=protocol(n,depth,p1,p2),base_seed=root,stream=stream,propagation_batch_size=128,observable='global_z')

@lru_cache(None)
def _ideal_for_depth(depth):
    depth=int(depth);assert depth in (2,3)
    ideal_protocol=ExplicitPECProtocol(n_qubits=4,n_layers=depth,noise_strength_1q=0.,noise_strength_2q=0.,raw_samples_per_expectation=1,chunk_size=256)
    @jax.jit
    def evaluate(theta,x):return _ideal_scores(theta,x,protocol=ideal_protocol,observable='global_z')
    return evaluate

def ideal_means(theta,x,depth):return _ideal_for_depth(depth)(theta,x)

@lru_cache(None)
def _update_for_depth(depth):
    mean_fn=_ideal_for_depth(depth)
    @jax.jit
    def one_update(theta,state,x,y,scores):
        _,back=jax.vjp(lambda p:mean_fn(p,x),theta)
        loss,cotangent=jax.value_and_grad(lambda s:jnp.mean(PsiG(y*s,.5)))(scores)
        g=back(cotangent)[0]
        delta,state=optimizer().update(g,state,theta);theta=optax.apply_updates(theta,delta)
        return theta,state,dict(loss=loss,gradient_norm=optax.global_norm(g),parameter_finite=jnp.all(jnp.isfinite(theta)),gradient_finite=jnp.all(jnp.isfinite(g)),sampled_outside_unit=jnp.sum(jnp.abs(scores)>1))
    return one_update

def update(theta,state,x,y,scores,depth):return _update_for_depth(depth)(theta,state,x,y,scores)
