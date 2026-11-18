from reference_kernel import *
from functools import lru_cache
from boosting_qml.pec.explicit_gatewise import ExplicitPECProtocol,explicit_pec_predict_batch_grouped,_ideal_scores

@lru_cache(None)
def protocol(n):
    assert n in (120,180)
    return ExplicitPECProtocol(n_qubits=4,n_layers=3,noise_strength_1q=.01,noise_strength_2q=.01,raw_samples_per_expectation=n,chunk_size=256)

def predict(theta,x,root,stream,n):
    return explicit_pec_predict_batch_grouped(theta,x,protocol=protocol(n),base_seed=root,stream=stream,propagation_batch_size=128,observable='global_z')

@jax.jit
def ideal_means(theta,x):return _ideal_scores(theta,x,protocol=protocol(180),observable='global_z')

@jax.jit
def loss_and_gradient(theta,x,y,scores):
    _,back=jax.vjp(lambda p:ideal_means(p,x),theta)
    loss,cotangent=jax.value_and_grad(lambda s:jnp.mean(PsiG(y*s,.5)))(scores)
    return loss,back(cotangent)[0]

@jax.jit
def update(theta,state,x,y,scores):
    loss,g=loss_and_gradient(theta,x,y,scores);delta,state=optimizer().update(g,state,theta);theta=optax.apply_updates(theta,delta)
    return theta,state,dict(loss=loss,gradient_norm=optax.global_norm(g),parameter_finite=jnp.all(jnp.isfinite(theta)),gradient_finite=jnp.all(jnp.isfinite(g)),sampled_outside_unit=jnp.sum(jnp.abs(scores)>1))
