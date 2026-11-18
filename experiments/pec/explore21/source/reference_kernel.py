"""Identical historical forward values; only backward multiplier differs."""
from common import *
import jax
import jax.numpy as jnp
import numpy as np
import optax
from boosting_qml.pqc.density_matrix import exact_noisy_reuploading_density_score
from boosting_qml.pqc.measurement import _apply_measurement_gradient_mode
from boosting_qml.losses.surrogates import PsiG

CONVENTIONS=('historical_clip','mean_jacobian_same_forward')

def mean_one(params,x):
    return exact_noisy_reuploading_density_score(x,n_qubits=4,params=params,n_layers=3,noise_strength_1q=0.,noise_strength_2q=0.,apply_two_qubit_noise=True,observable='global_z')
def means(params,x):return jax.vmap(lambda v:mean_one(params,v))(x)

def historical_score_from_count(mean,count,n_shots):
    mean=jnp.clip(mean,-1.,1.)
    raw=2.*count.astype(mean.dtype)/float(n_shots)-1.
    return jnp.clip(_apply_measurement_gradient_mode(raw,mean,gradient_mode='straight_through_expectation'),-1.,1.)

def draw_counts(mu,keys,n_shots):
    clipped=jnp.clip(mu,-1.,1.)
    probability=jnp.clip((1.+clipped)/2.,0.,1.)
    outcomes=jax.vmap(lambda k,p:jax.random.bernoulli(k,p,shape=(n_shots,)))(keys,probability)
    return jnp.sum(outcomes,axis=-1)

def scores_and_multipliers(mu,counts,n_shots):
    return jax.vmap(jax.value_and_grad(lambda m,k:historical_score_from_count(m,k,n_shots),argnums=0))(mu,counts)

def loss_gradient_given_counts(theta,x,y,counts,n_shots,convention):
    """Controlled derivative test hook; supplied counts perform no sampling."""
    if convention not in CONVENTIONS:raise ValueError(convention)
    mu,pullback=jax.vjp(lambda t:means(t,x),theta)
    scores,historical_multiplier=scores_and_multipliers(mu,counts,n_shots)
    loss,cotangent=jax.value_and_grad(lambda s:jnp.mean(PsiG(y*s,.5)))(scores)
    multiplier=historical_multiplier if convention=='historical_clip' else jnp.ones_like(mu)
    g=pullback(cotangent*multiplier)[0]
    return loss,g,scores,mu,historical_multiplier

def optimizer():return optax.chain(optax.clip_by_global_norm(1.),optax.adam(.01,b1=.9,b2=.999,eps=1e-8,eps_root=0.))

def make_step(n_shots,convention):
    if convention not in CONVENTIONS:raise ValueError(convention)
    opt=optimizer()
    @jax.jit
    def step(theta,state,x,y,keys):
        mu,pullback=jax.vjp(lambda t:means(t,x),theta)
        counts=draw_counts(mu,keys,n_shots)
        scores,hm=scores_and_multipliers(mu,counts,n_shots)
        loss,cotangent=jax.value_and_grad(lambda s:jnp.mean(PsiG(y*s,.5)))(scores)
        g=pullback(cotangent*(hm if convention=='historical_clip' else jnp.ones_like(mu)))[0]
        delta,state=opt.update(g,state,theta);updated=optax.apply_updates(theta,delta)
        info=dict(loss=loss,boundary_minus=jnp.sum(counts==0),boundary_plus=jnp.sum(counts==n_shots),ties=jnp.sum(2*counts==n_shots),executed_score_ties=jnp.sum(scores==0),mean_outside=jnp.sum(jnp.abs(mu)>1),mean_at_boundary=jnp.sum(jnp.abs(mu)==1),historical_multiplier_nonunit=jnp.sum(hm!=1),historical_multiplier_half=jnp.sum(hm==.5),gradient_norm=optax.global_norm(g),gradient_clipped=optax.global_norm(g)>1.,adam_increment_norm=optax.global_norm(delta),gradient_finite=jnp.all(jnp.isfinite(g)),parameter_finite=jnp.all(jnp.isfinite(updated)))
        return updated,state,info
    return step

def make_evaluator(n_shots=100):
    @jax.jit
    def evaluate(theta,x,y,keys):
        mu=means(theta,x);counts=draw_counts(mu,keys,n_shots)
        scores,_=scores_and_multipliers(mu,counts,n_shots)
        ramp=lambda m:jnp.mean(jnp.where(m>=.1,0.,jnp.where(m>=0,1-m/.1,1.)))
        return scores,dict(correct=jnp.sum(jnp.where(scores>=0,1,-1)==y),boundary_minus=jnp.sum(counts==0),boundary_plus=jnp.sum(counts==n_shots),ties=jnp.sum(2*counts==n_shots),executed_score_ties=jnp.sum(scores==0),mean_outside=jnp.sum(jnp.abs(mu)>1),mean_at_boundary=jnp.sum(jnp.abs(mu)==1),finite=jnp.all(jnp.isfinite(mu)),sampled_ramp_mu0p1=ramp(y*scores),mean_ramp_mu0p1=ramp(y*mu))
    return evaluate

def array_hash(a):
    v=np.ascontiguousarray(np.asarray(a));return canonical(dict(dtype=str(v.dtype),shape=list(v.shape),bytes_sha256=hashlib.sha256(v.tobytes()).hexdigest()))
def stream_keys(root_seed,event,n):return jax.random.split(jax.random.fold_in(jax.random.PRNGKey(root_seed),event),n)
def host(value):return jax.tree_util.tree_map(lambda v:np.asarray(v).item() if np.asarray(v).ndim==0 else np.asarray(v).tolist(),value)
