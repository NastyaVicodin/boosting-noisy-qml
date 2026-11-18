"""Check both depths and preserve the historical gradient with supplied scores."""
from common import *
import importlib.util
import numpy as np
import jax
import jax.numpy as jnp
from pec_kernel import protocol,ideal_means,update
from pec_measurements import _metrics_for_depth
from reference_kernel import optimizer
from boosting_qml.pec.explicit_gatewise import ExplicitPECProtocol,_ideal_scores

OLD=ROOT/'artifacts/strict_v5_patience100_v1_20261019T140521Z/source/pec_kernel.py'

def run():
    spec=importlib.util.spec_from_file_location('historical_pec_kernel_for_control',OLD)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    rows=jsonl(OUT/'manifest/runs.jsonl')
    checks=[]
    for depth in (2,3):
        row=next(r for r in rows if r['depth']==depth)
        theta=jnp.asarray(np.load(row['initial_path'],allow_pickle=False))
        with np.load(row['train_split'],allow_pickle=False) as split:
            x=jnp.asarray(split['X'][:2],dtype=jnp.float32)
            y=jnp.asarray(split['y'][:2],dtype=jnp.float32)
        scores=.2*y
        n=row['N_fit'];p=row['noise_strength_1q']
        q=protocol(n,depth,p,p)
        assert q.raw_samples_per_expectation==n and q.n_layers==depth
        assert q.gate_counts==(16*depth+12,4*depth)
        direct=ExplicitPECProtocol(n_qubits=4,n_layers=depth,noise_strength_1q=0.,noise_strength_2q=0.,raw_samples_per_expectation=1)
        actual=np.asarray(ideal_means(theta,x,depth));expected=np.asarray(_ideal_scores(theta,x,protocol=direct,observable='global_z'))
        assert actual.shape==(2,) and np.allclose(actual,expected,rtol=0,atol=1e-6)
        state=optimizer().init(theta)
        updated,_,info=update(theta,state,x,y,scores,depth)
        jax.block_until_ready(updated)
        assert np.all(np.isfinite(np.asarray(updated))) and bool(np.asarray(info['gradient_finite']))
        metrics=_metrics_for_depth(depth)(theta,x,y,scores)
        assert bool(np.asarray(metrics['finite']))
        if depth==3:
            old_updated,_,old_info=old.update(theta,state,x,y,scores)
            jax.block_until_ready(old_updated)
            assert np.allclose(np.asarray(updated),np.asarray(old_updated),rtol=0,atol=1e-6)
            assert np.allclose(np.asarray(info['loss']),np.asarray(old_info['loss']),rtol=0,atol=1e-6)
        checks.append(dict(depth=depth,noise=p,shots=n,ideal_means_match_direct=True,gradient_finite=True,
                           historical_depth3_update_match=(depth==3),new_measurement_samples=0))
    new_json(OUT/'validation/deterministic_checks.json',dict(utc=now(),status='PASS',checks=checks,
             historical_kernel=str(OLD),historical_kernel_sha256=digest(OLD),new_measurement_samples=0,test_access=False))
    print(json.dumps(dict(status='PASS',depths=[2,3],new_measurement_samples=0)))

if __name__=='__main__':run()
