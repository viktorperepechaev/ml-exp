"""Independent topology, analytical-gradient and end-to-end-gradient checks."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import hashlib,json,itertools,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage
from euler_layer import smooth_euler,cell_heights,SmallClassifier,EulerCurveLayer
OUT=Path(__file__).resolve().parent
torch.set_num_threads(1)


def independent_chi(m):
    b0=ndimage.label(m,np.ones((3,3)))[1]
    _,nb=ndimage.label(~np.pad(m,1),ndimage.generate_binary_structure(2,1))
    return b0-(nb-1)


def main():
    # Exhaust all 3x3 binary masks: diagonal connectivity, boundary and holes.
    masks=np.array(list(itertools.product([0,1],repeat=9))).reshape(-1,3,3)
    hard=torch.from_numpy(masks[:,None]).double();t=torch.tensor([[.5]],dtype=torch.double)
    got=smooth_euler(hard,t,.001).detach().numpy().ravel();expected=np.array([independent_chi(m.astype(bool)) for m in masks])
    np.testing.assert_allclose(got,expected,atol=1e-12,rtol=0)
    v,eh,ev,f=cell_heights(hard)
    binary=(v>=.5).sum((-1,-2))-(eh>=.5).sum((-1,-2))-(ev>=.5).sum((-1,-2))+(f>=.5).sum((-1,-2))
    np.testing.assert_array_equal(binary.numpy().ravel(),expected)
    # Independent exact integration over intervals between sorted face values:
    # sum chi(K_t) * P(logistic threshold lies in that interval).
    rng=np.random.default_rng(208741);z=rng.normal(size=(4,3));tau=.25;temperature=.31
    points=np.r_[-np.inf,np.sort(z.ravel()),np.inf];weighted=0.
    from scipy.special import expit
    for a,b in zip(points[:-1],points[1:]):
        mid=b-1 if not np.isfinite(a) else (a+1 if not np.isfinite(b) else (a+b)/2)
        weighted+=independent_chi(z>=mid)*(expit((b-tau)/temperature)-expit((a-tau)/temperature))
    observed=float(smooth_euler(torch.tensor(z)[None,None],torch.tensor([[tau]]),temperature))
    assert abs(weighted-observed)<1e-6,(weighted,observed)
    # Random strictly ordered heights avoid undefined derivatives at max ties.
    zz=torch.tensor(rng.normal(size=(1,1,3,4)),requires_grad=True);tt=torch.tensor([[.1,.7]],dtype=torch.double,requires_grad=True)
    assert torch.autograd.gradcheck(lambda a,b:smooth_euler(a,b,.24),(zz,tt),eps=1e-6,atol=2e-5,rtol=2e-4)
    # Gradient must reach the upstream feature convolution and the thresholds.
    torch.manual_seed(45);net=SmallClassifier('euler_learned',torch.zeros(64),torch.ones(64),torch.ones(64,dtype=torch.bool))
    images=torch.rand(32,1,8,8,requires_grad=True);labels=torch.arange(32)%10;loss=torch.nn.functional.cross_entropy(net(images),labels);loss.backward()
    gradients={k:float(v.grad.norm()) for k,v in net.named_parameters() if k.startswith('front.')}
    assert all(np.isfinite(v) and v>1e-8 for v in gradients.values());assert float(images.grad.norm())>1e-8
    counts={}
    kinds=['euler_learned','euler_frozen','area_learned','perimeter_learned','geometry_learned','mlp']+[f'cnn_{s}c{c}' for s in ['', 'skip_'] for c in [1,2,4]]
    for k in kinds:
        m=SmallClassifier(k,torch.zeros(64),torch.ones(64),torch.ones(64,dtype=torch.bool));assert m(images.detach()).shape==(32,10)
        counts[k]=dict(trainable=m.trainable_parameters,stored=m.stored_parameters,hidden=m.hidden)
    # Runtime estimate without changing any hyperparameters based on outcomes.
    images=torch.rand(256,1,8,8);labels=torch.arange(256)%10
    for _ in range(3):net.zero_grad();torch.nn.functional.cross_entropy(net(images),labels).backward()
    start=time.perf_counter()
    for _ in range(20):net.zero_grad();torch.nn.functional.cross_entropy(net(images),labels).backward()
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),passed=True,code_sha256=hashlib.sha256((OUT/'euler_layer.py').read_bytes()).hexdigest(),
        binary_topology_cases=512,hard_binary_euler_values_exact=True,sharp_soft_limit_matches_to_tolerance=True,logistic_expectation_matches_independent_exact_integration=True,finite_difference_gradcheck=True,
        classification_gradient_norms=gradients,input_gradient_nonzero=True,parameter_counts=counts,forward_backward_batch256_seconds=(time.perf_counter()-start)/20,
        test_data_read=False,warning='The soft output is an expectation of Euler characteristic, not an integer Betti number. Differentiability in max-based face extensions is almost everywhere.')
    (OUT/'layer_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
