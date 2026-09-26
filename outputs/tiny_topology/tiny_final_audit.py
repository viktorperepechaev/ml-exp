"""Post-analysis integrity audit; does not tune any model or alter primary data."""
import json
from datetime import datetime,timezone
import numpy as np
from scipy.stats import binom
import tiny_confirmation as e


def exact_by_grouped_binomials(a):
    a=np.asarray(a,dtype=int);weights=np.abs(a[a!=0]);distribution=np.array([1.]);total=int(weights.sum())
    for w in range(1,11):
        n=int((weights==w).sum())
        if not n:continue
        factor=np.zeros(n*w+1);factor[::w]=binom.pmf(np.arange(n+1),n,.5)
        distribution=np.convolve(distribution,factor)
    assert abs(distribution.sum()-1)<1e-11
    threshold=(int(a.sum())+total+1)//2
    return float(distribution[max(0,threshold):].sum())


def main():
    p=e.verify();r=json.loads((e.OUT/'tiny_confirmation_test_results.json').read_text());images,y,member=e.read_wdep()
    with np.load(e.OUT/'tiny_confirmation_predictions.npz') as z:
        t=z['topology_predictions']==y;b=z['baseline_predictions']==y;diff=t.astype(int)-b.astype(int)
        for key,cfg in p['pipelines'].items():
            prob=z[key+'_probabilities'];pred=z[key+'_predictions']
            assert prob.shape==(10,943,10) and np.isfinite(prob).all();np.testing.assert_allclose(prob.sum(2),1,atol=3e-7,rtol=3e-7);np.testing.assert_array_equal(prob.argmax(2),pred)
        for key,value in [('topology_mean_accuracy',t.mean()),('baseline_mean_accuracy',b.mean()),('mean_accuracy_difference',diff.mean())]:assert abs(value-r['primary'][key])<1e-12
    independent_p=exact_by_grouped_binomials(diff.sum(0));assert abs(independent_p-r['primary']['one_sided_paired_image_permutation_p'])<1e-12
    assert p['pipelines']['topology']['parameters']<=512 and p['pipelines']['baseline']['parameters']<=512
    train_images=np.concatenate([e.read_original('tra')[0],e.read_original('cv')[0]])
    train_hashes={np.packbits(a).tobytes() for a in train_images};test_hashes=[np.packbits(a).tobytes() for a in images]
    cross_duplicates=sum(h in train_hashes for h in test_hashes);within_duplicates=len(test_hashes)-len(set(test_hashes))
    intervals=np.array(r['primary']['crossed_seed_image_bootstrap95'])
    expected_success=bool(diff.mean()>0 and min(1.,6*independent_p)<=.05 and intervals[0]>0 and (diff.mean(1)>0).sum()>=8)
    assert expected_success==r['success_gate_passed']
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),passed=True,protocol_sha256=e.sha(e.PROTO),probabilities_and_accuracy_recomputed=True,
      independent_exact_p_via_grouped_binomial_convolution=independent_p,primary_gate_recomputed=True,
      train_test_identical_bitmap_count=cross_duplicates,within_test_duplicate_bitmap_count=within_duplicates,
      label_counts={str(k):int((y==k).sum()) for k in range(10)},test_n=len(y),registered_models_verified=80,
      limitation='Exact bitmap duplicate audit only; handwriting authors overlap by design. No independent-writer confidence claim.')
    e.write(e.OUT/'tiny_confirmation_final_audit.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
