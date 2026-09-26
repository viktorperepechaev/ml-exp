"""Independent integrity checks; never opens the confirmation TEST member."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,itertools,json,sys
import numpy as np
import torch
import tiny_confirmation as experiment
from scipy import ndimage
ROOT=experiment.ROOT;OUT=experiment.OUT;WORK=experiment.WORK

def run(require_complete=False):
    checks={}
    for a in [[],[1]*5,[1,-1],[10],[2,1,-3,2],[10,-3,0,2,-1]]:
        got=experiment.exact_signflip(a)
        weights=np.abs(np.asarray(a,dtype=int));observed=sum(a)
        expected=np.mean([sum(s*w for s,w in zip(signs,weights))>=observed for signs in itertools.product([-1,1],repeat=len(a))])
        assert abs(got-expected)<1e-12,(a,got,expected)
    checks['exact_weighted_sign_flip_matches_enumeration']=True
    # Independent holes = bounded 4-connected background components, with exterior padding.
    rng=np.random.default_rng(51621)
    masks=np.zeros((16,32,32),bool)
    masks[1,4:20,4:20]=True
    masks[2,4:20,4:20]=True;masks[2,8:16,8:16]=False
    masks[3,2,2]=True;masks[3,3,3]=True
    masks[4:]=rng.random((12,32,32))<np.linspace(.1,.9,12)[:,None,None]
    actual,_=experiment.scale_features(masks)
    for ri,radius in enumerate([0,1,2,3]):
        disk=(np.mgrid[-radius:radius+1,-radius:radius+1]**2).sum(0)<=radius**2
        transformed=masks if radius==0 else np.stack([ndimage.binary_closing(np.pad(m,radius),structure=disk)[radius:-radius,radius:-radius] for m in masks])
        for i,m in enumerate(transformed):
            bg,n=ndimage.label(~np.pad(m,1),structure=ndimage.generate_binary_structure(2,1))
            holes=n-1
            assert actual[i,2*ri+1]==holes,(i,radius,actual[i,2*ri+1],holes)
            components=ndimage.label(m,np.ones((3,3)))[1]
            assert actual[i,2*ri]==components
    assert actual[0,0]==0 and actual[1,1]==0 and actual[2,1]==1 and actual[3,0]==1
    checks['betti_features_match_independent_background_hole_count']=True
    labels=experiment.read_original('cv')[1]
    total=0
    for family,path,folder,expected_n in [('mlp',OUT/'tiny_mlp_validation.json',WORK,456),('cnn',OUT/'tiny_cnn_validation.json',experiment.cnn.JOBS,288)]:
        obj=json.loads(path.read_text());rows=obj['runs']
        if require_complete:assert len(rows)==expected_n
        for r in rows:
            with np.load(folder/(r['id']+'.npz')) as z:
                prob=z['probabilities'];pred=z['predictions']
            assert prob.shape==(946,10) and np.isfinite(prob).all()
            assert (prob>=0).all();np.testing.assert_allclose(prob.sum(1),1,atol=2e-7,rtol=2e-7)
            np.testing.assert_array_equal(prob.argmax(1),pred)
            assert abs((pred==labels).mean()-r['accuracy'])<1e-12
            ce=-np.log(prob[np.arange(946),labels].astype(np.float64)).mean()
            assert abs(ce-r['ce'])<3e-6,(r['id'],ce,r['ce'])
            total+=1
        checks[family+'_prediction_files_verified']=len(rows)
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),passed=True,test_read=False,complete_required=require_complete,checks=checks,verified_prediction_files=total)
    experiment.write(OUT/('tiny_complete_development_audit.json' if require_complete else 'tiny_preflight_audit.json'),result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':run('--complete' in sys.argv)
