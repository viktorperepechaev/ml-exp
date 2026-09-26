"""TRAIN-only descriptor audit and separately timed equivalent implementations.

This reports the current CPU implementation, not an embedded-device benchmark.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,time,platform,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from scipy import ndimage
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/tiny_topology'
sys.path.insert(0,str(ROOT/'outputs/real_topology'))
from optdigits_pilot import read_original
from optdigits_ablation import scale_features

def descriptors(images,kind):
    features=[]
    for radius in [0,1,2,3]:
        disk=(np.mgrid[-radius:radius+1,-radius:radius+1]**2).sum(0)<=radius*radius
        masks=images if radius==0 else np.stack([ndimage.binary_closing(np.pad(im,radius),structure=disk)[radius:-radius,radius:-radius] for im in images])
        faces=masks.sum((1,2))
        if kind=='geometry':
            perimeter=4*faces-2*((masks[:,:-1]&masks[:,1:]).sum((1,2))+(masks[:,:,:-1]&masks[:,:,1:]).sum((1,2)))
            features.extend([faces,perimeter])
        else:
            p=np.pad(masks,((0,0),(1,1),(1,1)))
            h=(p[:,:-1,1:-1]|p[:,1:,1:-1]).sum((1,2))
            v=(p[:,1:-1,:-1]|p[:,1:-1,1:]).sum((1,2))
            vertices=(p[:,:-1,:-1]|p[:,1:,:-1]|p[:,:-1,1:]|p[:,1:,1:]).sum((1,2))
            components=np.array([ndimage.label(im,np.ones((3,3)))[1] for im in masks])
            features.extend([components,components-(vertices-h-v+faces)])
    return np.stack(features,1)


def main():
    images,labels=read_original('tra');top,geo=scale_features(images)
    for kind,target in [('topology',top),('geometry',geo)]:np.testing.assert_array_equal(descriptors(images,kind),target)
    sample=images[:256];times={k:[] for k in ['topology','geometry']}
    for k in times:descriptors(sample,k)
    for i in range(12):
        for k in (['topology','geometry'] if i%2==0 else ['geometry','topology']):
            start=time.perf_counter();descriptors(sample,k);times[k].append(time.perf_counter()-start)
    r=dict(created_utc=datetime.now(timezone.utc).isoformat(),test_read=False,data='Official TRA1934 only',descriptor_equivalence_verified=True,
        topology_present=dict(any_raw_hole_fraction=float((top[:,1]>0).mean()),multiple_components_fraction=float((top[:,0]>1).mean()),
            betti0_values_counts=dict(zip(*[a.tolist() for a in np.unique(top[:,0].astype(int),return_counts=True)])),
            betti1_values_counts=dict(zip(*[a.tolist() for a in np.unique(top[:,1].astype(int),return_counts=True)])),
            class_hole_fraction={str(k):float((top[labels==k,1]>0).mean()) for k in range(10)}),
        timing=dict(batch_size=256,alternating_repeats=12,milliseconds_per_image={k:float(np.median(v)*1000/256) for k,v in times.items()},raw_batch_seconds=times,
            platform=platform.platform(),processor=platform.machine(),concurrent_load='Other CPU training workers running. Batched throughput includes four mask profiles; single-query latency and embedded energy are unmeasured.',
            topology_speedup_claim=False),
        memory='A 508parameter float32 network has2032bytes of numerical weights. Current stored checkpoints also include normalization/statistics and metadata. A fixed descriptor extractor, working arrays and code are additional memory; they are not represented by the weight count.')
    (OUT/'tiny_feature_cost_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2),flush=True)
if __name__=='__main__':main()
