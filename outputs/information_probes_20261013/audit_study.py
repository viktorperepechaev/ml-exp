"""Independent normalization, checkpoint replay, selection and chronology audit."""
from __future__ import annotations
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'1')
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
from scipy.special import logsumexp
import torch

def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',type=Path,required=True);ap.add_argument('--repo',type=Path,required=True)
    args=ap.parse_args();run=args.run;repo=args.repo;torch.set_num_threads(1)
    p=load(run/'protocol.json');selection=load(run/'selection.json');ev=load(run/'evaluation.json')
    source=run/'runner_frozen.py'
    assert sha(source if source.exists() else Path(__file__).with_name('run_study.py'))==p['runner_sha256']
    for path,digest in p['source_sha256'].items():assert sha(repo/path)==digest
    assert sha(run/'development.npz')==p['development_sha256']
    receipt=load(run/'weights_receipt.json');testreceipt=load(run/'test_receipt.json')
    assert receipt['frozen_utc']<=testreceipt['parsed_utc']<=ev['completed_utc']
    assert testreceipt['weights_receipt_sha256']==sha(run/'weights_receipt.json')
    assert testreceipt['features_sha256']==sha(run/'tests.npz')
    assert receipt['selection_sha256']==sha(run/'selection.json')
    assert len(receipt['weights'])==158 and selection['fits']==510 and len(ev['rows'])==158
    for path,digest in receipt['weights'].items():assert sha(run/path)==digest
    raw={}
    for file,parts in [('development.npz',['tra','cv']),('tests.npz',['wdep','windep'])]:
        with np.load(run/file) as a:
            for part in parts:raw[part]={k:a[f'{part}_{k}'].copy() for k in ['x','g','t','y']}
    raw['full']={k:np.concatenate([raw['tra'][k],raw['cv'][k]]) for k in ['x','g','t','y']}
    # Standardization is reimplemented, without calling the fitting script.
    cache={}
    def inputs(part,branch,trainpart):
        key=(part,branch,trainpart)
        if key in cache:return cache[key]
        blocks={}
        for k in ['x','g','t']:
            reference=raw[trainpart][k]
            mu=np.sum(reference,axis=0)/len(reference)
            variance=np.sum((reference-mu)**2,axis=0)/len(reference)
            sigma=np.sqrt(variance);active=sigma>=1e-10
            vals=np.zeros_like(raw[part][k],dtype=np.float64)
            vals[:,active]=(raw[part][k][:,active]-mu[active])/sigma[active]
            blocks[k]=vals.astype(np.float32)
        vals=[blocks['x']]
        if branch=='xg':vals.append(blocks['g'])
        if branch=='xt':vals.append(blocks['t'])
        if branch=='xgt':vals.extend([blocks['g'],blocks['t']])
        if branch=='xb0':vals.append(blocks['t'][:,::2])
        if branch=='xb1':vals.append(blocks['t'][:,1::2])
        if branch=='xshuffle':
            idx=['tra','cv','full','wdep','windep'].index(part)
            order=np.random.default_rng(2026101300+idx).permutation(len(raw[part]['y']))
            vals.append(blocks['t'][order])
        result=np.column_stack(vals);result=np.pad(result,((0,0),(0,80-result.shape[1])))
        cache[key]=result;return result
    def make(width):return torch.nn.Sequential(torch.nn.Linear(80,width),torch.nn.ReLU(),torch.nn.Linear(width,10))
    def replay(record,base,part,trainpart):
        z=inputs(part,record['branch'],trainpart)
        if record['family']=='mlp':
            model=make(record['width'])
            checkpoint=torch.load(base.with_suffix('.pt'),map_location='cpu',weights_only=False)
            assert checkpoint['protocol_sha256']==sha(run/'protocol.json')
            model.load_state_dict(checkpoint['state_dict']);model.eval()
            with torch.no_grad():out=model(torch.from_numpy(z)).numpy()
        else:
            with base.with_suffix('.pkl').open('rb') as f:model=pickle.load(f)
            assert np.array_equal(model.classes_,np.arange(10))
            out=model.predict_log_proba(z.astype(np.float64))
        return out
    tune=[];maxerror=0.;exact=0
    for path in sorted((run/'tuning').glob('*.json')):
        r=load(path);base=path.with_suffix('')
        for ext,digest in r['files'].items():assert sha(base.with_suffix(ext))==digest
        logits=replay(r,base,'cv','tra')
        with np.load(base.with_suffix('.npz')) as a:
            expected=a['logits'];np.testing.assert_allclose(logits,expected,rtol=0,atol=1e-6)
            exact+=int(np.array_equal(logits,expected));maxerror=max(maxerror,float(abs(logits-expected).max()))
            if r['family']=='mlp':
                curve=a['curve'];assert r['selected_epoch']==int(curve[:,1].argmin())+1
                assert len(curve)==r['epochs_run']
        y=raw['cv']['y'];bits=float((logsumexp(logits.astype(float),axis=1)-logits[np.arange(len(y)),y])/np.log(2) @ np.full(len(y),1/len(y)))
        assert abs(bits-r['bits'])<1e-7
        tune.append(r)
    # Check every selected configuration against every candidate on CV.
    for key,choice in selection['selected'].items():
        config=choice['config'];candidates=[r for r in tune if r['family']==config['family'] and r['branch']==config['branch'] and r.get('width')==config.get('width')]
        groups={}
        for r in candidates:groups.setdefault((r.get('lr'),r.get('wd'),r.get('C'),r.get('gamma')),[]).append(r)
        best=min(float(np.mean([r['bits'] for r in group])) for group in groups.values())
        assert abs(best-choice['cv_bits'])<1e-12
        chosen=[r for r in candidates if r['id'] in choice['jobs']]
        if config['family']=='mlp':
            assert len(chosen)==3 and config['epochs']==int(np.median([r['selected_epoch'] for r in chosen]))
    for r in ev['rows']:
        base=run/'final'/r['id'];predfile=run/'predictions'/(r['id']+'.npz')
        assert sha(predfile)==r['prediction_sha256']
        with np.load(predfile) as a:
            for part in ['wdep','windep']:
                logits=replay(r,base,part,'full');expected=a[part+'_logits']
                np.testing.assert_allclose(logits,expected,rtol=0,atol=1e-6)
                maxerror=max(maxerror,float(abs(logits-expected).max()));exact+=int(np.array_equal(logits,expected))
                y=raw[part]['y'];bits=np.mean(logsumexp(logits.astype(float),axis=1)-logits[np.arange(len(y)),y])/np.log(2)
                assert abs(bits-r[part]['bits'])<1e-7
                assert float((logits.argmax(1)==y).mean())==r[part]['accuracy']
    with np.load(repo/'runs/wdep-alpha1-20261011/wdep_features.npz') as old:
        for key,oldkey in [('x','pixels'),('g','geometry'),('t','topology')]:np.testing.assert_array_equal(raw['wdep'][key],old['raw_'+oldkey])
        np.testing.assert_array_equal(raw['wdep']['y'],old['test_y'])
    analysis=load(run/'analysis.json') if (run/'analysis.json').exists() else None
    if analysis:
        for key,s in analysis['summaries'].items():
            group=[r for r in ev['rows'] if ((f"mlp{r['width']}" if r['family']=='mlp' else r['family'])+'_'+r['branch'])==key]
            for part in ['wdep','windep']:
                for metric in ['bits','accuracy']:assert abs(s[part][metric]-np.mean([r[part][metric] for r in group]))<1e-12
    result=dict(passed=True,tuning_checkpoints=len(tune),final_checkpoints=len(ev['rows']),prediction_arrays_replayed=len(tune)+2*len(ev['rows']),
        bitwise_exact_arrays=exact,max_logit_error=maxerror,normalization_independently_recomputed=True,
        source_hashes_verified=True,cv_selection_verified=True,test_read_after_all_weights=True,wdep_matches_historical_features=True,
        no_test_selection_in_this_run=True,no_population_mi_or_family_optimality_certification=True,auditor_sha256=sha(__file__))
    (run/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
