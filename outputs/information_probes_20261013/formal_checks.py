"""Post-fit certificates and exact removal of unused constant input slots.

No refitting or test-driven parameter selection. The dual objective is evaluated
in float64; feasibility of rational dual probabilities is checked in integers.
"""
from __future__ import annotations
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'1')
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
from scipy.special import logsumexp,softmax,xlogy
import torch
import run_study as study
from analyze_study import bootstrap

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',required=True,type=Path)
    args=ap.parse_args();run=args.run;torch.set_num_threads(1)
    raw=study.make_raw(run,True);results={};dual_arrays={}
    for branch in study.MAIN:
        with (run/'final'/f'linear_{branch}.pkl').open('rb') as f:model=pickle.load(f)
        x=study.matrix(raw,'full',branch,'full').astype(np.float64);y=raw['full']['y'];n=len(y)
        scores=x@model.coef_.T+model.intercept_;probs=softmax(scores,axis=1)
        onehot=np.eye(10)[y];lam=1/(model.C*n)
        primal=float((logsumexp(scores,axis=1)-scores[np.arange(n),y]).mean()+lam/2*np.square(model.coef_).sum())
        grad=(probs-onehot).T@x/n+lam*model.coef_
        grad_b=(probs-onehot).mean(0)
        # Iterative proportional fitting enforces the unpenalized intercept's
        # dual constraint. Integer rounding below makes feasibility exact.
        q=probs.copy();target=np.bincount(y,minlength=10)
        for iteration in range(100000):
            q*=target/q.sum(0);q/=q.sum(1)[:,None]
            if np.abs(q.sum(0)-target).max()<1e-11:break
        else:raise RuntimeError('Dual balancing did not converge')
        denominator=2**40
        counts=np.floor(q*denominator).astype(np.int64)
        counts[:,9]+=denominator-counts.sum(1)
        wanted=target.astype(np.int64)*denominator
        for c in range(9):
            delta=int(wanted[c]-counts[:,c].sum())
            row=int(np.argmax(counts[:,9] if delta>=0 else counts[:,c]))
            counts[row,c]+=delta;counts[row,9]-=delta
        assert counts.min()>=0 and np.all(counts.sum(1)==denominator)
        np.testing.assert_array_equal(counts.sum(0),wanted)
        feasible=counts/denominator
        b=(feasible-onehot).T@x/n
        dual=float(-xlogy(feasible,feasible).sum()/n-np.square(b).sum()/(2*lam))
        gap=primal-dual
        assert gap>=-1e-10
        results[branch]=dict(C=model.C,n=n,lambda_=lam,primal_nats=primal,dual_nats=dual,
            numerical_duality_gap_nats=gap,gradient_weight_max_abs=float(abs(grad).max()),gradient_intercept_max_abs=float(abs(grad_b).max()),
            balancing_iterations=iteration+1,dual_denominator=denominator,exact_integer_dual_feasibility=True,
            scope='Certificate for the regularized empirical training objective only; arithmetic evaluation in float64, not interval arithmetic. Not a population entropy or MI certificate.')
        dual_arrays[branch+'_counts']=counts
    np.savez_compressed(run/'linear_dual_witnesses.npz',**dual_arrays)
    # All unused input slots are identically zero: delete their incident weights.
    directory=run/'compact_holes';directory.mkdir(exist_ok=True)
    scaling={}
    source=np.column_stack([raw['full']['x'],raw['full']['t'][:,1::2]])
    mean,std=source.mean(0),source.std(0);keep=std>=1e-10
    scaling=dict(mean=mean,std=np.where(keep,std,1),keep=keep)
    np.savez_compressed(directory/'scaler.npz',**scaling)
    records=[]
    for seed in range(10,20):
        path=run/'final'/f'mlp6_xb1_s{seed}.pt'
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        state=checkpoint['state_dict'];new={k:v.clone() for k,v in state.items()};new['0.weight']=new['0.weight'][:,:68].clone()
        model=torch.nn.Sequential(torch.nn.Linear(68,6),torch.nn.ReLU(),torch.nn.Linear(6,10));model.load_state_dict(new);model.eval()
        assert sum(v.numel() for v in model.parameters())==484
        record=dict(seed=seed,parameters=484)
        for part in ['wdep','windep']:
            z=np.column_stack([raw[part]['x'],raw[part]['t'][:,1::2]])
            z=np.where(keep,(z-mean)/scaling['std'],0).astype(np.float32)
            with torch.no_grad():logits=model(torch.from_numpy(z)).numpy()
            with np.load(run/'predictions'/f'mlp6_xb1_s{seed}.npz') as a:old=a[part+'_logits']
            np.testing.assert_allclose(logits,old,rtol=0,atol=5e-6)
            np.testing.assert_array_equal(logits.argmax(1),old.argmax(1))
            record[part]=dict(max_logit_error=float(abs(logits-old).max()),predictions_identical=True,**study.metrics(logits,raw[part]['y']))
        targetfile=directory/f'holes_s{seed}.pt'
        torch.save(dict(state_dict=new,parameters=484,architecture='68->6->10 ReLU',input='X64 plus four beta1 at closing r=0,1,2,3; scaler.npz',
                        source_sha256=study.sha(path),protocol_sha256=study.sha(run/'protocol.json'),seed=seed),targetfile)
        record['sha256']=study.sha(targetfile);records.append(record)
    with np.load(run/'losses.npz') as a:
        comparisons={}
        for competitor in ['xg','xt']:
            comparisons[competitor]={}
            for part in ['wdep','windep']:
                d=a[f'mlp6_{competitor}_{part}_bits']-a[f'mlp6_xb1_{part}_bits']
                acc=100*(a[f'mlp6_xb1_{part}_correct'].astype(float)-a[f'mlp6_{competitor}_{part}_correct'])
                comparisons[competitor][part]=dict(bits=bootstrap(d,raw[part]['y']),accuracy_pp=bootstrap(acc,raw[part]['y']),
                    status='Exploratory contrast inspected after the main run; no new model selection or training.')
    profile={}
    for part in ['tra','cv','wdep','windep']:
        y=raw[part]['y'];t=raw[part]['t']
        profile[part]=dict(beta0_counts={str(int(k)):int(v) for k,v in zip(*np.unique(t[:,0],return_counts=True))},
                          beta1_counts={str(int(k)):int(v) for k,v in zip(*np.unique(t[:,1],return_counts=True))},
                          fraction_holes_by_digit={str(c):float((t[y==c,1]>0).mean()) for c in range(10)})
    result=dict(linear_certificates=results,compact_models=records,holes_comparisons=comparisons,raw_topology_profiles=profile,
                formal_data_processing='T1 is a deterministic projection of T: I(Y;X,T1)<=I(Y;X,T). This theorem does not require estimating MI.',
                code_sha256=study.sha(__file__))
    study.write(run/'formal_checks.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
