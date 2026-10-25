"""Prespecified descriptive analysis of the frozen conditional-probing run."""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/topological-invariants-matplotlib')
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
BOOTSTRAPS={}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,o):Path(p).write_text(json.dumps(o,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def load(p):return json.loads(Path(p).read_text())

def bootstrap(difference,labels,replicates=5000):
    # Paired images and paired initialization seeds. No independence of writers claimed.
    s,n=difference.shape
    key=(s,n,replicates,labels.tobytes())
    if key not in BOOTSTRAPS:
        rng=np.random.default_rng(20261013)
        sw=rng.multinomial(s,np.full(s,1/s),size=replicates)/s
        iw=np.zeros((replicates,n),np.float64)
        for k in range(10):
            ids=np.flatnonzero(labels==k)
            iw[:,ids]=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=replicates)/n
        BOOTSTRAPS[key]=(sw,iw)
    sw,iw=BOOTSTRAPS[key]
    means=np.einsum('bi,bi->b',sw@difference,iw)
    return dict(mean=float(difference.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),
                ci975=np.quantile(means,[.0125,.9875]).tolist(),positive_seeds=int((difference.mean(1)>0).sum()),
                seeds=s,seed_means=difference.mean(1).tolist(),
                seed_sd=float(difference.mean(1).std(ddof=1)) if s>1 else None,
                caveat='Paired crossed bootstrap, digit-stratified images; descriptive, no writer clustering or adaptive test-reuse correction.')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',required=True,type=Path)
    args=ap.parse_args();run=args.run
    ev=load(run/'evaluation.json');sel=load(run/'selection.json');p=load(run/'protocol.json')
    with np.load(run/'tests.npz') as a:labels={part:a[part+'_y'].copy() for part in ['wdep','windep']}
    rows=ev['rows'];groups={};arrays={};summaries={}
    for r in rows:
        key=(f"mlp{r['width']}" if r['family']=='mlp' else r['family'])+'_'+r['branch']
        groups.setdefault(key,[]).append(r)
    for key,group in groups.items():
        group.sort(key=lambda r:r.get('seed',0));summaries[key]={}
        for part,y in labels.items():
            logits=[]
            for row in group:
                path=run/'predictions'/(row['id']+'.npz');assert sha(path)==row['prediction_sha256']
                with np.load(path) as a:logits.append(a[part+'_logits'].astype(np.float64))
            logits=np.stack(logits);assert np.isfinite(logits).all()
            bits=(logsumexp(logits,axis=2)-np.take_along_axis(logits,y[None,:,None],axis=2)[:,:,0])/np.log(2)
            correct=(logits.argmax(2)==y)
            arrays[f'{key}_{part}_bits']=bits;arrays[f'{key}_{part}_correct']=correct
            summaries[key][part]=dict(n=len(y),models=len(group),bits=float(bits.mean()),accuracy=float(correct.mean()),
                 seed_bits=bits.mean(1).tolist(),seed_accuracy=correct.mean(1).tolist(),
                 accuracy_seed_sd=float(correct.mean(1).std(ddof=1)) if len(group)>1 else None,
                 class_bits={str(c):float(bits[:,y==c].mean()) for c in range(10)})
        summaries[key]['selection']=sel['selected'][key]
        if 'nominal_parameters' in group[0]:
            summaries[key]['nominal_parameters']=group[0]['nominal_parameters']
            summaries[key]['effective_parameters']=group[0]['effective_parameters']
    np.savez_compressed(run/'losses.npz',**arrays,**{f'{k}_labels':v for k,v in labels.items()})
    contrasts={}
    comparison_pairs=[('topology_vs_geometry','xg','xt'),('topology_given_geometry','xg','xgt'),
                      ('geometry_given_topology','xt','xgt'),('topology_given_pixels','x','xt'),('geometry_given_pixels','x','xg')]
    for family in ['mlp6','mlp32','mlp128','linear','rbf']:
        for name,left,right in comparison_pairs:
            key=f'{family}_{name}';contrasts[key]={}
            for part,y in labels.items():
                diff=arrays[f'{family}_{left}_{part}_bits']-arrays[f'{family}_{right}_{part}_bits']
                acc=100*(arrays[f'{family}_{right}_{part}_correct'].astype(float)-arrays[f'{family}_{left}_{part}_correct'])
                contrasts[key][part]=dict(bits=bootstrap(diff,y),accuracy_pp=bootstrap(acc,y),
                        class_bits={str(c):float(diff[:,y==c].mean()) for c in range(10)})
    for branch in ['xb0','xb1','xshuffle']:
        key='mlp6_'+branch+'_vs_pixels';contrasts[key]={}
        for part,y in labels.items():
            diff=arrays[f'mlp6_x_{part}_bits']-arrays[f'mlp6_{branch}_{part}_bits']
            acc=100*(arrays[f'mlp6_{branch}_{part}_correct'].astype(float)-arrays[f'mlp6_x_{part}_correct'])
            contrasts[key][part]=dict(bits=bootstrap(diff,y),accuracy_pp=bootstrap(acc,y))
    interaction={}
    for part,y in labels.items():
        d6=arrays[f'mlp6_xg_{part}_bits']-arrays[f'mlp6_xt_{part}_bits']
        d128=arrays[f'mlp128_xg_{part}_bits']-arrays[f'mlp128_xt_{part}_bits']
        interaction[part]=bootstrap(d6-d128,y)
    # Finite empirical distribution diagnostic, not an estimate of population MI.
    with np.load(run/'development.npz') as d,np.load(run/'tests.npz') as t:
        xs=np.concatenate([d['tra_x'],d['cv_x'],t['wdep_x'],t['windep_x']])
        empirical=dict(n=len(xs),distinct_pooled_images=len(np.unique(xs,axis=0)))
        raw_top=np.concatenate([d['tra_t'],d['cv_t'],t['wdep_t'],t['windep_t']])
        empirical['distinct_betti_profiles']=len(np.unique(raw_top,axis=0))
        prior=np.bincount(np.concatenate([d['tra_y'],d['cv_y']]),minlength=10)/2880
    null={part:float(-np.log2(prior[y]).mean()) for part,y in labels.items()}
    result=dict(summaries=summaries,contrasts=contrasts,width_interaction=interaction,empirical_diagnostics=empirical,
        train_prior_bits=null,models=len(rows),tuning_fits=sel['fits'],censored=sel['censored'],warnings=sel['warnings'],
        bootstrap_replicates=5000,primary_contrasts=['mlp6_topology_vs_geometry/windep','mlp6_topology_given_geometry/windep'],
        primary_interval='97.5% marginal, Bonferroni2; exploratory descriptive intervals with stated dependence limits.',
        protocol_sha256=sha(run/'protocol.json'),evaluation_sha256=sha(run/'evaluation.json'),analysis_code_sha256=sha(__file__))
    write(run/'analysis.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(x='#777777',xg='#d97824',xt='#176bb0',xgt='#23874b')
    names=dict(x='Pixels',xg='Pixels + geometry',xt='Pixels + Betti',xgt='Pixels + both')
    fig,axs=plt.subplots(1,2,figsize=(10,4),sharey=True,layout='constrained')
    for ax,part in zip(axs,['wdep','windep']):
        for branch in ['x','xg','xt','xgt']:
            values=[summaries[f'mlp{w}_{branch}'][part]['bits'] for w in [6,32,128]]
            ax.plot([6,32,128],values,'o-',label=names[branch],color=colors[branch])
        ax.set_xscale('log');ax.set_xticks([6,32,128],['6','32','128']);ax.set_xlabel('Hidden width')
        ax.set_title(part.upper());ax.grid(alpha=.2)
    axs[0].set_ylabel('Mean label log-loss (bits)');axs[1].legend(fontsize=8)
    fig.suptitle('OptDigits: fixed features, separately CV-selected decoders\nMean of 10 individual fits; no ensemble',fontsize=11)
    fig.savefig(run/'capacity.png',dpi=180);fig.savefig(run/'capacity.svg');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4),sharey=True,layout='constrained')
    branches=['x','xshuffle','xg','xb0','xb1','xt','xgt']
    labels_display=['Pixels','Shuffled Betti','Geometry','Components only','Holes only','Betti','Geometry + Betti']
    for ax,part in zip(axs,['wdep','windep']):
        values=[summaries[f'mlp6_{b}'][part]['bits'] for b in branches]
        ax.barh(np.arange(len(branches)),values,color=['#999999','#999999','#d97824','#66a2cf','#66a2cf','#176bb0','#23874b'])
        ax.set_yticks(np.arange(len(branches)),labels_display);ax.set_title(part.upper());ax.set_xlabel('Mean label log-loss (bits)')
        ax.grid(axis='x',alpha=.2)
    axs[0].invert_yaxis()
    fig.suptitle('What supplies the gain? Width-6 MLP ablations',fontsize=11)
    fig.savefig(run/'ablations.png',dpi=180);fig.savefig(run/'ablations.svg');plt.close(fig)
    print(json.dumps(dict(summaries=summaries,width_interaction=interaction,empirical_diagnostics=empirical),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
