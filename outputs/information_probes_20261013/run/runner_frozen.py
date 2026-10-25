"""Frozen conditional-probing study of OptDigits, using only local archived data.

Run all stages, or resume completed jobs with the same source and protocol.
No test features are parsed until every selected model has been refitted/frozen.
Both test sets have been inspected in earlier research: this is exploratory.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import time
import warnings
import zipfile
import numpy as np
from scipy.special import logsumexp
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import torch

MAIN = ['x', 'xg', 'xt', 'xgt']
ABLATIONS = ['xb0', 'xb1', 'xshuffle']
RAW = None
OUT = None
PROTOCOL_HASH = None


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def write(p, obj):
    p = Path(p)
    q = p.with_suffix(p.suffix + '.tmp')
    q.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    q.replace(p)
def load(p): return json.loads(Path(p).read_text())


def extract(repo, part):
    with zipfile.ZipFile(repo / 'work/real_data/optdigits.zip') as z:
        packed = z.read(f'optdigits-orig.{part}.Z')
    text = subprocess.run(['gzip', '-dc'], input=packed, capture_output=True, check=True).stdout.decode('ascii')
    lines = [line.strip() for line in text.splitlines()]
    imgs, labels, i = [], [], 0
    while i < len(lines):
        if len(lines[i]) == 32 and set(lines[i]) <= {'0', '1'}:
            rows = lines[i:i+32]
            assert len(rows) == 32 and all(len(s) == 32 and set(s) <= {'0', '1'} for s in rows)
            assert len(lines[i+32]) == 1 and lines[i+32].isdigit()
            imgs.append(np.frombuffer(''.join(rows).encode(), np.uint8).reshape(32, 32) == 49)
            labels.append(int(lines[i+32])); i += 33
        else: i += 1
    images, y = np.stack(imgs), np.array(labels, dtype=np.int64)
    assert len(y) == dict(tra=1934, cv=946, wdep=943, windep=1797)[part]
    sys.path.insert(0, str(repo / 'outputs/real_topology'))
    from optdigits_ablation import scale_features
    t, g = scale_features(images)
    x = images.reshape(-1, 8, 4, 8, 4).mean((2, 4)).reshape(-1, 64)
    return dict(x=x, g=g, t=t, y=y, image_hash=np.array(hashlib.sha256(images.tobytes()).hexdigest()))


def prepare(repo, run):
    sources = ['outputs/real_topology/optdigits_pilot.py', 'outputs/real_topology/optdigits_ablation.py',
               'work/real_data/optdigits.zip', 'pyproject.toml', 'uv.lock']
    if (run/'protocol.json').exists():
        p = load(run/'protocol.json')
        assert p['runner_sha256'] == sha(__file__), 'Runner changed; use a new run directory.'
        for s, digest in p['source_sha256'].items(): assert sha(repo/s) == digest
        assert sha(run/'development.npz') == p['development_sha256']
        return p
    assert not run.exists() or not any(run.iterdir()), 'Refusing nonempty unregistered directory'
    run.mkdir(parents=True, exist_ok=True)
    for folder in ['tuning', 'final', 'predictions']: (run/folder).mkdir()
    dev = {}
    for part in ['tra', 'cv']:
        dev.update({f'{part}_{k}': v for k, v in extract(repo, part).items()})
    # Exact comparison with pre-existing feature cache; no new representation.
    with np.load(repo/'work/real_data/optdigits_complement_trainval_v1.npz') as old:
        for part, pref in [('tra', 'tr'), ('cv', 'va')]:
            for k, oldk in [('x', 'pixels'), ('g', 'global_geometry'), ('t', 'global_topology')]:
                np.testing.assert_array_equal(dev[f'{part}_{k}'], old[f'{pref}_{oldk}'])
    np.savez_compressed(run/'development.npz', **dev)
    p = dict(created_utc=now(), question='Does topology supply usable class information beyond geometry, and is its benefit specific to tiny decoders?',
        design='Exploratory frozen follow-up. WDEP and WINDEP were previously used. Not a fresh confirmatory holdout.',
        primary='MLP width6 on WINDEP: xg minus xt logloss; xg minus xgt logloss. Positive means topology helps.',
        secondary='WDEP; widths32/128; linear/RBF; beta0/beta1/shuffled topology ablations at width6; width interaction.',
        train='TRA1934; tuning on CV946; fixed refit TRA+CV2880; WDEP943/WINDEP1797 only after weights frozen.',
        features='Original 32x32 binary input: X64 block means, G8 area/perimeter and T8 beta0/beta1 for independent disk closings r0,1,2,3. Side scale1.',
        branches=MAIN, ablations=ABLATIONS, widths=[6,32,128],
        inputs='All decoders receive 80 slots: X then G or T in same8 slots, zeros to80; XGT uses G then T. XB0/XB1 have4 features then zeros. Zero weights are nominal parameters but not effective capacity.',
        normalization='Training-partition-only mean/std(ddof0); std<1e-10 -> zero; refit on TRA+CV.',
        shuffle='T rows permuted independently within each partition, RNG2026101300+part index(tra0,cv1,full2,wdep3,windep4); fixed across seeds; never condition on labels.',
        mlp=dict(rates=[.0003,.001,.003,.01,.03], weight_decays=[0.,.01], tune_seeds=[0,1,2], final_seeds=list(range(10,20)),
                 max_epochs=3000, patience=120, batch_size=256, optimizer='AdamW', activation='ReLU',
                 selection='Minimum mean3seed CV cross-entropy; ties LR then WD. Per-run lowest CV CE checkpoint; refit epochs = integer median3 selected epochs.',
                 initialization='torch seed; independent minibatch RNG202610130+seed; same across branches.'),
        linear=dict(Cs=[.001,.01,.1,1.,10.,100.], solver='lbfgs', max_iter=4000, tol=1e-9),
        rbf=dict(Cs=[1.,10.,100.], gammas=[.001,.01,.1], probability=True, random_state=20261013),
        classical_selection='Minimum CV logloss; tie by job id. Refit once on TRA+CV. SVC probability calibration uses internal training-only folds.',
        uncertainty='5000 paired crossed seed/image bootstrap replicates, images stratified by digit; 95% descriptive intervals. Primary2 contrasts also97.5% marginal intervals (Bonferroni2). No writer-cluster adjustment: no writer IDs. No confirmatory p-value.',
        code_and_data_prior_exposure='Protocol fixed before this run, not preregistration before data exposure. Prior results influenced choice of features and architecture.',
        population_shannon_mi='Not identified by finite-sample probe losses. No claim that trained MLPs achieve family/global infima.',
        runner_sha256=sha(__file__), source_sha256={s:sha(repo/s) for s in sources}, development_sha256=sha(run/'development.npz'),
        versions=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__, sklearn=sklearn.__version__))
    write(run/'protocol.json', p)
    return p


def make_raw(run, final=False):
    with np.load(run/'development.npz') as a:
        raw = {part:{k:a[f'{part}_{k}'].copy() for k in ['x','g','t','y']} for part in ['tra','cv']}
    raw['full'] = {k:np.concatenate([raw['tra'][k],raw['cv'][k]]) for k in ['x','g','t','y']}
    if final and (run/'tests.npz').exists():
        with np.load(run/'tests.npz') as a:
            for part in ['wdep','windep']: raw[part] = {k:a[f'{part}_{k}'].copy() for k in ['x','g','t','y']}
    return raw


def matrix(raw, part, branch, trainpart):
    def norm(k):
        a, b = raw[trainpart][k], raw[part][k]
        mean, std = a.mean(0), a.std(0)
        keep = std >= 1e-10
        return np.where(keep, (b-mean)/np.where(keep,std,1), 0).astype(np.float32)
    x,g,t = norm('x'),norm('g'),norm('t')
    z = np.zeros((len(x),80), np.float32); z[:,:64] = x
    if branch == 'xg': z[:,64:72] = g
    elif branch == 'xt': z[:,64:72] = t
    elif branch == 'xgt': z[:,64:72],z[:,72:80] = g,t
    elif branch in ['xb0','xb1']: z[:,64:68] = t[:,0 if branch=='xb0' else 1::2]
    elif branch == 'xshuffle':
        idx = dict(tra=0,cv=1,full=2,wdep=3,windep=4)[part]
        z[:,64:72] = t[np.random.default_rng(2026101300+idx).permutation(len(t))]
    else: assert branch == 'x'
    return z


def net(width): return torch.nn.Sequential(torch.nn.Linear(80,width),torch.nn.ReLU(),torch.nn.Linear(width,10))
def metrics(logits,y):
    logits = np.asarray(logits,dtype=np.float64)
    bits = (logsumexp(logits,axis=1)-logits[np.arange(len(y)),y])/np.log(2)
    return dict(bits=float(bits.mean()), accuracy=float((logits.argmax(1)==y).mean()))


def init_worker(run, final=False):
    global RAW, OUT, PROTOCOL_HASH
    torch.set_num_threads(1)
    OUT = Path(run); RAW = make_raw(OUT,final); PROTOCOL_HASH=sha(OUT/'protocol.json')


def fit_job(job):
    started=time.perf_counter()
    final=job['phase']=='final'
    folder=OUT/('final' if final else 'tuning'); base=folder/job['id']
    if base.with_suffix('.json').exists():
        result=load(base.with_suffix('.json'))
        assert result['protocol_sha256']==PROTOCOL_HASH
        for ext,digest in result['files'].items(): assert sha(base.with_suffix(ext))==digest
        return result
    trainpart='full' if final else 'tra'
    x=matrix(RAW,trainpart,job['branch'],trainpart); y=RAW[trainpart]['y']
    v=matrix(RAW,'cv',job['branch'],trainpart); vy=RAW['cv']['y']
    files={}; curve=[]; caution=[]
    if job['family']=='mlp':
        x,y,v,vy=map(torch.from_numpy,[x,y,v,vy])
        torch.manual_seed(job['seed']); model=net(job['width'])
        initial=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
        opt=torch.optim.AdamW(model.parameters(),lr=job['lr'],weight_decay=job['wd'])
        rng=torch.Generator().manual_seed(202610130+job['seed'])
        best=float('inf'); stale=0; chosen=None
        limit=job['epochs'] if final else 3000
        for epoch in range(1,limit+1):
            model.train(); total=0.
            for ids in torch.randperm(len(y),generator=rng).split(256):
                opt.zero_grad(set_to_none=True)
                loss=torch.nn.functional.cross_entropy(model(x[ids]),y[ids]); assert torch.isfinite(loss)
                loss.backward(); opt.step(); total+=float(loss.detach())*len(ids)
            if final: curve.append([total/len(y)])
            else:
                model.eval()
                with torch.no_grad():
                    logits=model(v); ce=float(torch.nn.functional.cross_entropy(logits,vy))
                    acc=float((logits.argmax(1)==vy).double().mean())
                curve.append([total/len(y),ce,acc])
                if ce < best-1e-8:
                    best=ce; chosen=epoch; stale=0
                    state={k:w.detach().clone() for k,w in model.state_dict().items()}; bestlogits=logits.numpy().copy()
                else: stale+=1
                if stale>=120: break
        if final: state=model.state_dict(); chosen=epoch
        torch.save(dict(state_dict=state,job=job,protocol_sha256=PROTOCOL_HASH),base.with_suffix('.pt'))
        files['.pt']=sha(base.with_suffix('.pt'))
        result=dict(**job, selected_epoch=chosen, epochs_run=epoch, censored=not final and stale<120,
                    nominal_parameters=sum(p.numel() for p in model.parameters()),
                    effective_parameters=({'x':64,'xg':72,'xt':72,'xgt':80,'xb0':68,'xb1':68,'xshuffle':72}[job['branch']]+11)*job['width']+10,
                    initial_sha256=initial)
        if not final:
            np.savez_compressed(base.with_suffix('.npz'),logits=bestlogits,curve=np.asarray(curve))
            result.update(metrics(bestlogits,RAW['cv']['y']))
        else: np.savez_compressed(base.with_suffix('.npz'),curve=np.asarray(curve))
        files['.npz']=sha(base.with_suffix('.npz'))
    else:
        if job['family']=='linear':
            model=LogisticRegression(C=job['C'],solver='lbfgs',max_iter=4000,tol=1e-9)
        else: model=SVC(C=job['C'],gamma=job['gamma'],probability=True,random_state=20261013)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always'); model.fit(x.astype(np.float64),y)
            caution=[str(w.message) for w in ws]
        with base.with_suffix('.pkl').open('wb') as f: pickle.dump(model,f)
        files['.pkl']=sha(base.with_suffix('.pkl'))
        result=dict(**job,warnings=caution,n_iter=np.asarray(model.n_iter_).tolist())
        if job['family']=='rbf': result['support_vectors']=int(len(model.support_))
        if not final:
            logits=model.predict_log_proba(v.astype(np.float64))
            np.savez_compressed(base.with_suffix('.npz'),logits=logits)
            files['.npz']=sha(base.with_suffix('.npz')); result.update(metrics(logits,vy))
    result.update(protocol_sha256=PROTOCOL_HASH,files=files,seconds=time.perf_counter()-started)
    write(base.with_suffix('.json'),result)
    return result


def parallel(jobs,run,workers):
    rows=[]; started=time.perf_counter(); last=started
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=init_worker,initargs=(str(run),)) as pool:
        futures=[pool.submit(fit_job,j) for j in jobs]
        for future in as_completed(futures):
            rows.append(future.result())
            if time.perf_counter()-last>25 or len(rows)==len(jobs):
                print(json.dumps(dict(phase=jobs[0]['phase'],done=len(rows),total=len(jobs),seconds=round(time.perf_counter()-started,1))),flush=True)
                last=time.perf_counter()
    return rows


def tune(run,p,workers):
    jobs=[]
    for width in p['widths']:
        for branch in MAIN+(ABLATIONS if width==6 else []):
            for li,lr in enumerate(p['mlp']['rates']):
                for wi,wd in enumerate(p['mlp']['weight_decays']):
                    for seed in p['mlp']['tune_seeds']:
                        jobs.append(dict(id=f'mlp_h{width}_{branch}_l{li}_w{wi}_s{seed}',phase='tuning',family='mlp',width=width,branch=branch,lr=lr,wd=wd,seed=seed))
    for branch in MAIN:
        for ci,c in enumerate(p['linear']['Cs']): jobs.append(dict(id=f'linear_{branch}_c{ci}',phase='tuning',family='linear',branch=branch,C=c))
        for ci,c in enumerate(p['rbf']['Cs']):
            for gi,gamma in enumerate(p['rbf']['gammas']): jobs.append(dict(id=f'rbf_{branch}_c{ci}_g{gi}',phase='tuning',family='rbf',branch=branch,C=c,gamma=gamma))
    rows=parallel(jobs,run,workers)
    groups={}
    for r in rows:
        family=f"mlp{r['width']}" if r['family']=='mlp' else r['family']
        key=f"{family}_{r['branch']}"
        hkey=(r.get('lr'),r.get('wd'),r.get('C'),r.get('gamma'))
        groups.setdefault(key,{}).setdefault(hkey,[]).append(r)
    selected={}
    for key,options in groups.items():
        group=min(options.values(),key=lambda group:(np.mean([r['bits'] for r in group]),group[0].get('lr',0),group[0].get('wd',0),group[0]['id']))
        group=sorted(group,key=lambda r:r.get('seed',0)); first=group[0]
        chosen={k:first[k] for k in ['family','width','branch','lr','wd','C','gamma'] if k in first}
        if first['family']=='mlp': chosen['epochs']=int(np.median([r['selected_epoch'] for r in group]))
        selected[key]=dict(config=chosen,cv_bits=float(np.mean([r['bits'] for r in group])),cv_accuracy=float(np.mean([r['accuracy'] for r in group])),jobs=[r['id'] for r in group])
    result=dict(selected=selected,fits=len(rows),censored=[r['id'] for r in rows if r.get('censored')],warnings={r['id']:r['warnings'] for r in rows if r.get('warnings')},protocol_sha256=sha(run/'protocol.json'))
    write(run/'selection.json',result)
    print(json.dumps(dict(selection_complete=True,fits=len(rows),censored=len(result['censored']),warnings=len(result['warnings']))),flush=True)
    return result


def refit(run,p,selection,workers):
    jobs=[]
    for key,value in sorted(selection['selected'].items()):
        config=value['config']
        for seed in p['mlp']['final_seeds'] if config['family']=='mlp' else [None]:
            job=dict(**config,id=key+(f'_s{seed}' if seed is not None else ''),phase='final')
            if seed is not None:job['seed']=seed
            jobs.append(job)
    rows=parallel(jobs,run,workers)
    weights={str(path.relative_to(run)):sha(path) for path in sorted((run/'final').iterdir()) if path.suffix in ['.pt','.pkl']}
    if (run/'weights_receipt.json').exists():
        old=load(run/'weights_receipt.json')
        assert old['weights']==weights and old['selection_sha256']==sha(run/'selection.json')
    else:
        write(run/'weights_receipt.json',dict(frozen_utc=now(),models=len(rows),weights=weights,selection_sha256=sha(run/'selection.json'),protocol_sha256=sha(run/'protocol.json'),test_parsed_in_this_run=False))
    return rows


def evaluate(repo,run):
    receipt=load(run/'weights_receipt.json')
    assert receipt['selection_sha256']==sha(run/'selection.json')
    for path,digest in receipt['weights'].items():assert sha(run/path)==digest
    if not (run/'tests.npz').exists():
        arrays={}
        for part in ['wdep','windep']: arrays.update({f'{part}_{k}':v for k,v in extract(repo,part).items()})
        np.savez_compressed(run/'tests.npz',**arrays)
        write(run/'test_receipt.json',dict(parsed_utc=now(),features_sha256=sha(run/'tests.npz'),weights_receipt_sha256=sha(run/'weights_receipt.json')))
    assert load(run/'test_receipt.json')['features_sha256']==sha(run/'tests.npz')
    raw=make_raw(run,True);torch.set_num_threads(1)
    rows=[]
    for record in sorted((run/'final').glob('*.json')):
        r=load(record);base=record.with_suffix('');arr={}
        for ext,digest in r['files'].items(): assert sha(base.with_suffix(ext))==digest
        if r['family']=='mlp':
            model=net(r['width']);checkpoint=torch.load(base.with_suffix('.pt'),map_location='cpu',weights_only=False)
            model.load_state_dict(checkpoint['state_dict']);model.eval()
        else:
            with base.with_suffix('.pkl').open('rb') as f:model=pickle.load(f)
        for part in ['wdep','windep']:
            x=matrix(raw,part,r['branch'],'full')
            if r['family']=='mlp':
                with torch.no_grad(): logits=model(torch.from_numpy(x)).numpy()
            else:logits=model.predict_log_proba(x.astype(np.float64))
            arr[part+'_logits']=logits
            r[part]=metrics(logits,raw[part]['y'])
        target=run/'predictions'/(r['id']+'.npz');np.savez_compressed(target,**arr)
        r['prediction_sha256']=sha(target);rows.append(r)
    write(run/'evaluation.json',dict(rows=rows,completed_utc=now(),protocol_sha256=sha(run/'protocol.json')))
    print(json.dumps(dict(evaluation_complete=True,models=len(rows))),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--stage',choices=['all','prepare','tune','refit','evaluate'],default='all')
    args=ap.parse_args();repo=args.repo.resolve();run=args.run.resolve()
    p=prepare(repo,run)
    if args.stage=='prepare':return
    if args.stage in ['all','tune']: selection=tune(run,p,args.workers)
    else:selection=load(run/'selection.json')
    if args.stage in ['all','refit']:refit(run,p,selection,args.workers)
    if args.stage in ['all','evaluate']:evaluate(repo,run)


if __name__=='__main__':main()
