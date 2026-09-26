"""Development experiment for a trainable topological layer; TRA/CV only.

Previously observed OptDigits tests are deliberately not read. No new independent
confirmation or novelty claim can be obtained from this development experiment.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import hashlib,json,sys,time,argparse
import numpy as np
import torch
from euler_layer import SmallClassifier
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/learned_topology';WORK=ROOT/'work/learned_topology';JOBS=WORK/'jobs';CACHE=WORK/'train_cv.npz';PROTOCOL=OUT/'pilot_protocol.json'
sys.path.insert(0,str(ROOT/'outputs/real_topology'))
from optdigits_pilot import read_original
KINDS=['euler_learned','euler_frozen','area_learned','perimeter_learned','geometry_learned','mlp']+[f'cnn_{s}c{c}' for s in ['', 'skip_'] for c in [1,2,4]]
A=None;torch.set_num_threads(1)
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,r):
    t=Path(str(p)+'.tmp');t.write_text(json.dumps(r,indent=2)+'\n');t.replace(p)

def prepare():
    if CACHE.exists():return
    arrays={}
    for name,part in [('train','tra'),('validation','cv')]:
        im,y=read_original(part);x=im.reshape(-1,8,4,8,4).mean((2,4)).astype(np.float32)[:,None]
        arrays[name+'_x']=x;arrays[name+'_y']=y
    x=arrays['train_x'].reshape(-1,64);mean=x.mean(0);std=x.std(0);keep=std>=1e-10
    arrays.update(mean=mean,std=np.where(keep,std,1),keep=keep)
    np.savez_compressed(CACHE,**arrays)

def freeze():
    prepare()
    if PROTOCOL.exists():
        r=json.loads(PROTOCOL.read_text())
        for k,v in r['source_sha256'].items():assert sha(OUT/k)==v
        assert sha(CACHE)==r['cache_sha256'];return r
    checks=json.loads((OUT/'layer_checks.json').read_text());assert checks['passed'] and checks['code_sha256']==sha(OUT/'euler_layer.py')
    configs=[dict(id=f'{kind}_o{oi}_s{s}',kind=kind,lr=lr,seed=s,budget=512) for kind in KINDS for oi,lr in enumerate([.003,.01]) for s in range(3)]
    r=dict(registered_utc=now(),scope='Exploratory development only on previously used TRA/CV. WDEP/WINDEP/USPS TEST are already seen in earlier research and will not be relabeled as untouched evidence.',
       input='Identical8x8 block means from original32x32 for EVERY branch. Layer, ordinaryCNN and pixel skip all receive this same tensor; no full-resolution side information.',
       layer='Two learned3x3 filters, four learned threshold levels per field; logistic smoothingT=.1; cubical superlevel face-height extension by max. Output8Euler values inside the autodiff graph; no cached topological features.',
       geometry='Same trainable filter/threshold setup using expected area or perimeter. Combined geometry uses2thresholds per field, giving4area+4perimeter values. BN8 and classifier width5 shared.',
       frozen='Same Euler-layer initialization and identical classifier width;28filter/threshold parameters held fixed, giving441trainable and469storedparameters. Explicit ablation, not an equal-trainable-capacity baseline.',
       cnn='One learned3x3 conv with1/2/4channels, BN/ReLU, adaptive pooling4x4; with or without the same64pixel skip. Largest dense hidden width under512including all biases and BNaffine.',
       initialization='Common identity/blur/sharpen filter initialization plus Gaussian.005jitter for all learnable image filters; independent common seeded model RNG and minibatch permutation RNG.',
       budgets='All stored scalar parameters <=512, frozen Euler also counted in storage; BNrunning-stat buffers and program operations additional. Not a FLOP/time budget.',
       selection='Per-kind maximum mean3seed CVaccuracy, then meanCE, fewer trainableparameters, lexicalID. Record all fits. Strongest control is the best selected non-Euler kind; no TEST-based choice.',
       optimizer='AdamW',learning_rates=[.003,.01],weight_decay=.01,batch_size=256,initial_max_epochs=500,patience=80,
       convergence_rule='Same for every model: if not early-stopped and best CVCE epoch>=450at500, automatically extend cap to1000. Stop after80epochs without CEimprovement>1e-8. Mark unresolved if final best lies within50epochs of hardcap.',
       seed_rule='model seed0/1/2; independent shared permutation generator2026092700+seed.',
       checkpoints='Minimum CVCE, ties keep earliest. All final model states and validation probabilities saved; parameter-change and feature-change diagnostics are taken from selected checkpoints.',
       configs=configs,fits=len(configs),source_sha256={n:sha(OUT/n) for n in ['euler_layer.py','pilot.py','check_layer.py','layer_checks.json']},
       cache_sha256=sha(CACHE),source_data_sha256=sha(ROOT/'work/real_data/optdigits.zip'),test_read=False)
    write(PROTOCOL,r);return r

def init():
    global A
    torch.set_num_threads(1)
    with np.load(CACHE) as z:A={k:torch.from_numpy(z[k].copy()) for k in z.files}

def fit(cfg,digest):
    JOBS.mkdir(exist_ok=True);name=cfg['id'];done=JOBS/(name+'.json')
    if done.exists():
        r=json.loads(done.read_text());assert r['protocol_sha256']==digest;return r
    x,y,v,vy=[A[k] for k in ['train_x','train_y','validation_x','validation_y']]
    torch.manual_seed(cfg['seed']);net=SmallClassifier(cfg['kind'],A['mean'],A['std'],A['keep']);initial={k:p.detach().clone() for k,p in net.named_parameters()}
    raw_initial=None
    if hasattr(net.front,'thresholds'):
        with torch.no_grad():raw_initial=net.front(x[:128]).detach().clone()
    opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=cfg['lr'],weight_decay=.01)
    rng=torch.Generator().manual_seed(2026092700+cfg['seed']);best=float('inf');stale=0;maxepoch=500;epoch=0;curve=[];t=time.perf_counter();first_grad={};extended=False
    while epoch<maxepoch:
        epoch+=1;net.train()
        for j,ix in enumerate(torch.randperm(len(x),generator=rng).split(256)):
            opt.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(net(x[ix]),y[ix]);loss.backward()
            if epoch==1 and j==0:first_grad={k:(None if p.grad is None else float(p.grad.norm())) for k,p in net.named_parameters() if k.startswith('front.')}
            opt.step()
        net.eval()
        with torch.no_grad():logits=net(v);ce=float(torch.nn.functional.cross_entropy(logits,vy));acc=float((logits.argmax(1)==vy).double().mean())
        assert np.isfinite(ce)
        curve.append([ce,acc])
        if ce<best-1e-8:
            best=ce;best_epoch=epoch;stale=0;state={k:p.detach().clone() for k,p in net.state_dict().items()};prob=logits.softmax(1).numpy().copy()
        else:stale+=1
        if epoch%25==0:write(JOBS/(name+'.progress.json'),dict(updated_utc=now(),id=name,epoch=epoch,max_epochs=maxepoch,best_epoch=best_epoch,best_ce=best,seconds=time.perf_counter()-t))
        if stale>=80:break
        if epoch==500 and best_epoch>=450:maxepoch=1000;extended=True
    net.load_state_dict(state);net.eval();changes={k:float((p.detach()-initial[k]).norm()) for k,p in net.named_parameters() if k.startswith('front.')}
    feature_rms=None
    if raw_initial is not None:
        with torch.no_grad():feature_rms=float((net.front(x[:128])-raw_initial).square().mean().sqrt())
    r=dict(**cfg,trainable_parameters=net.trainable_parameters,stored_parameters=net.stored_parameters,hidden=net.hidden,
        accuracy=float((prob.argmax(1)==vy.numpy()).mean()),ce=best,selected_epoch=best_epoch,epochs_run=epoch,extended=extended,
        hardcap_convergence_unresolved=bool(epoch==maxepoch and best_epoch>=maxepoch-50),seconds=time.perf_counter()-t,curve=curve,
        first_batch_front_gradient_norms=first_grad,selected_front_parameter_change_norms=changes,selected_feature_probe_rms_change=feature_rms,protocol_sha256=digest)
    torch.save(dict(state_dict=state,config={k:v for k,v in r.items() if k!='curve'},initial_parameters=initial,protocol_sha256=digest),JOBS/(name+'.pt'))
    np.savez_compressed(JOBS/(name+'.npz'),probabilities=prob,predictions=prob.argmax(1));write(done,r);return r

def summarize(rows,digest):
    from collections import defaultdict
    groups=defaultdict(list)
    for r in rows:groups[r['id'].rsplit('_s',1)[0]].append(r)
    summaries=[]
    for key,rs in groups.items():
        rs.sort(key=lambda r:r['seed']);assert [r['seed'] for r in rs]==[0,1,2];r=rs[0]
        summaries.append(dict(id=key,kind=r['kind'],lr=r['lr'],trainable_parameters=r['trainable_parameters'],stored_parameters=r['stored_parameters'],hidden=r['hidden'],
          mean_accuracy=float(np.mean([v['accuracy'] for v in rs])),mean_ce=float(np.mean([v['ce'] for v in rs])),seed_accuracy=[v['accuracy'] for v in rs],selected_epochs=[v['selected_epoch'] for v in rs],
          unresolved_runs=sum(v['hardcap_convergence_unresolved'] for v in rs),parameter_change=[v['selected_front_parameter_change_norms'] for v in rs],feature_change=[v['selected_feature_probe_rms_change'] for v in rs]))
    rank=lambda r:(-r['mean_accuracy'],r['mean_ce'],r['trainable_parameters'],r['id']);summaries.sort(key=rank)
    selected={k:min([r for r in summaries if r['kind']==k],key=rank) for k in KINDS}
    strongest=min([r for k,r in selected.items() if not k.startswith('euler_')],key=rank)
    result=dict(created_utc=now(),scope='Exploratory TRAIN/CV; no independent-test or significance claim',protocol_sha256=digest,fits=len(rows),selected=selected,strongest_nontopological_control=strongest,all_configurations=summaries,test_read=False)
    write(OUT/'pilot_summary.json',result);print(json.dumps(dict(selected=selected,strongest_control=strongest),indent=2),flush=True)

def main(workers):
    p=freeze();digest=sha(PROTOCOL);rows=[];t=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=init) as pool:
        futures=[pool.submit(fit,c,digest) for c in p['configs']]
        for f in as_completed(futures):
            r=f.result();rows.append(r);write(OUT/'pilot_results.json',dict(completed=len(rows),total=len(p['configs']),protocol_sha256=digest,runs=rows,test_read=False))
            write(OUT/'pilot_status.json',dict(updated_utc=now(),pid=os.getpid(),completed=len(rows),total=len(p['configs']),seconds=time.perf_counter()-t,last_completed=r['id']))
            print(f"{len(rows)}/{len(p['configs'])} {r['id']} {r['accuracy']:.6f} epoch={r['selected_epoch']}/{r['epochs_run']}",flush=True)
    summarize(rows,digest);write(OUT/'pilot_status.json',dict(updated_utc=now(),stage='completed',completed=len(rows),total=len(p['configs'])))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);args=parser.parse_args();main(args.workers)
