"""Strong compact CNN controls/candidates at the preselected512weight budget.

TRAIN/CV only; all branches see original32x32 binary OptDigits images. No TEST.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import multiprocessing as mp
import argparse,hashlib,json,sys,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/tiny_topology';WORK=ROOT/'work/tiny_topology'
sys.path.insert(0,str(ROOT/'outputs/real_topology'))
from optdigits_pilot import read_original
from optdigits_ablation import scale_features

torch.set_num_threads(1)
PROTOCOL=OUT/'tiny_cnn_protocol.json';RESULT=OUT/'tiny_cnn_validation.json';CACHE=WORK/'cnn_bank.npz';JOBS=WORK/'cnn_jobs';A=None


def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 t=Path(str(p)+'.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)


def parameter_count(family,c,h,m):
 if family=='hidden':return 18*c*c+12*c+(32*c+11)*h+10+10*m
 if family=='linear_head':return 18*c*c+332*c+10+10*m
 if family=='depthwise':return c*c+181*c+10+10*m
 raise ValueError(family)


class CompactCNN(torch.nn.Module):
 def __init__(self,family,c,h,m):
  super().__init__();self.family=family
  if family=='depthwise':
   self.body=torch.nn.Sequential(torch.nn.Conv2d(1,c,3,padding=1),torch.nn.ReLU(),torch.nn.MaxPool2d(2),
      torch.nn.Conv2d(c,c,3,padding=1,groups=c),torch.nn.ReLU(),torch.nn.Conv2d(c,c,1),torch.nn.ReLU(),torch.nn.MaxPool2d(2),torch.nn.AdaptiveAvgPool2d(4),torch.nn.Flatten())
   d=16*c
  else:
   layers=[torch.nn.Conv2d(1,c,3,padding=1),torch.nn.ReLU(),torch.nn.MaxPool2d(2),torch.nn.Conv2d(c,2*c,3,padding=1),torch.nn.ReLU(),torch.nn.MaxPool2d(2),torch.nn.AdaptiveAvgPool2d(4),torch.nn.Flatten()]
   if family=='hidden':layers += [torch.nn.Linear(32*c,h),torch.nn.ReLU()]
   self.body=torch.nn.Sequential(*layers);d=h if family=='hidden' else 32*c
  self.head=torch.nn.Linear(d+m,10)
 def forward(self,x,f):return self.head(torch.cat([self.body(x),f],1))


def catalogue():
 cfgs=[];excluded=[]
 for family in ['hidden','linear_head','depthwise']:
  for c in ([1,2,4] if family=='hidden' else [1,2,3]):
   for branch in ['raw','geometry','topology']:
    m=0 if branch=='raw' else 8
    h=(512-(18*c*c+12*c+10+10*m))//(32*c+11) if family=='hidden' else 0
    count=parameter_count(family,c,h,m)
    if (family=='hidden' and h<1) or count>512:
     excluded.append(dict(family=family,channels=c,branch=branch,reason='No feasible positive hidden width' if family=='hidden' else 'Exceeds512parameters'));continue
    for aug in ['none','bank5']:
     for oi,lr in enumerate([.001,.003,.01]):
      for seed in [0,1,2]:
       cfgs.append(dict(id=f'{family}_c{c}_{branch}_{aug}_o{oi}_s{seed}',family=family,channels=c,hidden_width=h,branch=branch,feature_dim=m,
         parameters=count,budget=512,augmentation=aug,optimizer_index=oi,lr=lr,weight_decay=1e-4,seed=seed))
 return cfgs,excluded


def freeze():
 configs,excluded=catalogue()
 for cfg in configs:
  net=CompactCNN(cfg['family'],cfg['channels'],cfg['hidden_width'],cfg['feature_dim']);assert sum(p.numel() for p in net.parameters())==cfg['parameters']<=512
  with torch.no_grad():z=net(torch.zeros((2,1,32,32)),torch.zeros((2,cfg['feature_dim'])))
  assert z.shape==(2,10)
 if PROTOCOL.exists():
  p=json.loads(PROTOCOL.read_text());assert p['code_sha256']==sha(__file__);return p
 p=dict(registered_utc=now(),scope='Development after tinyMLP pilot, before any unused wdep TEST access. Check whether full-input compact CNNs erase the apparent MLP topology advantage.',
  budget=512,configs=configs,infeasible_architectures=excluded,fits=len(configs),test_read=False,
  input='Original32x32 binary image as float32, no input normalization. All branches see the identical full image.',
  architecture='hidden: two3x3Conv/ReLU/MaxPool2 stages, AdaptiveAvgPool4, densehidden/ReLU, linear10; linear_head omits hidden dense; depthwise uses first3x3Conv/ReLU/pool, depthwise3x3/ReLU, pointwise1x1/ReLU/pool, AdaptiveAvgPool4, linear10.',
  side='8topology or matched8area/perimeter features on the exact CNN input at padded disk closing radii0/1/2/3. Side concatenated before final linear classifier; all80side weights counted.',
  augmentation='Identity or uniform5bank: identity/right1/left1/down1/up1, integer zero-filled shifts. Recompute side features on every shifted image, no data-label augmentation asymmetry. Validation identity.',
  feature_scaling='Per-column mean/std fitted to all internalTRAIN banks sampled by that augmentation; columns with std<1e-10 zeroed; branch weight0.3.',
  optimizer='AdamW',lr_grid=[.001,.003,.01],weight_decay=1e-4,batch_size=256,max_epochs=250,patience=30,
  checkpoint='Minimum CVcrossentropy, improvement>1e-8; earliest tie;30epoch patience, all branches identical.',
  seed_rule='Initialization seed0/1/2; independent common batch permutation and bank RNG seed2026092900+seed.',
  selection='Mean3seed CVaccuracy, then meanCE, fewer trainable parameters, lexicalid; all configurations and predictions retained.',
  preflight='All feasible catalogue entries checked against actual PyTorch parameter count and forward shape before fitting.',
  code_sha256=sha(__file__),data_sha256=sha(ROOT/'work/real_data/optdigits.zip'),
  dependencies_sha256={n:sha(ROOT/'outputs/real_topology'/n) for n in ['optdigits_pilot.py','optdigits_ablation.py']})
 write(PROTOCOL,p);return p


def prepare(p):
 if CACHE.exists():
  meta=json.loads((WORK/'cnn_bank_meta.json').read_text());assert meta['protocol_sha256']==sha(PROTOCOL) and meta['cache_sha256']==sha(CACHE);return
 arrays={}
 for part,partition in [('train','tra'),('validation','cv')]:
  im,y=read_original(partition)
  banks=np.zeros((5 if part=='train' else 1,*im.shape),bool);banks[0]=im
  if part=='train':
   banks[1,:,:,1:]=im[:,:,:-1];banks[2,:,:,:-1]=im[:,:,1:];banks[3,:,1:,:]=im[:,:-1,:];banks[4,:,:-1,:]=im[:,1:,:]
  top=[];geo=[]
  for b in banks:
   t,g=scale_features(b);top.append(t);geo.append(g)
  arrays[part+'_images']=banks.astype(np.float32);arrays[part+'_topology']=np.stack(top).astype(np.float32);arrays[part+'_geometry']=np.stack(geo).astype(np.float32);arrays[part+'_labels']=y
 np.savez_compressed(CACHE,**arrays);write(WORK/'cnn_bank_meta.json',dict(protocol_sha256=sha(PROTOCOL),cache_sha256=sha(CACHE),test_read=False))


def initialize():
 global A
 torch.set_num_threads(1)
 with np.load(CACHE) as z:A={k:z[k].copy() for k in z.files}


def fit(cfg,digest):
 JOBS.mkdir(exist_ok=True);done=JOBS/(cfg['id']+'.json')
 if done.exists():
  old=json.loads(done.read_text());assert old['protocol_sha256']==digest;return old
 banks=1 if cfg['augmentation']=='none' else 5;x=torch.from_numpy(A['train_images'][:banks,:,None]);vx=torch.from_numpy(A['validation_images'][0,:,None]);y=torch.from_numpy(A['train_labels']);vy=torch.from_numpy(A['validation_labels']);m=cfg['feature_dim']
 if m:
  raw=A['train_'+cfg['branch']][:banks];mean=raw.reshape(-1,8).mean(0);std=raw.reshape(-1,8).std(0);keep=std>=1e-10;std=np.where(keep,std,1)
  f=torch.from_numpy((.3*np.where(keep,(raw-mean)/std,0)).astype(np.float32));vf=torch.from_numpy((.3*np.where(keep,(A['validation_'+cfg['branch']][0]-mean)/std,0)).astype(np.float32))
 else:f=torch.empty((banks,len(y),0));vf=torch.empty((len(vy),0));mean=std=keep=np.empty(0)
 torch.manual_seed(cfg['seed']);net=CompactCNN(cfg['family'],cfg['channels'],cfg['hidden_width'],m)
 opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);g=torch.Generator().manual_seed(2026092900+cfg['seed']);best=float('inf');stale=0;curve=[];t=time.perf_counter()
 for epoch in range(1,251):
  net.train();permutation=torch.randperm(len(y),generator=g);bids=torch.randint(banks,(len(y),),generator=g)
  for j in range(0,len(y),256):
   ids=permutation[j:j+256];bi=bids[j:j+256];opt.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(net(x[bi,ids],f[bi,ids]),y[ids]);loss.backward();opt.step()
  net.eval()
  with torch.no_grad():logits=net(vx,vf);ce=float(torch.nn.functional.cross_entropy(logits,vy));acc=float((logits.argmax(1)==vy).double().mean())
  curve.append([ce,acc])
  if ce<best-1e-8:
   best=ce;stale=0;best_epoch=epoch;state={k:v.detach().clone() for k,v in net.state_dict().items()};prob=logits.softmax(1).numpy();pred=prob.argmax(1)
  else:stale+=1
  if stale>=30:break
 r=dict(**cfg,topological=cfg['branch']=='topology',accuracy=float((pred==vy.numpy()).mean()),ce=best,selected_epoch=best_epoch,epochs_run=epoch,seconds=time.perf_counter()-t,curve=curve,protocol_sha256=digest)
 torch.save(dict(state_dict=state,config={k:v for k,v in r.items() if k!='curve'},mean=mean,std=std,keep=keep,protocol_sha256=digest),JOBS/(cfg['id']+'.pt'))
 np.savez_compressed(JOBS/(cfg['id']+'.npz'),predictions=pred,probabilities=prob);write(done,r);return r


def summarize(rows,digest):
 from collections import defaultdict
 groups=defaultdict(list)
 for r in rows:groups[r['id'].rsplit('_s',1)[0]].append(r)
 summaries=[]
 for name,rs in groups.items():
  rs.sort(key=lambda r:r['seed']);assert [r['seed'] for r in rs]==[0,1,2]
  r=rs[0];s={k:r[k] for k in ['family','branch','channels','hidden_width','feature_dim','parameters','budget','augmentation','optimizer_index','lr','weight_decay','topological']}
  s.update(id=name,mean_accuracy=float(np.mean([r['accuracy'] for r in rs])),mean_ce=float(np.mean([r['ce'] for r in rs])),seed_accuracy=[r['accuracy'] for r in rs],selected_epochs=[r['selected_epoch'] for r in rs]);summaries.append(s)
 key=lambda s:(-s['mean_accuracy'],s['mean_ce'],s['parameters'],s['id']);summaries.sort(key=key)
 selected={label:min([s for s in summaries if s['topological']==flag],key=key) for label,flag in [('topology',True),('baseline',False)]}
 write(OUT/'tiny_cnn_summary.json',dict(protocol_sha256=digest,test_read=False,all_configurations=summaries,selected=selected,scope='Development only, budget512; no test evidence.'));print(json.dumps(selected,indent=2),flush=True)


def main(workers):
 p=freeze();prepare(p);digest=sha(PROTOCOL);rows=[];started=time.perf_counter()
 with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
  fs=[pool.submit(fit,c,digest) for c in p['configs']]
  for f in as_completed(fs):
   r=f.result();rows.append(r);write(RESULT,dict(protocol_sha256=digest,test_read=False,completed=len(rows),total=len(p['configs']),runs=rows))
   write(OUT/'tiny_cnn_status.json',dict(updated_utc=now(),pid=os.getpid(),completed=len(rows),total=len(p['configs']),elapsed_seconds=time.perf_counter()-started))
   if len(rows)%12==0:print(json.dumps(dict(completed=len(rows),total=len(p['configs']),seconds=time.perf_counter()-started)),flush=True)
 summarize(rows,digest);write(OUT/'tiny_cnn_status.json',dict(updated_utc=now(),status='completed',completed=len(rows),total=len(p['configs'])))


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);a=parser.parse_args();main(a.workers)
