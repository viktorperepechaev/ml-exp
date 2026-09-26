"""Resource-constrained OptDigits MLP development, TRAIN/CV only.

No wdep or windep test member can be read by this program. All controls receive
identical tuning opportunities and every trainable side-branch weight is counted.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import argparse,hashlib,json,sys,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/tiny_topology';WORK=ROOT/'work/tiny_topology'
sys.path.insert(0,str(ROOT/'outputs/real_topology'))
from optdigits_pilot import read_original

torch.set_num_threads(1)
SOURCE=ROOT/'work/real_data/optdigits_complement_trainval_v1.npz'
PROTOCOL=OUT/'tiny_mlp_protocol.json';RESULT=OUT/'tiny_mlp_validation.json';CACHE=WORK/'mlp_arrays.npz'
BRANCHES={
 'pixels':dict(base='pixels',side=None,fusion='early'),
 'pixels_hog':dict(base='hog',side=None,fusion='early'),
 'pixels_geometry_early':dict(base='pixels',side='geometry',fusion='early'),
 'pixels_topology_early':dict(base='pixels',side='topology',fusion='early'),
 'pixels_geometry_late':dict(base='pixels',side='geometry',fusion='late'),
 'pixels_topology_late':dict(base='pixels',side='topology',fusion='late'),
 'hog_geometry_early':dict(base='hog',side='geometry',fusion='early'),
 'hog_topology_early':dict(base='hog',side='topology',fusion='early'),
 'hog_geometry_late':dict(base='hog',side='geometry',fusion='late'),
 'hog_topology_late':dict(base='hog',side='topology',fusion='late')}
ARRAYS=None


def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 t=Path(str(p)+'.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)


def count(d,m,h,depth,fusion):
 return (d+(m if fusion=='early' else 0)+11)*h+10+(depth-1)*(h*h+h)+(10*m if fusion=='late' else 0)

def width(d,m,depth,budget,fusion):
 h=0
 while count(d,m,h+1,depth,fusion)<=budget:h+=1
 if h<1:raise ValueError('No feasible hidden width')
 return h


class Tiny(torch.nn.Module):
 def __init__(self,d,m,h,depth,fusion):
  super().__init__();self.fusion=fusion
  layers=[torch.nn.Linear(d+(m if fusion=='early' else 0),h),torch.nn.ReLU()]
  for _ in range(depth-1):layers += [torch.nn.Linear(h,h),torch.nn.ReLU()]
  self.body=torch.nn.Sequential(*layers);self.head=torch.nn.Linear(h+(m if fusion=='late' else 0),10)
 def forward(self,x,f):
  h=self.body(torch.cat([x,f],1) if self.fusion=='early' else x)
  return self.head(torch.cat([h,f],1) if self.fusion=='late' else h)


def freeze():
 source_protocol=ROOT/'outputs/topology_complement/optdigits_complement_v1_protocol.json'
 if PROTOCOL.exists():
  p=json.loads(PROTOCOL.read_text());assert p['code_sha256']==sha(__file__) and p['feature_cache_sha256']==sha(SOURCE);return p
 p=dict(registered_utc=now(),question='Does explicit topology help especially when neural trainable-parameter budgets are256/512/1024/2048?',
  scope='Exploratory development only. OptDigits TRA1934/CV946, repeatedly observed development data. Neither official test read; no new confirmatory claim from this pilot.',
  primary_budget_for_future_confirmation=512,budgets=[256,512,1024,2048],branches=BRANCHES,
  features='Pixels64:4x4block means of original32x32binary input. HOG128: existing corrected spatial unsigned gradients. Topology8: beta0,beta1 at padded disk closing radii0/1/2/3. Geometry8: area/perimeter of the same masks. No invariants added to geometry branches.',
  side_layer='Late fusion logits=B ReLU(Ax+a)+C phi+b; all C weights counted. Early fusion allows nonlinear interactions before hidden bottleneck. Geometry has identical fusion alternatives and feature dimension.',
  normalization='Fit per-feature mean/std on TRA only; columns with std<1e-10 set0 in both TRA and CV. Pixelweight1; HOG and side features weight0.3.',
  architectures=dict(depths=[1,2],hidden_width='Largest positive integer under the declared total parameter budget, with all hidden layers same width.',
   early_count='(base_dim+side_dim+11)*h+10+(depth-1)*(h*h+h)',late_count='(base_dim+11)*h+10+(depth-1)*(h*h+h)+10*side_dim'),
  optimizers=[dict(lr=.003,weight_decay=.01),dict(lr=.01,weight_decay=.01)],optimizer='AdamW',max_epochs=250,patience=30,batch_size=256,seeds=[0,1,2],
  checkpoint='Lowest CV cross-entropy, improvement>1e-8, earliest tie; patience30. Same criterion for all branches.',
  rng='Model initialization seed=s; independent torch permutation generator seed2026092800+s shared across all representations and budgets.',
  selection='Within each budget and topology status: maximum mean3seed CVaccuracy, then meanCE, fewer parameters, lexicalid. Keep every result. No significance inference after CVselection.',
  limitations='MLPs only in this first pilot. A tiny CNN can be a stronger resource-matched control and must be checked before a broad claim. Feature computation, scalers and fixed preprocessing are not free and require separate memory/latency accounting.',
  fits=4*len(BRANCHES)*2*2*3,code_sha256=sha(__file__),feature_cache_sha256=sha(SOURCE),
  upstream_protocol_sha256=sha(source_protocol),upstream_code_sha256=json.loads(source_protocol.read_text())['code_sha256'],test_read=False)
 write(PROTOCOL,p);return p


def prepare(p):
 if CACHE.exists():
  meta=json.loads((WORK/'mlp_arrays_meta.json').read_text());assert meta['source_sha256']==sha(SOURCE) and meta['cache_sha256']==sha(CACHE);return
 with np.load(SOURCE) as z:tr={k[3:]:z[k] for k in z.files if k.startswith('tr_')};va={k[3:]:z[k] for k in z.files if k.startswith('va_')}
 _,ty=read_original('tra');_,vy=read_original('cv');assert len(ty)==1934 and len(vy)==946
 arrays=dict(train_y=ty,validation_y=vy)
 for key in ['pixels','gradients','global_topology','global_geometry']:
  mean=tr[key].mean(0);std=tr[key].std(0);keep=std>=1e-10;std=np.where(keep,std,1)
  arrays['train_'+key]=np.where(keep,(tr[key]-mean)/std,0).astype(np.float32)
  arrays['validation_'+key]=np.where(keep,(va[key]-mean)/std,0).astype(np.float32)
  arrays['mean_'+key]=mean;arrays['std_'+key]=std;arrays['keep_'+key]=keep
 np.savez_compressed(CACHE,**arrays);write(WORK/'mlp_arrays_meta.json',dict(source_sha256=sha(SOURCE),cache_sha256=sha(CACHE),protocol_sha256=sha(PROTOCOL),test_read=False))


def initialize():
 global ARRAYS
 torch.set_num_threads(1)
 with np.load(CACHE) as z:ARRAYS={k:torch.from_numpy(z[k].copy()) for k in z.files if k.startswith(('train_','validation_'))}


def inputs(branch,part):
 b=BRANCHES[branch];x=ARRAYS[part+'_pixels']
 if b['base']=='hog':x=torch.cat([x,.3*ARRAYS[part+'_gradients']],1)
 f=torch.empty((len(x),0)) if b['side'] is None else .3*ARRAYS[part+'_global_'+b['side']]
 return x,f


def fit(cfg,digest):
 name=cfg['id'];done=WORK/(name+'.json');weight=WORK/(name+'.pt');predfile=WORK/(name+'.npz')
 if done.exists():
  r=json.loads(done.read_text());assert r['protocol_sha256']==digest;return r
 x,f=inputs(cfg['branch'],'train');v,vf=inputs(cfg['branch'],'validation');y=ARRAYS['train_y'];vy=ARRAYS['validation_y'];b=BRANCHES[cfg['branch']]
 d,m=x.shape[1],f.shape[1];h=width(d,m,cfg['depth'],cfg['budget'],b['fusion']);torch.manual_seed(cfg['seed'])
 net=Tiny(d,m,h,cfg['depth'],b['fusion']);parameters=sum(p.numel() for p in net.parameters());assert parameters==count(d,m,h,cfg['depth'],b['fusion'])<=cfg['budget']
 opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=.01);g=torch.Generator().manual_seed(2026092800+cfg['seed'])
 best=float('inf');stale=0;curve=[];t=time.perf_counter()
 for epoch in range(1,251):
  net.train()
  for ix in torch.randperm(len(y),generator=g).split(256):
   opt.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(net(x[ix],f[ix]),y[ix]);loss.backward();opt.step()
  net.eval()
  with torch.no_grad():logits=net(v,vf);ce=float(torch.nn.functional.cross_entropy(logits,vy));acc=float((logits.argmax(1)==vy).double().mean())
  curve.append([ce,acc])
  if ce<best-1e-8:
   best=ce;stale=0;best_epoch=epoch;state={k:p.detach().clone() for k,p in net.state_dict().items()};prob=logits.softmax(1).numpy();pred=prob.argmax(1)
  else:stale+=1
  if stale>=30:break
 r=dict(**cfg,base_dim=d,side_dim=m,fusion=b['fusion'],topological=b['side']=='topology',hidden_width=h,parameters=parameters,
  accuracy=float((pred==vy.numpy()).mean()),ce=best,selected_epoch=best_epoch,epochs_run=epoch,seconds=time.perf_counter()-t,curve=curve,protocol_sha256=digest)
 torch.save(dict(state_dict=state,config={k:v for k,v in r.items() if k!='curve'},protocol_sha256=digest),weight)
 np.savez_compressed(predfile,probabilities=prob,predictions=pred);write(done,r);return r


def summarize(rows,digest):
 from collections import defaultdict
 groups=defaultdict(list)
 for r in rows:groups[r['id'].rsplit('_s',1)[0]].append(r)
 summaries=[]
 for name,rs in sorted(groups.items()):
  rs.sort(key=lambda r:r['seed']);assert [r['seed'] for r in rs]==[0,1,2]
  r=rs[0];s={k:r[k] for k in ['budget','branch','depth','lr','parameters','topological','fusion','hidden_width','base_dim','side_dim']}
  s.update(id=name,mean_accuracy=float(np.mean([r['accuracy'] for r in rs])),mean_ce=float(np.mean([r['ce'] for r in rs])),seed_accuracy=[r['accuracy'] for r in rs],selected_epochs=[r['selected_epoch'] for r in rs])
  summaries.append(s)
 key=lambda s:(-s['mean_accuracy'],s['mean_ce'],s['parameters'],s['id'])
 selected={}
 for budget in [256,512,1024,2048]:
  selected[str(budget)]={label:min([s for s in summaries if s['budget']==budget and s['topological']==topo],key=key) for label,topo in [('topology',True),('baseline',False)]}
 result=dict(created_utc=now(),protocol_sha256=digest,test_read=False,scope='Exploratory TRAIN/CV results, no independent-test confirmation; not all tiny NN families yet.',fits=len(rows),configurations=len(summaries),selected=selected,all_configurations=summaries)
 write(OUT/'tiny_mlp_summary.json',result);print(json.dumps(selected,indent=2),flush=True)


def main(workers):
 p=freeze();prepare(p);digest=sha(PROTOCOL)
 configs=[dict(id=f'b{b}_{name}_d{d}_o{o}_s{s}',budget=b,branch=name,depth=d,lr=opt['lr'],optimizer_index=o,seed=s)
   for b in p['budgets'] for name in BRANCHES for d in [1,2] for o,opt in enumerate(p['optimizers']) for s in p['seeds']]
 rows=[];start=time.perf_counter()
 with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
  futures=[pool.submit(fit,c,digest) for c in configs]
  for future in as_completed(futures):
   r=future.result();rows.append(r)
   write(RESULT,dict(protocol_sha256=digest,test_read=False,completed=len(rows),total=len(configs),runs=rows))
   write(OUT/'tiny_mlp_status.json',dict(updated_utc=now(),pid=os.getpid(),completed=len(rows),total=len(configs),elapsed_seconds=time.perf_counter()-start,current_completed=r['id']))
   if len(rows)%24==0:print(json.dumps(dict(completed=len(rows),total=len(configs),seconds=time.perf_counter()-start)),flush=True)
 summarize(rows,digest)
 write(OUT/'tiny_mlp_status.json',dict(updated_utc=now(),status='completed',completed=len(rows),total=len(configs),elapsed_seconds=time.perf_counter()-start))


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);a=parser.parse_args();main(a.workers)
