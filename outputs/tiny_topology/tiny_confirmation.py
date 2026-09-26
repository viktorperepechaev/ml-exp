"""Frozen <=512parameter primary comparison and a descriptive capacity curve.

Selection/train never read wdep. After every registered model is fixed, predict
parses the image/label format without using labels and seals predictions.
Statistics uses labels only after this receipt has been written.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import argparse,hashlib,json,sys,time,subprocess,zipfile,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
import tiny_mlp_pilot as mlp
import tiny_cnn_pilot as cnn
ROOT=mlp.ROOT;OUT=mlp.OUT;WORK=mlp.WORK;PROTO=OUT/'tiny_confirmation_protocol.json';MODELS=WORK/'confirmation_models';TRAIN=WORK/'confirmation_train.npz'
sys.path.insert(0,str(ROOT/'outputs/real_topology'))
from optdigits_pilot import read_original,gradient_histograms
from optdigits_ablation import scale_features

torch.set_num_threads(1)


def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 t=Path(str(p)+'.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)


def register_rule():
 path=OUT/'tiny_confirmation_rule.json'
 if path.exists():return json.loads(path.read_text())
 p=dict(registered_utc=now(),scope='New user hypothesis: topology is useful under a strict small-neural-model budget; not a claim over unconstrained networks.',
  timing='After MLP exploration, while compact CNN grid remains running; before reading unused official wdep member. Primary512budget was declared before all tiny MLP fits.',
  budget=512,development='Official TRA1934/CV946; keep all completed feasible MLP configurations and all registered compactCNN configurations.',
  selection='Global best topology and global best non-topology by mean3seed CVaccuracy, then meanCE, fewer parameters, lexical family+id. No selection on TEST.',
  confirmation_seeds=list(range(10)),refit='All2880TRA+CV, same selected optimizer and architecture; fixed median3development-selected epochs. Refit all scalers on fullTRAIN; no further stopping/tuning.',
  held_out='optdigits-orig.wdep.Z,943images from writers already represented in training. Previously excluded from this research. This is a new-image test, NOT a new-writer test.',
  primary='Mean accuracy of individual <=512parameter models across10fixed seeds. No ensemble classifier; deploying one network uses its single-model weight count.',
  uncertainty='Paired image sign-flip test on10seed-averaged per-image correctness differences; crossed paired-seed/image bootstrap95% CI. Images, not seed-image pairs, are permutation units. Writer identities unavailable, so no writer-independent confidence claim.',
  multiple_testing='Bonferroni6: old OptDigits-windep, PenDigits, CharacterTrajectories, USPS, RIM-ONE DL and this new OptDigits-wdep comparison. p_adjusted=min(1,6*p). Any prior5test family claim needs reevaluation at6for a combined-series claim.',
  success='Positive mean accuracy difference, adjustedp<=.05, crossed-bootstrap95% lower bound>0, at least8/10positive seed pairs. Report failures. No SOTA/priority or total-latency reduction inferred from fewer trained weights.',
  statistics=dict(bootstrap_repetitions=20000,signflip_draws=99999,seed=20260930),test_read=False)
 write(path,p);return p


def select_freeze():
 rule=register_rule()
 if PROTO.exists():raise RuntimeError('Confirmation already frozen')
 m=json.loads((OUT/'tiny_mlp_summary.json').read_text());c=json.loads((OUT/'tiny_cnn_summary.json').read_text())
 assert m['fits']==456
 cv=json.loads((OUT/'tiny_cnn_validation.json').read_text())
 assert cv['completed']==cv['total']==288 and len(c['all_configurations'])==96
 rows=[]
 for r in m['all_configurations']:
  if r['budget']==512:rows.append(dict(**r,model_family='mlp',weight_decay=.01))
 for r in c['all_configurations']:rows.append(dict(**r,model_family='cnn'))
 rank=lambda r:(-r['mean_accuracy'],r['mean_ce'],r['parameters'],r['model_family']+'_'+r['id'])
 chosen={label:min([r for r in rows if r['topological']==flag],key=rank) for label,flag in [('topology',True),('baseline',False)]}
 for budget in [256,1024,2048]:
  for label,flag in [('topology',True),('baseline',False)]:
   selected=min([r for r in m['all_configurations'] if r['budget']==budget and r['topological']==flag],key=lambda r:(-r['mean_accuracy'],r['mean_ce'],r['parameters'],r['id']))
   chosen[f'b{budget}_{label}']=dict(**selected,model_family='mlp',weight_decay=.01)
 for r in chosen.values():r['final_epochs']=int(np.median(r['selected_epochs']));assert r['parameters']<=r['budget']
 audit_selected(chosen)
 rule=dict(rule,exact_test_amendment=json.loads((OUT/'tiny_exact_test_amendment.json').read_text()),secondary_curve_amendment=json.loads((OUT/'tiny_secondary_curve_amendment.json').read_text()))
 dependencies=['tiny_confirmation.py','tiny_confirmation_rule.json','tiny_mlp_pilot.py','tiny_cnn_pilot.py','tiny_mlp_summary.json','tiny_cnn_summary.json','tiny_cnn_protocol.json','tiny_mlp_protocol.json','tiny_selected_development_audit.json','tiny_exact_test_amendment.json','tiny_secondary_curve_amendment.json']
 p=dict(frozen_utc=now(),rule=rule,pipelines=chosen,seeds=rule['confirmation_seeds'],
  source_sha256={n:sha(OUT/n) for n in dependencies},data_sha256=sha(ROOT/'work/real_data/optdigits.zip'),
  upstream_features_sha256=sha(ROOT/'work/real_data/optdigits_complement_trainval_v1.npz'),
  upstream_code_sha256={n:sha(ROOT/'outputs/real_topology'/n) for n in ['optdigits_pilot.py','optdigits_ablation.py']},test_read=False)
 write(PROTO,p);print(json.dumps(chosen,indent=2),flush=True)


def verify():
 p=json.loads(PROTO.read_text())
 for n,h in p['source_sha256'].items():assert sha(OUT/n)==h
 for n,h in p['upstream_code_sha256'].items():assert sha(ROOT/'outputs/real_topology'/n)==h
 assert sha(ROOT/'work/real_data/optdigits.zip')==p['data_sha256']
 assert sha(ROOT/'work/real_data/optdigits_complement_trainval_v1.npz')==p['upstream_features_sha256']
 return p


def prepare():
 p=verify()
 if TRAIN.exists():return
 ti,ty=read_original('tra');vi,vy=read_original('cv');images=np.concatenate([ti,vi]);labels=np.concatenate([ty,vy]);assert len(labels)==2880
 bank=np.zeros((5,*images.shape),bool);bank[0]=images;bank[1,:,:,1:]=images[:,:,:-1];bank[2,:,:,:-1]=images[:,:,1:];bank[3,:,1:,:]=images[:,:-1,:];bank[4,:,:-1,:]=images[:,1:,:]
 top=[];geo=[]
 for b in bank:t,g=scale_features(b);top.append(t);geo.append(g)
 arrays=dict(images=bank.astype(np.float32),topology=np.stack(top).astype(np.float32),geometry=np.stack(geo).astype(np.float32),labels=labels)
 with np.load(ROOT/'work/real_data/optdigits_complement_trainval_v1.npz') as z:
  for name in ['pixels','gradients','global_topology','global_geometry']:arrays['mlp_'+name]=np.concatenate([z['tr_'+name],z['va_'+name]])
 np.testing.assert_array_equal(arrays['mlp_global_topology'].astype(np.float32),arrays['topology'][0]);np.testing.assert_array_equal(arrays['mlp_global_geometry'].astype(np.float32),arrays['geometry'][0])
 np.savez_compressed(TRAIN,**arrays);write(WORK/'confirmation_train_meta.json',dict(protocol_sha256=sha(PROTO),cache_sha256=sha(TRAIN),n=2880,test_read=False))


def scaler(x):
 mean=x.mean(0);std=x.std(0);keep=std>=1e-10;return dict(mean=mean,std=np.where(keep,std,1),keep=keep)

def scaled(x,s):return np.where(s['keep'],(x-s['mean'])/s['std'],0)


def model(cfg):
 if cfg['model_family']=='mlp':net=mlp.Tiny(cfg['base_dim'],cfg['side_dim'],cfg['hidden_width'],cfg['depth'],cfg['fusion'])
 else:net=cnn.CompactCNN(cfg['family'],cfg['channels'],cfg['hidden_width'],cfg['feature_dim'])
 assert sum(p.numel() for p in net.parameters())==cfg['parameters']<=cfg['budget'];return net


def training_arrays(cfg):
 with np.load(TRAIN) as z:
  y=torch.from_numpy(z['labels'].copy())
  if cfg['model_family']=='mlp':
   branch=mlp.BRANCHES[cfg['branch']];names=['pixels']+(['gradients'] if branch['base']=='hog' else [])+(['global_'+branch['side']] if branch['side'] else [])
   groups={k:z['mlp_'+k].copy() for k in names};sc={k:scaler(v) for k,v in groups.items()};v={k:scaled(x,sc[k]).astype(np.float32) for k,x in groups.items()}
   x=v['pixels'];x=np.column_stack([x,.3*v['gradients']]) if branch['base']=='hog' else x
   f=np.empty((len(y),0),np.float32) if branch['side'] is None else .3*v['global_'+branch['side']]
   return torch.from_numpy(x),torch.from_numpy(f),y,sc,1
  banks=1 if cfg['augmentation']=='none' else 5;x=torch.from_numpy(z['images'][:banks,:,None].copy())
  if cfg['branch']=='raw':f=torch.empty((banks,len(y),0));sc={}
  else:
   a=z[cfg['branch']][:banks];sc=scaler(a.reshape(-1,8));f=torch.from_numpy((.3*scaled(a,sc)).astype(np.float32))
  return x,f,y,sc,banks


def train_one(label,cfg,seed,digest):
 torch.set_num_threads(1);name=f'{label}_s{seed}';MODELS.mkdir(exist_ok=True);done=MODELS/(name+'.json');weight=MODELS/(name+'.pt')
 if done.exists():
  r=json.loads(done.read_text());assert r['protocol_sha256']==digest and r['weight_sha256']==sha(weight);return r
 x,f,y,sc,banks=training_arrays(cfg);torch.manual_seed(seed);net=model(cfg);opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
 base_seed=2026092800 if cfg['model_family']=='mlp' else 2026092900;g=torch.Generator().manual_seed(base_seed+seed);curve=[];t=time.perf_counter()
 for epoch in range(1,cfg['final_epochs']+1):
  net.train();perm=torch.randperm(len(y),generator=g);bid=torch.randint(banks,(len(y),),generator=g) if cfg['model_family']=='cnn' else None;total=0
  for j in range(0,len(y),256):
   ids=perm[j:j+256];opt.zero_grad(set_to_none=True)
   logits=net(x[ids],f[ids]) if cfg['model_family']=='mlp' else net(x[bid[j:j+256],ids],f[bid[j:j+256],ids])
   loss=torch.nn.functional.cross_entropy(logits,y[ids]);loss.backward();opt.step();total+=float(loss.detach())*len(ids)
  curve.append(total/len(y))
 torch.save(dict(state_dict=net.state_dict(),config=cfg,seed=seed,scaling=sc,protocol_sha256=digest),weight)
 r=dict(name=name,seed=seed,pipeline=label,protocol_sha256=digest,weight_sha256=sha(weight),parameters=cfg['parameters'],epochs=cfg['final_epochs'],seconds=time.perf_counter()-t,training_loss=curve,test_read=False)
 write(done,r);return r


def train(workers):
 p=verify();prepare();digest=sha(PROTO);rows=[]
 with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
  fs=[pool.submit(train_one,label,cfg,s,digest) for label,cfg in p['pipelines'].items() for s in p['seeds']]
  for f in as_completed(fs):
   r=f.result();rows.append(r);write(OUT/'tiny_confirmation_training_status.json',dict(updated_utc=now(),completed=len(rows),total=len(p['pipelines'])*len(p['seeds']),runs=rows,test_read=False));print(r['name'],flush=True)
 write(OUT/'tiny_confirmation_weights_receipt.json',dict(frozen_utc=now(),protocol_sha256=digest,weights={p.name:sha(p) for p in MODELS.glob('*.pt')},test_read=False))


def read_wdep():
 """Gated parser; every registered model must have a frozen-weight receipt."""
 p=verify();receipt=json.loads((OUT/'tiny_confirmation_weights_receipt.json').read_text());assert receipt['protocol_sha256']==sha(PROTO) and len(receipt['weights'])==len(p['pipelines'])*len(p['seeds'])
 expected={f'{label}_s{s}.pt' for label in p['pipelines'] for s in p['seeds']}
 assert set(receipt['weights'])==expected
 for name,digest in receipt['weights'].items():assert sha(MODELS/name)==digest
 with zipfile.ZipFile(ROOT/'work/real_data/optdigits.zip') as z:compressed=z.read('optdigits-orig.wdep.Z')
 text=subprocess.run(['gzip','-dc'],input=compressed,capture_output=True,check=True).stdout.decode('ascii');lines=[s.strip() for s in text.splitlines()];images=[];labels=[];i=0
 while i<len(lines):
  if len(lines[i])==32 and set(lines[i])<={'0','1'}:
   rows=lines[i:i+32];assert len(rows)==32 and all(len(s)==32 and set(s)<={'0','1'} for s in rows);lab=lines[i+32];assert len(lab)==1 and lab.isdigit()
   images.append(np.frombuffer(''.join(rows).encode(),np.uint8).reshape(32,32)==ord('1'));labels.append(int(lab));i+=33
  else:i+=1
 assert len(labels)==943
 return np.stack(images),np.array(labels),hashlib.sha256(compressed).hexdigest()


def eval_arrays(images,cfg,sc):
 if cfg['model_family']=='mlp':
  branch=mlp.BRANCHES[cfg['branch']];groups=dict(pixels=images.reshape(-1,8,4,8,4).mean((2,4)).reshape(len(images),64))
  if branch['base']=='hog':groups['gradients']=gradient_histograms(images)
  if branch['side']:
   t,g=scale_features(images);groups['global_'+branch['side']]=t if branch['side']=='topology' else g
  v={k:scaled(a,sc[k]).astype(np.float32) for k,a in groups.items()};x=v['pixels'];x=np.column_stack([x,.3*v['gradients']]) if branch['base']=='hog' else x
  f=np.empty((len(x),0),np.float32) if branch['side'] is None else .3*v['global_'+branch['side']]
  return torch.from_numpy(x),torch.from_numpy(f)
 x=torch.from_numpy(images[:,None].astype(np.float32))
 if cfg['branch']=='raw':f=np.empty((len(images),0),np.float32)
 else:
  t,g=scale_features(images);raw=(t if cfg['branch']=='topology' else g).astype(np.float32);f=(.3*scaled(raw,sc)).astype(np.float32)
 return x,torch.from_numpy(f)


def audit_selected(chosen):
 images,y=read_original('cv');checks=[]
 for label,cfg in chosen.items():
  name=cfg['id']+'_s0'
  if cfg['model_family']=='mlp':
   ck=torch.load(WORK/(name+'.pt'),weights_only=False)
   branch=mlp.BRANCHES[cfg['branch']];names=['pixels']+(['gradients'] if branch['base']=='hog' else [])+(['global_'+branch['side']] if branch['side'] else [])
   with np.load(mlp.CACHE) as z:sc={k:{a:z[a+'_'+k] for a in ['mean','std','keep']} for k in names}
   predpath=WORK/(name+'.npz')
  else:
   ck=torch.load(cnn.JOBS/(name+'.pt'),weights_only=False);sc={a:ck[a] for a in ['mean','std','keep']};predpath=cnn.JOBS/(name+'.npz')
  x,f=eval_arrays(images,cfg,sc);net=model(cfg);net.load_state_dict(ck['state_dict']);net.eval()
  with torch.no_grad():prob=net(x,f).softmax(1).numpy()
  with np.load(predpath) as z:
   np.testing.assert_allclose(prob,z['probabilities'],atol=3e-7,rtol=3e-6);np.testing.assert_array_equal(prob.argmax(1),z['predictions'])
  checks.append(dict(pipeline=label,model=name,validation_predictions_replayed=True,accuracy=float((prob.argmax(1)==y).mean())))
 write(OUT/'tiny_selected_development_audit.json',dict(created_utc=now(),test_read=False,checks=checks,passed=True))


def exact_signflip(integer_effects):
 """Exact weighted Rademacher tail via a Bernoulli subset-sum distribution.

 Each weight is the sum of paired correctness differences across seeds for ONE
 image. Duplicating seeds does not create additional independent observations.
 """
 a=np.asarray(integer_effects,dtype=np.int64);weights=np.abs(a[a!=0]);total=int(weights.sum());observed=int(a.sum())
 if not len(weights):return 1.
 distribution=np.zeros(total+1);distribution[0]=1.;used=0
 for w in weights:
  w=int(w);old=distribution[:used+1].copy();distribution[:used+w+1]=0
  distribution[:used+1]+=.5*old;distribution[w:w+used+1]+=.5*old;used+=w
 assert abs(distribution.sum()-1)<1e-12
 threshold=(observed+total+1)//2
 return float(distribution[max(0,threshold):].sum())


def predict():
 p=verify();dest=OUT/'tiny_confirmation_predictions.npz'
 if dest.exists():raise RuntimeError('Predictions already exist; preserve them')
 images,unused_labels,member_hash=read_wdep();arrays={};timing=[]
 for label,cfg in p['pipelines'].items():
  first=torch.load(MODELS/f'{label}_s0.pt',weights_only=False);t=time.perf_counter();x,f=eval_arrays(images,cfg,first['scaling']);preptime=time.perf_counter()-t;probs=[];times=[]
  for seed in p['seeds']:
   ck=torch.load(MODELS/f'{label}_s{seed}.pt',weights_only=False);net=model(cfg);net.load_state_dict(ck['state_dict']);net.eval();t=time.perf_counter()
   with torch.no_grad():pr=torch.cat([net(x[i:i+256],f[i:i+256]).softmax(1) for i in range(0,len(x),256)]).numpy()
   times.append(time.perf_counter()-t);probs.append(pr)
  arrays[label+'_probabilities']=np.stack(probs);arrays[label+'_predictions']=arrays[label+'_probabilities'].argmax(2)
  timing.append(dict(pipeline=label,preprocessing_seconds=preptime,individual_network_seconds=times,
   caveat='Exploratory shared-extraction timing; scale_features computes both topology and geometry, so this does NOT isolate necessary per-pipeline feature cost or establish speed savings.'))
 np.savez_compressed(dest,**arrays)
 write(OUT/'tiny_confirmation_prediction_receipt.json',dict(predicted_utc=now(),protocol_sha256=sha(PROTO),predictions_sha256=sha(dest),member_sha256=member_hash,timing=timing,
  note='Parser reads fixed-format labels along with images, but prediction function never uses labels or computes metrics; saved predictions precede analysis.',metrics_computed=False))


def stats():
 p=verify();dest=OUT/'tiny_confirmation_test_results.json'
 if dest.exists():raise RuntimeError('Final analysis already exists')
 rec=json.loads((OUT/'tiny_confirmation_prediction_receipt.json').read_text());assert rec['predictions_sha256']==sha(OUT/'tiny_confirmation_predictions.npz')
 _,y,member_hash=read_wdep();assert member_hash==rec['member_sha256']
 with np.load(OUT/'tiny_confirmation_predictions.npz') as z:
  t=(z['topology_predictions']==y).astype(int);b=(z['baseline_predictions']==y).astype(int)
  curve=[]
  for budget in [256,512,1024,2048]:
   prefix='' if budget==512 else f'b{budget}_';ta=(z[prefix+'topology_predictions']==y).mean(1);ba=(z[prefix+'baseline_predictions']==y).mean(1)
   curve.append(dict(budget=budget,topology_mean_accuracy=float(ta.mean()),baseline_mean_accuracy=float(ba.mean()),mean_difference=float((ta-ba).mean()),topology_seed_accuracy=ta.tolist(),baseline_seed_accuracy=ba.tolist(),scope='Primary globalMLP/CNN comparison' if budget==512 else 'Prespecified secondary MLP-only descriptive point, not an alternative primary test'))
 diff=t-b;rng=np.random.default_rng(p['rule']['statistics']['seed']);rawp=exact_signflip(diff.sum(0))
 patterns,counts=np.unique(diff.T,axis=0,return_counts=True);ci=[];s,n=diff.shape
 for i in range(0,20000,256):
  k=min(256,20000-i);iw=rng.multinomial(n,counts/n,size=k);sw=rng.multinomial(s,np.ones(s)/s,size=k);ci.extend(((iw@patterns)*sw).sum(1)/(s*n))
 interval=np.quantile(ci,[.025,.975]);seeds=diff.mean(1);adj=min(1.,6*rawp);success=bool(diff.mean()>0 and adj<=.05 and interval[0]>0 and (seeds>0).sum()>=8)
 r=dict(analyzed_utc=now(),protocol_sha256=sha(PROTO),test_n=n,test='Official writer-dependent wdep, new images of training writers; no claim on unseen writers.',
  primary=dict(topology_mean_accuracy=float(t.mean()),baseline_mean_accuracy=float(b.mean()),mean_accuracy_difference=float(diff.mean()),
   crossed_seed_image_bootstrap95=interval.tolist(),one_sided_paired_image_permutation_p=rawp,permutation_computation='Exact weighted sign-flip via subset-sum dynamic programming, not Monte Carlo; image units.',bonferroni6_p=adj,
   relative_mean_error_reduction=float((b.mean()-t.mean())/(b.mean()-1)) if b.mean()<1 else None),
  seeds=dict(topology_accuracy=t.mean(1).tolist(),baseline_accuracy=b.mean(1).tolist(),paired_differences=seeds.tolist(),positive_pairs=int((seeds>0).sum())),
  pipelines=p['pipelines'],secondary_budget_curve=curve,success_gate_passed=success,success_gate=p['rule']['success'],single_model_not_ensemble=True,
  scope='Finite declared <=512parameter MLP/CNN families. No SOTA/novelty claim. Preprocessing and scaler cost additional; learned parameter count alone is not total compute cost.')
 write(dest,r);print(json.dumps(r,indent=2),flush=True)


if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['register','freeze','train','predict','stats']);a.add_argument('--workers',type=int,default=4);args=a.parse_args()
 if args.mode=='register':register_rule()
 elif args.mode=='freeze':select_freeze()
 elif args.mode=='train':train(args.workers)
 elif args.mode=='predict':predict()
 else:stats()
