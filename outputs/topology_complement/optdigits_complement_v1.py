"""TRAIN/CV-only search for topology complementary to strong digit gradients.

No test loader exists here. Historical OptDigits test scores are not accessed.
All model families and features are fixed in the protocol before any fitting.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import sys
import json
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'outputs/real_topology'))
from optdigits_pilot import read_original, gradient_histograms, cells
from optdigits_ablation import scale_features

torch.set_num_threads(1)
OUT=ROOT/'outputs/topology_complement'
DATA=ROOT/'work/real_data'
PROTOCOL=OUT/'optdigits_complement_v1_protocol.json'
RESULT=OUT/'optdigits_complement_v1_validation.json'
BRANCHES={'gradient_base':[],
          'global_topology':['global_topology'],
          'global_geometry':['global_geometry'],
          'global_both':['global_topology','global_geometry'],
          'local_topology':['local_topology'],
          'local_geometry':['local_geometry'],
          'local_both':['local_topology','local_geometry']}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_protocol():
    if PROTOCOL.exists():return json.loads(PROTOCOL.read_text())
    if RESULT.exists():raise RuntimeError('Results predate protocol')
    OUT.mkdir(parents=True,exist_ok=True)
    p={'registered_at_utc':datetime.now(timezone.utc).isoformat(),
       'purpose':'Development-only search for Betti features complementary to a strong gradient NN. Old OptDigits test is not reused; any claim requires a future untouched external dataset.',
       'data':'Official original OptDigits binary32x32: tra1934 fitting, cv946 validation. wdep and windep members never read.',
       'data_archive_sha256':sha(DATA/'optdigits.zip'),
       'code_sha256':{str(f.relative_to(ROOT)):sha(f) for f in [Path(__file__),ROOT/'outputs/real_topology/optdigits_pilot.py',ROOT/'outputs/real_topology/optdigits_ablation.py']},
       'features':{'pixels':'64 block-averaged pixel values, each4x4 block.',
                   'gradient':'128 unsigned-gradient cell-histogram values (4x4 spatial cells,8bins), image-independent Sobel; preexisting corrected implementation.',
                   'global_topology':'beta0,beta1 at padded disk closing radii0,1,2,3:8features.',
                   'global_geometry':'area,4-edge perimeter of exactly the same four masks:8features.',
                   'local_topology':'Four disjoint16x16 quadrants, cropped BEFORE radius0,1,2,3 padded disk closing; beta0,beta1:32features.',
                   'local_geometry':'area,4-edge perimeter of the identical quadrant masks:32features.',
                   'connectivity':'closed foreground pixel squares;8-connectedforeground, beta1=beta0−(V−E+F); equivalent4-connectedbackground.'},
       'scaling':'StandardScaler fitted independently per feature group to tra only. Pixelweight1; gradient and all appended topology/geometry groups weight0.3. No feature-weight search.',
       'branches':BRANCHES,'parameter_budgets':[2500,5000],
       'architectures':{'one_hidden':'Linear(d,h)-ReLU-Linear(h,10); largest integer h satisfying(d+11)h+10<=budget',
                        'two_hidden':'Linear(d,h)-ReLU-Linear(h,h)-ReLU-Linear(h,10); largest integer h satisfyingh²+(d+12)h+10<=budget'},
       'training':{'optimizer':'AdamW','learning_rate':.01,'weight_decay':.01,'batch_size':256,'epochs':250,'seeds':[0,1,2,3,4],
                   'checkpoint':'maximum cv accuracy; minimum cv crossentropy breaks ties; earliest epoch breaks exact remaining tie; same rule and250epochs for every run'},
       'analysis':'Report all140runs (7representations×2budgets×2depths×5seeds), all paired validation differences. Best validation scores are development estimates, not untouched-test confirmation.',
       'selection_for_future_transfer':'Use validation only; no test access or test-driven feature/architecture changes.',
       'software':{'numpy':np.__version__,'torch':torch.__version__}}
    PROTOCOL.write_text(json.dumps(p,indent=2));return p


def verify_protocol(p):
    for path,digest in p['code_sha256'].items():
        if sha(ROOT/path)!=digest:raise RuntimeError('Code changed after protocol: '+path)
    if sha(DATA/'optdigits.zip')!=p['data_archive_sha256']:raise RuntimeError('Data archive changed')


def features(images):
    t=time.perf_counter()
    pixels=images.reshape(-1,8,4,8,4).mean(axis=(2,4)).reshape(len(images),64)
    grad=gradient_histograms(images);gradient_seconds=time.perf_counter()-t
    t=time.perf_counter();global_top,global_geom=scale_features(images)
    global_seconds=time.perf_counter()-t
    t=time.perf_counter();lt,lg=[],[]
    for row,col in [(0,0),(0,1),(1,0),(1,1)]:
        a,b=scale_features(images[:,row*16:(row+1)*16,col*16:(col+1)*16])
        lt.append(a);lg.append(b)
    groups={'pixels':pixels,'gradients':grad,'global_topology':global_top,'global_geometry':global_geom,
            'local_topology':np.column_stack(lt),'local_geometry':np.column_stack(lg)}
    return groups,{'n':len(images),'gradient_and_pixels_seconds':gradient_seconds,
                  'global_topology_and_geometry_seconds':global_seconds,'local_topology_and_geometry_seconds':time.perf_counter()-t}


def widths(d,budget,depth):
    h=1
    def params(w):return (d+11)*w+10 if depth==1 else w*w+(d+12)*w+10
    while params(h+1)<=budget:h+=1
    assert params(h)<=budget
    return h,params(h)


def make_model(d,h,depth):
    layers=[torch.nn.Linear(d,h),torch.nn.ReLU()]
    if depth==2:layers.extend([torch.nn.Linear(h,h),torch.nn.ReLU()])
    layers.append(torch.nn.Linear(h,10))
    return torch.nn.Sequential(*layers)


def fit(x,y,v,vy,budget,depth,seed):
    torch.manual_seed(seed);h,parameters=widths(x.shape[1],budget,depth)
    net=make_model(x.shape[1],h,depth)
    assert sum(p.numel() for p in net.parameters())==parameters
    xt,yt=torch.tensor(x,dtype=torch.float32),torch.tensor(y,dtype=torch.long)
    vt,vyt=torch.tensor(v,dtype=torch.float32),torch.tensor(vy,dtype=torch.long)
    optimizer=torch.optim.AdamW(net.parameters(),lr=.01,weight_decay=.01)
    best_accuracy=-1.;best_loss=float('inf');curve=[];t=time.perf_counter()
    for epoch in range(250):
        net.train()
        for batch in torch.randperm(len(xt)).split(256):
            optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(net(xt[batch]),yt[batch]);loss.backward();optimizer.step()
        net.eval()
        with torch.no_grad():
            logits=net(vt);pred=logits.argmax(1).numpy();acc=float((pred==vy).mean())
            vloss=float(torch.nn.functional.cross_entropy(logits,vyt))
        curve.append((acc,vloss))
        if acc>best_accuracy or (acc==best_accuracy and vloss<best_loss):
            best_accuracy,best_loss,best_epoch=acc,vloss,epoch
            best_pred=pred.copy();best_probs=logits.softmax(1).numpy()
    return {'seed':seed,'input_dim':x.shape[1],'width':h,'depth':depth,'budget':budget,'parameters':parameters,
            'accuracy':best_accuracy,'errors':int((best_pred!=vy).sum()),'cross_entropy':best_loss,
            'selected_epoch_zero_based':best_epoch,'seconds':time.perf_counter()-t},best_pred,best_probs,np.array(curve)


def main():
    if RESULT.exists():raise RuntimeError('Pilot results already exist; use preserved protocol and outputs, do not silently rerun')
    p=freeze_protocol();verify_protocol(p)
    train_img,ty=read_original('tra');val_img,vy=read_original('cv')
    train,ta=features(train_img);valid,va=features(val_img)
    scalers={k:StandardScaler().fit(a) for k,a in train.items()}
    tr={k:scalers[k].transform(a) for k,a in train.items()}
    va_arrays={k:scalers[k].transform(a) for k,a in valid.items()}
    np.savez_compressed(DATA/'optdigits_complement_trainval_v1.npz',
                        **{'tr_'+k:v for k,v in train.items()},**{'va_'+k:v for k,v in valid.items()})
    matrices={}
    for name,keys in BRANCHES.items():
        matrices[name]=(np.column_stack([tr['pixels'],.3*tr['gradients']]+[.3*tr[k] for k in keys]),
                        np.column_stack([va_arrays['pixels'],.3*va_arrays['gradients']]+[.3*va_arrays[k] for k in keys]))
    rows=[];predictions={};probabilities={};curves={};summaries=[];start=time.perf_counter()
    def save():
        RESULT.write_text(json.dumps({'protocol_sha256':sha(PROTOCOL),'test_read':False,'train_n':len(ty),'validation_n':len(vy),
                                     'feature_timing':{'train':ta,'validation':va},'elapsed_seconds':time.perf_counter()-start,
                                     'runs':rows,'summaries':summaries},indent=2))
    for budget in p['parameter_budgets']:
        for depth in (1,2):
            for branch,(x,v) in matrices.items():
                group=[]
                for seed in p['training']['seeds']:
                    r,pred,prob,curve=fit(x,ty,v,vy,budget,depth,seed);r['branch']=branch
                    key=f'b{budget}_d{depth}_{branch}_s{seed}'
                    rows.append(r);group.append(r);predictions[key]=pred;probabilities[key]=prob;curves[key]=curve
                    save()
                s={'budget':budget,'depth':depth,'branch':branch,'parameters':r['parameters'],
                   'mean':float(np.mean([r['accuracy'] for r in group])),
                   'std':float(np.std([r['accuracy'] for r in group],ddof=1)),
                   'seeds':[r['accuracy'] for r in group],
                   'epochs':[r['selected_epoch_zero_based']+1 for r in group]}
                summaries.append(s);save();print(json.dumps(s),flush=True)
    np.savez_compressed(OUT/'optdigits_complement_v1_predictions.npz',labels=vy,**predictions)
    np.savez_compressed(OUT/'optdigits_complement_v1_probabilities.npz',labels=vy,**probabilities)
    np.savez_compressed(OUT/'optdigits_complement_v1_learning_curves.npz',**curves)
    save()


if __name__=='__main__':main()
