"""Transparent supplementary development: a smaller front end at the same cap."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import json,time,os
import torch
import pilot as p
from single_field import SingleFieldClassifier
BASE=p.OUT;OUT=BASE/'single_field';OUT.mkdir(exist_ok=True);JOBS=p.WORK/'single_field_jobs';JOBS.mkdir(exist_ok=True)
KINDS=['euler_learned','euler_frozen','area_learned','perimeter_learned','geometry_learned']

def initialize():
    p.SmallClassifier=SingleFieldClassifier;p.JOBS=JOBS;p.init()

def main():
    protocol=OUT/'pilot_protocol.json'
    counts={}
    for k in KINDS:
        torch.manual_seed(0);m=SingleFieldClassifier(k,torch.zeros(64),torch.ones(64),torch.ones(64,dtype=torch.bool));x=torch.rand(32,1,8,8)
        loss=torch.nn.functional.cross_entropy(m(x),torch.arange(32)%10);loss.backward()
        if k=='euler_learned':assert m.front.conv.weight.grad.norm()>0 and m.front.thresholds.grad.norm()>0
        counts[k]=dict(trainable=m.trainable_parameters,stored=m.stored_parameters,hidden=m.hidden)
    configs=[dict(id=f'single_{kind}_o{o}_s{s}',kind=kind,lr=lr,seed=s,budget=512) for kind in KINDS for o,lr in enumerate([.003,.01]) for s in range(3)]
    data=dict(registered_utc=p.now(),timing='After partial first-pilot development; no new TEST access. Supplementary exploration, not a preregistered independent confirmation.',
        rationale='Two fields/eightoutputs force the head from6to5hidden units under512. One field/fouroutputs gives506weights and6hidden units; apply the same change to all three learned geometric controls and the frozen-Euler ablation.',
        inherited_training_protocol_sha256=p.sha(BASE/'pilot_protocol.json'),training_rule='Identical optimizer, learning-rate grid, batches, seeds, stopping and automatic500to1000extension as the first pilot. Same TRA/CV arrays.',
        counts=counts,configs=configs,fits=len(configs),source_sha256={n:p.sha(BASE/n) for n in ['euler_layer.py','pilot.py','single_field.py','single_field_pilot.py']},test_read=False)
    if not protocol.exists():p.write(protocol,data)
    else:
        saved=json.loads(protocol.read_text())
        for k,v in saved['source_sha256'].items():assert p.sha(BASE/k)==v
    digest=p.sha(protocol);rows=[];t=time.perf_counter()
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        futures=[pool.submit(p.fit,c,digest) for c in configs]
        for f in as_completed(futures):
            r=f.result();rows.append(r);p.write(OUT/'pilot_results.json',dict(completed=len(rows),total=len(configs),runs=rows,protocol_sha256=digest,test_read=False))
            p.write(OUT/'pilot_status.json',dict(updated_utc=p.now(),pid=os.getpid(),completed=len(rows),total=len(configs),seconds=time.perf_counter()-t))
            print(r['id'],r['accuracy'],flush=True)
    p.KINDS=KINDS;p.OUT=OUT;p.summarize(rows,digest);p.write(OUT/'pilot_status.json',dict(updated_utc=p.now(),stage='completed',completed=len(rows),total=len(configs)))
if __name__=='__main__':main()
