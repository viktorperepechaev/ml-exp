"""Descriptive transfer check on an already-seen older TEST; never confirmatory.

Runs only after WDEP primary analysis. Uses the same frozen primary models,
without training, selection, calibration or alternative significance testing.
"""
from datetime import datetime,timezone
from pathlib import Path
import json,subprocess,time,zipfile,hashlib
import numpy as np
import torch
import tiny_confirmation as e

def main():
    planpath=e.OUT/'tiny_seen_windep_replay_plan.json'
    if not planpath.exists():
        assert not (e.OUT/'tiny_confirmation_prediction_receipt.json').exists()
        e.write(planpath,dict(registered_utc=datetime.now(timezone.utc).isoformat(),primary_wdep_test_read=False,
            scope='Descriptive reuse of old, previously evaluated writer-independent WINDЕP. It is NOT an untouched validation set and cannot rescue the primary WDEP comparison.',
            rule='Evaluate the exact two primary-selected pipelines, all10seeds, with weights fixed on TRA+CV. No model choice, calibration, refit, hypothesis test, or confidence/novelty claim from this reuse.',
            reason='Original UCI metadata calls writer-independent testing the main generalization measure. Check the new models for transfer, while explicitly retaining the prior test-use limitation.',
            code_sha256=e.sha(__file__)))
    while not (e.OUT/'tiny_confirmation_test_results.json').exists():
        st=json.loads((e.OUT/'tiny_continuation_status.json').read_text())
        if st['stage']=='failed':raise RuntimeError(st['error'])
        time.sleep(15)
    p=e.verify();assert (e.OUT/'tiny_confirmation_weights_receipt.json').exists()
    with zipfile.ZipFile(e.ROOT/'work/real_data/optdigits.zip') as z:compressed=z.read('optdigits-orig.windep.Z')
    lines=[s.strip() for s in subprocess.run(['gzip','-dc'],input=compressed,capture_output=True,check=True).stdout.decode('ascii').splitlines()]
    images=[];labels=[];i=0
    while i<len(lines):
        if len(lines[i])==32 and set(lines[i])<={'0','1'}:
            rows=lines[i:i+32];assert len(rows)==32 and all(len(s)==32 and set(s)<={'0','1'} for s in rows)
            lab=lines[i+32];assert len(lab)==1 and lab.isdigit();images.append(np.frombuffer(''.join(rows).encode(),np.uint8).reshape(32,32)==ord('1'));labels.append(int(lab));i+=33
        else:i+=1
    assert len(labels)==1797;images=np.stack(images);y=np.array(labels);arrays={}
    for label in ['topology','baseline']:
        cfg=p['pipelines'][label];first=torch.load(e.MODELS/f'{label}_s0.pt',weights_only=False);x,f=e.eval_arrays(images,cfg,first['scaling']);pr=[]
        for seed in p['seeds']:
            ck=torch.load(e.MODELS/f'{label}_s{seed}.pt',weights_only=False);net=e.model(cfg);net.load_state_dict(ck['state_dict']);net.eval()
            with torch.no_grad():prob=torch.cat([net(x[j:j+256],f[j:j+256]).softmax(1) for j in range(0,len(y),256)]).numpy()
            pr.append(prob)
        arrays[label+'_probabilities']=np.stack(pr)
    np.savez_compressed(e.OUT/'tiny_seen_windep_predictions.npz',**arrays)
    ta=(arrays['topology_probabilities'].argmax(2)==y).mean(1);ba=(arrays['baseline_probabilities'].argmax(2)==y).mean(1)
    result=dict(analyzed_utc=datetime.now(timezone.utc).isoformat(),scope='Descriptive, already-seen original WINDЕP; not another untouched confirmation test; no change to primary conclusion.',
        test_n=len(y),test_member_sha256=hashlib.sha256(compressed).hexdigest(),protocol_sha256=e.sha(e.PROTO),
        baseline_mean_accuracy=float(ba.mean()),topology_mean_accuracy=float(ta.mean()),mean_difference=float((ta-ba).mean()),
        baseline_seed_accuracy=ba.tolist(),topology_seed_accuracy=ta.tolist(),positive_pairs=int((ta>ba).sum()),
        training_or_tuning_performed=False,significance_claim=False)
    e.write(e.OUT/'tiny_seen_windep_results.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
