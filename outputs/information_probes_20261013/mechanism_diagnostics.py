"""Explicitly post-hoc descriptive diagnosis; no fitting or selected settings."""
import argparse
from pathlib import Path
import numpy as np
import torch
import run_study as study

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args();run=args.run
    raw=study.make_raw(run,True);out={}
    train=raw['full']['t'][:,::2];mean,std=train.mean(0),train.std(0)
    with np.load(run/'losses.npz') as losses:
        for part in ['tra','cv','full','wdep','windep']:
            t=raw[part]['t'];rare=(t[:,::2]!=1).any(1)
            result=dict(n=len(t),any_multiple_components=int(rare.sum()),
                beta0_counts_by_radius=[{str(int(k)):int(v) for k,v in zip(*np.unique(t[:,c],return_counts=True))} for c in [0,2,4,6]],
                max_abs_standardized_beta0_by_radius=np.max(np.abs((t[:,::2]-mean)/np.where(std>=1e-10,std,1)),axis=0).tolist())
            if part in ['wdep','windep']:
                diff=losses[f'mlp6_xt_{part}_bits']-losses[f'mlp6_xb1_{part}_bits']
                result['full_betti_minus_holes_loss']=dict(
                    total_bits_per_image=float(diff.mean()),
                    rare_group_mean=float(diff[:,rare].mean()) if rare.any() else None,
                    usual_group_mean=float(diff[:,~rare].mean()),
                    rare_group_contribution_to_overall_mean=float(diff[:,rare].sum()/diff.size),
                    usual_group_contribution_to_overall_mean=float(diff[:,~rare].sum()/diff.size))
                result['topology_given_geometry_by_digit']={str(c):float((losses[f'mlp6_xg_{part}_bits']-losses[f'mlp6_xgt_{part}_bits'])[:,raw[part]['y']==c].mean()) for c in range(10)}
            out[part]=result
    # Controlled intervention inside each frozen model, with all weights,
    # pixels and beta1 held fixed. This is not a validated replacement model.
    torch.set_num_threads(1)
    interventions={};predictions={}
    for part in ['wdep','windep']:
        z=study.matrix(raw,part,'xt','full');altered=z.copy()
        altered[:,[64,66,68,70]]=((1-mean)/np.where(std>=1e-10,std,1)).astype(np.float32)
        unchanged=(raw[part]['t'][:,::2]==1).all(1)
        np.testing.assert_array_equal(altered[unchanged],z[unchanged])
        before=[];after=[];logits_all=[]
        for seed in range(10,20):
            checkpoint=torch.load(run/'final'/f'mlp6_xt_s{seed}.pt',map_location='cpu',weights_only=False)
            model=study.net(6);model.load_state_dict(checkpoint['state_dict']);model.eval()
            with torch.no_grad():logits=model(torch.from_numpy(altered)).numpy()
            with np.load(run/'predictions'/f'mlp6_xt_s{seed}.npz') as a:original=a[part+'_logits']
            np.testing.assert_array_equal(original[unchanged],logits[unchanged])
            before.append(study.metrics(original,raw[part]['y']));after.append(study.metrics(logits,raw[part]['y']));logits_all.append(logits)
        predictions[part+'_logits']=np.stack(logits_all)
        interventions[part]=dict(changed_images=int((~unchanged).sum()),untouched_predictions_exact=True,
            before={k:float(np.mean([r[k] for r in before])) for k in ['bits','accuracy']},
            after={k:float(np.mean([r[k] for r in after])) for k in ['bits','accuracy']},
            bits_improvement=float(np.mean([a['bits']-b['bits'] for a,b in zip(before,after)])))
    np.savez_compressed(run/'posthoc_component_intervention.npz',**predictions)
    study.write(run/'mechanism.json',dict(status='Post-hoc diagnosis. The fixed-model intervention isolates sensitivity to beta0 inputs; it is not a causal claim about natural digits or an independently validated model improvement.',results=out,
        component_intervention=interventions,intervention='Set all four beta0 to1, retain beta1/pixels/weights/normalization. Only originally non-unit beta0 inputs change.',code_sha256=study.sha(__file__)))
    print(out)

if __name__=='__main__':main()
