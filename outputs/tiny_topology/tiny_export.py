"""Optional exact-affine scaler folding for the selected tiny MLPs.

No optimization, tuning, test selection or changes to the frozen experiment.
Exports seed0 by declaration, never the best-performing test seed.
"""
from pathlib import Path
from datetime import datetime,timezone
import json,sys
import numpy as np
import torch
import tiny_confirmation as e
OUT=e.OUT;EXPORT=OUT/'deployable_weights'


def factors(sc,names):
    gain=[];shift=[]
    for name in names:
        q=sc[name];c=1. if name=='pixels' else .3
        a=np.where(q['keep'],c/q['std'],0)
        gain.extend(a);shift.extend(-q['mean']*a)
    return np.asarray(gain),np.asarray(shift)


def fold_affine(layer,gain,shift,start=0):
    n=len(gain);w=layer.weight.detach().double().numpy().copy();b=layer.bias.detach().double().numpy().copy()
    selected=w[:,start:start+n].copy();w[:,start:start+n]=selected*gain[None,:];b+=selected@shift
    with torch.no_grad():layer.weight.copy_(torch.from_numpy(w).to(layer.weight));layer.bias.copy_(torch.from_numpy(b).to(layer.bias))


def raw_inputs(images,cfg):
    branch=e.mlp.BRANCHES[cfg['branch']];x=images.reshape(-1,8,4,8,4).mean((2,4)).reshape(len(images),64)
    if branch['base']=='hog':x=np.column_stack([x,e.gradient_histograms(images)])
    if branch['side']:
        top,geo=e.scale_features(images);f=top if branch['side']=='topology' else geo
    else:f=np.empty((len(images),0))
    return torch.from_numpy(x.astype(np.float32)),torch.from_numpy(f.astype(np.float32))


def main():
    p=e.verify();assert (OUT/'tiny_confirmation_test_results.json').exists()
    EXPORT.mkdir(exist_ok=True);rows=[]
    # Same development images as an independent numerical equivalence check.
    images,_=e.read_original('cv')
    for label in ['topology','baseline']:
        cfg=p['pipelines'][label]
        if cfg['model_family']!='mlp':
            rows.append(dict(pipeline=label,status='not_exported',reason='This utility only exports MLPs'));continue
        ck=torch.load(e.MODELS/f'{label}_s0.pt',weights_only=False);reference=e.model(cfg);reference.load_state_dict(ck['state_dict']);net=e.model(cfg);net.load_state_dict(ck['state_dict']);sc=ck['scaling']
        branch=e.mlp.BRANCHES[cfg['branch']];names=['pixels']+(['gradients'] if branch['base']=='hog' else []);side=['global_'+branch['side']] if branch['side'] else []
        if cfg['fusion']=='early':
            gain,shift=factors(sc,names+side);fold_affine(net.body[0],gain,shift)
        else:
            gain,shift=factors(sc,names);fold_affine(net.body[0],gain,shift)
            gain,shift=factors(sc,side);fold_affine(net.head,gain,shift,start=cfg['hidden_width'])
        x,f=e.eval_arrays(images,cfg,sc);rx,rf=raw_inputs(images,cfg);reference.eval();net.eval()
        with torch.no_grad():a=reference(x,f);b=net(rx,rf);pa=a.softmax(1).numpy();pb=b.softmax(1).numpy()
        np.testing.assert_allclose(pa,pb,atol=2e-5,rtol=2e-4);np.testing.assert_array_equal(pa.argmax(1),pb.argmax(1))
        dest=EXPORT/(label+'_seed0_raw_features.npz');np.savez_compressed(dest,**{k:v.detach().numpy() for k,v in net.state_dict().items()})
        rows.append(dict(pipeline=label,seed=0,selection='Seed0 fixed, no test-seed selection',file=dest.name,sha256=e.sha(dest),architecture=cfg,
            numerical_float32_weight_bytes=sum(v.numel()*v.element_size() for v in net.state_dict().values()),stored_scalar_count=sum(v.numel() for v in net.state_dict().values()),
            independent_scaler_storage_required=False,reference_probability_max_abs_difference=float(np.abs(pa-pb).max()),all946_development_argmaxes_identical=True,
            extraction='Raw64 block means + raw8Betti/geometry profile; no per-feature normalization or branch0.3 scaling at runtime, absorbed in first affine map.' if cfg['base_dim']==64 else 'Raw feature groups in training order; scalers and0.3 multipliers absorbed in affine maps.'))
    receipt=dict(created_utc=datetime.now(timezone.utc).isoformat(),purpose='Numerically checked deployment export, not another experiment',protocol_sha256=e.sha(e.PROTO),models=rows,
        limitation='Descriptor extraction code, morphology and component-labeling workspace remain additional. Numerical weight bytes are not whole-program memory or a latency claim.')
    e.write(OUT/'tiny_export_receipt.json',receipt);print(json.dumps(receipt,indent=2),flush=True)

if __name__=='__main__':main()
