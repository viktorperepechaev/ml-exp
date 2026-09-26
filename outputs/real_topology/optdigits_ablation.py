"""Additional validation-only OptDigits checks; no writer-independent test access."""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json
import sys
import time
import numpy as np
from scipy import ndimage
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import torch
from optdigits_pilot import DATA, OUT, read_original, cells

torch.set_num_threads(1)


def scale_features(images):
    ts, gs = [], []
    yy, xx = np.mgrid[-1:1:32j, -1:1:32j]
    for radius in (0, 1, 2, 3):
        disk = ((np.mgrid[-radius:radius+1, -radius:radius+1]**2).sum(axis=0) <= radius*radius)
        masks = images if radius == 0 else np.stack([
            ndimage.binary_closing(np.pad(im, radius), structure=disk)[radius:-radius, radius:-radius]
            for im in images
        ])
        chi, area, perimeter = cells(masks)
        beta0 = np.array([ndimage.label(im, structure=np.ones((3, 3)))[1] for im in masks])
        holes = beta0 - chi
        # Euler profile (with redundant chi omitted) and matched geometry profile.
        ts.extend([beta0, holes])
        gs.extend([area, perimeter])
    return np.stack(ts, 1), np.stack(gs, 1)


def main():
    train_img, ty = read_original("tra")
    val_img, vy = read_original("cv")
    with np.load(DATA / "optdigits_trainval_features.npz") as z:
        tr = {k[3:]: z[k] for k in z.files if k.startswith("tr_")}
        va = {k[3:]: z[k] for k in z.files if k.startswith("va_")}
    tr["morphology_topology"], tr["morphology_geometry"] = scale_features(train_img)
    va["morphology_topology"], va["morphology_geometry"] = scale_features(val_img)
    normtr, normva = {}, {}
    for k in tr:
        s = StandardScaler().fit(tr[k])
        normtr[k], normva[k] = s.transform(tr[k]), s.transform(va[k])
    matched_only = "--matched-only" in sys.argv
    padded_only = "--padded-morph-only" in sys.argv
    rows = json.loads((OUT / "optdigits_ablation_validation.json").read_text())["runs"] if (matched_only or padded_only) else []
    if padded_only:
        rows = [r for r in rows if not r["branch"].startswith("morphology_")]
    def save():
        (OUT / "optdigits_ablation_validation.json").write_text(json.dumps({"test_read":False,"morphology_operation":"zero-padding radius before disk closing, then crop; no artificial frame erosion","runs":rows}, indent=2))
    # Does topology help an already strong geometric representation?
    for base in (() if (matched_only or padded_only) else ("pixels", "pixels_gradients")):
        bx, bv = normtr["pixels"], normva["pixels"]
        if base == "pixels_gradients":
            bx = np.column_stack([bx, .3 * normtr["gradients"]])
            bv = np.column_stack([bv, .3 * normva["gradients"]])
        for branch in ("none", "global_topology", "global_geometry", "morphology_topology", "morphology_geometry"):
            for weight in ([0.] if branch=="none" else [.1,.3,1.]):
                x = bx if branch=="none" else np.column_stack([bx,weight*normtr[branch]])
                v = bv if branch=="none" else np.column_stack([bv,weight*normva[branch]])
                for c in (1.,10.,100.):
                    for gamma in (.003,.01,.03):
                        start=time.perf_counter()
                        clf=SVC(C=c,gamma=gamma).fit(x,ty)
                        pred=clf.predict(v)
                        rows.append(dict(model="rbf_svm",base=base,branch=branch,weight=weight,C=c,gamma=gamma,accuracy=float((pred==vy).mean()),errors=int((pred!=vy).sum()),seconds=time.perf_counter()-start))
                        save()
            print(json.dumps(max([r for r in rows if r["base"]==base and r["branch"]==branch],key=lambda r:r["accuracy"])),flush=True)
    # Small network check; all branches receive the same training schedule.
    for branch in (("morphology_topology", "morphology_geometry") if padded_only else (("none", "gradients") if matched_only else ("none","euler","perimeter","gradients","morphology_topology","morphology_geometry"))):
        bx,bv=normtr["pixels"],normva["pixels"]
        if branch!="none":
            bx=np.column_stack([bx,.3*normtr[branch]])
            bv=np.column_stack([bv,.3*normva[branch]])
        for width in ((32,) if padded_only else ((36 if branch == "none" else 12,) if matched_only else (16,32,64))):
            for seed in (0,1,2):
                torch.manual_seed(seed)
                x,v=torch.tensor(bx,dtype=torch.float32),torch.tensor(bv,dtype=torch.float32)
                y=torch.tensor(ty,dtype=torch.long)
                net=torch.nn.Sequential(torch.nn.Linear(x.shape[1],width),torch.nn.ReLU(),torch.nn.Linear(width,10))
                optimizer=torch.optim.AdamW(net.parameters(),lr=.01,weight_decay=.01)
                best,best_epoch,wait=0.,0,0
                start=time.perf_counter()
                for epoch in range(250):
                    net.train()
                    for batch in torch.randperm(len(x)).split(256):
                        optimizer.zero_grad()
                        loss=torch.nn.functional.cross_entropy(net(x[batch]),y[batch])
                        loss.backward();optimizer.step()
                    net.eval()
                    with torch.no_grad():
                        accuracy=float((net(v).argmax(1).numpy()==vy).mean())
                    if accuracy>best:
                        best,best_epoch,wait=accuracy,epoch,0
                    else:wait+=1
                    if wait>=35:break
                rows.append(dict(model="mlp",base="pixels",branch=branch,width=width,seed=seed,parameters=sum(p.numel() for p in net.parameters()),accuracy=best,selected_epoch=best_epoch,seconds=time.perf_counter()-start))
                save()
            rs=[r for r in rows if r["model"]=="mlp" and r["branch"]==branch and r["width"]==width]
            print(json.dumps(dict(model="mlp",branch=branch,width=width,mean=float(np.mean([r["accuracy"] for r in rs])),min=float(np.min([r["accuracy"] for r in rs])),max=float(np.max([r["accuracy"] for r in rs])))),flush=True)
    save()


if __name__=="__main__":main()
