"""Independent checks for the mathematical witnesses and numerical report.

Set-based morphology is independent of scipy's image morphology. This does
not prove a population ordering of representations on OptDigits.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np


def closing(active, radius):
    disk = {(i, j) for i in range(-radius, radius+1) for j in range(-radius, radius+1) if i*i+j*j <= radius*radius}
    dilated = {(i+di, j+dj) for i, j in active for di, dj in disk}
    return {(i, j) for i, j in dilated if {(i+di, j+dj) for di, dj in disk} <= dilated}


def cell_values(active):
    vertices, edges = set(), set()
    for i, j in active:
        corners = [(i, j), (i+1, j), (i+1, j+1), (i, j+1)]
        vertices.update(corners)
        edges.update(tuple(sorted([corners[k], corners[(k+1) % 4]])) for k in range(4))
    remaining, components = set(active), 0
    while remaining:
        stack = [remaining.pop()]; components += 1
        while stack:
            i, j = stack.pop()
            neighbors = {(i+di, j+dj) for di in [-1, 0, 1] for dj in [-1, 0, 1]} & remaining
            remaining -= neighbors; stack.extend(neighbors)
    area = len(active)
    perimeter = sum((i+di, j+dj) not in active for i, j in active for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)])
    euler = len(vertices) - len(edges) + area
    return components, components-euler, area, perimeter


def entropy(p):
    p = np.asarray(p)
    p = p[p > 0]
    return float(-(p*np.log2(p)).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--search-holes', action='store_true')
    args = parser.parse_args()
    repo, directory = args.repo.resolve(), args.analysis.resolve()
    witnesses = json.loads((directory/'synthetic_witnesses.json').read_text())
    for name, pair in witnesses['pairs'].items():
        for index, patch in enumerate(pair['patches']):
            active = set(map(tuple, np.argwhere(patch)))
            top, geo = [], []
            for r in range(4):
                closed = closing(active, r)
                assert closed == active
                b0, b1, area, perimeter = cell_values(closed)
                top += [b0, b1]; geo += [area, perimeter]
            assert top == pair['topology'][index] and geo == pair['geometry'][index]
    # Explicit 508-parameter classifier realizes each informative separator.
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(repo/'outputs/tiny_topology'))
    sys.path.insert(0, str(repo/'outputs/real_topology'))
    from tiny_mlp_pilot import Tiny
    from optdigits_ablation import scale_features
    from scipy.special import softmax
    with np.load(directory/'synthetic_witnesses.npz') as z:
        images = z['images']
    top, geo = scale_features(images)
    pixels = images.reshape(-1, 8, 4, 8, 4).mean((2, 4)).reshape(-1, 64)
    arrays = dict(pixels=pixels, topology=top, geometry=geo)
    with np.load(repo/'runs/wdep-alpha1-20261011/train_features.npz') as train:
        scaled = {key: np.where(train['keep_'+key], (values-train['mean_'+key])/train['std_'+key], 0).astype(np.float32)
                  for key, values in arrays.items()}
        for branch, offset, coord, boundary, direction in [('topology', 0, 0, 1.5, 1), ('geometry', 2, 1, 7., -1)]:
            model = Tiny(64, 8, 6, 1, 'early')
            assert sum(p.numel() for p in model.parameters()) == 508
            with torch.no_grad():
                for p in model.parameters(): p.zero_()
                model.body[0].weight[0, 64+coord] = float(train['std_'+branch][coord])
                model.body[0].bias[0] = float(train['mean_'+branch][coord])
                model.head.weight[0, 0] = direction
                model.head.weight[1, 0] = -direction
                model.head.bias[0] = -direction*boundary
                model.head.bias[1] = direction*boundary
                model.head.bias[2:] = -10
                prediction = model(torch.from_numpy(scaled['pixels'][offset:offset+2]), torch.from_numpy(scaled[branch][offset:offset+2])).argmax(1).numpy()
            np.testing.assert_array_equal(prediction, [0, 1])
    # Chain-rule identity on a nontrivial joint distribution, axes X,T,G,Y.
    p = np.random.default_rng(13).random((2, 3, 4, 5)); p /= p.sum()
    def h(*axes):
        return entropy(p.sum(axis=tuple(a for a in range(4) if a not in axes)))
    def cmi(a, b, given):
        return h(*given, a)+h(*given, b)-h(*given)-h(*given, a, b)
    lhs = (h(0, 2, 3)-h(0, 2))-(h(0, 1, 3)-h(0, 1))
    rhs1 = cmi(3, 1, (0,))-cmi(3, 2, (0,))
    rhs2 = cmi(3, 1, (0, 2))-cmi(3, 2, (0, 1))
    assert max(abs(lhs-rhs1), abs(lhs-rhs2)) < 1e-12
    # A direct test of the coefficient reparameterization with our architecture.
    torch.manual_seed(13)
    net = Tiny(64, 8, 6, 1, 'early'); compensated = Tiny(64, 8, 6, 1, 'early')
    compensated.load_state_dict(net.state_dict())
    x, t = torch.randn(17, 64), torch.randn(17, 8)
    with torch.no_grad():
        compensated.body[0].weight[:, 64:] /= .3
        err = float((net(x, t)-compensated(x, .3*t)).abs().max())
    assert err < 1e-6
    hole_search = None
    if args.search_holes:
        seen, found = {}, None
        for start in range(0, 65536, 2048):
            codes = np.arange(start, start+2048, dtype=np.uint32)
            patches = ((codes[:, None] >> np.arange(16)) & 1).reshape(-1, 4, 4).astype(bool)
            images = np.zeros((len(codes), 32, 32), bool); images[:, 12:16, 12:16] = patches
            t, g = scale_features(images)
            for code, tt, gg in zip(codes, t.astype(int), g.astype(int)):
                if tt[0] != 1: continue
                key = tuple(gg)
                if key in seen and seen[key][1] != tt[1]:
                    found = [int(seen[key][0]), int(code)]; break
                seen.setdefault(key, (int(code), int(tt[1])))
            if found: break
        hole_search = dict(masks_checked=start+2048, raw_beta0=1, identical_all_geometry_required=True,
                           differing_raw_beta1_required=True, witness=found,
                           scope='Only all 4x4 patches embedded in one pooled cell; no statement for larger images.')
    findings = json.loads((directory/'real_data.json').read_text())
    for path, value in findings['sources'].items():
        assert hashlib.sha256((repo/path).read_bytes()).hexdigest() == value
    assert findings['logloss_gain']['positive_seed_means'] == 9
    assert findings['entropy_diagnostics']['combined_pixel_unique'] == 3823
    audit = dict(passed=True, set_based_morphology_and_cell_counts=True, constructed_508_parameter_classifiers_correct=True,
                 source_hashes_verified=True, information_identity_numeric_error=max(abs(lhs-rhs1), abs(lhs-rhs2)),
                 scaling_compensation_max_float32_logit_error=err,
                 mi_accuracy_counterexample=dict(erasure_mi_bits=.7, erasure_bayes_accuracy=.85,
                                                bitflip_mi_bits=1-entropy([.1,.9]), bitflip_bayes_accuracy=.9),
                 connected_hole_search=hole_search,
                 population_mutual_information_order_established=False)
    (directory/'independent_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(audit,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
