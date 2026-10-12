"""Read-only information-theoretic diagnostics of the archived OptDigits run.

No new models or hyperparameters are fitted. Empirical entropies are explicitly
finite-sample diagnostics, not estimates asserted to equal population values.
Synthetic witnesses use the exact four-scale feature extractor on 32x32 images.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/topological-invariants-matplotlib')
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.special import logsumexp


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def entropy(values):
    _, counts = np.unique(values, axis=0, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def discrete_summary(features, labels, permutations=200):
    _, inverse, counts = np.unique(features, axis=0, return_inverse=True, return_counts=True)
    n, m = len(labels), len(counts)
    py = np.bincount(labels, minlength=10) / n
    hy = float(-(py[py > 0] * np.log2(py[py > 0])).sum())
    hz = float(-((counts / n) * np.log2(counts / n)).sum())
    def hy_given(zlabels):
        joint = np.bincount(inverse * 10 + zlabels, minlength=m * 10).reshape(m, 10)
        ii, jj = np.nonzero(joint)
        return float(-(joint[ii, jj] / n * np.log2(joint[ii, jj] / counts[ii])).sum())
    conditional = hy_given(labels)
    rng = np.random.default_rng(20261013)
    null = np.array([hy - hy_given(rng.permutation(labels)) for _ in range(permutations)])
    return dict(n=n, distinct=m, singleton_samples=int(counts[counts == 1].sum()),
                empirical_h_feature_bits=hz, empirical_h_label_bits=hy,
                empirical_h_label_given_feature_bits=conditional,
                empirical_mi_bits=hy - conditional,
                shuffled_label_mi_mean_bits=float(null.mean()),
                shuffled_label_mi_sd_bits=float(null.std(ddof=1)),
                label_permutations=permutations,
                warning='Plug-in entropy under the empirical distribution only; high-cardinality features can memorize labels.')


def real_data(repo, out):
    run = repo / 'runs/wdep-alpha1-20261011'
    result = json.loads((run / 'result.json').read_text())
    receipt = json.loads((run / 'prediction_receipt.json').read_text())
    assert receipt['predictions_sha256'] == sha(run / 'predictions.npz')
    assert result['side_scale'] == 1. and result['models'] == 20
    with np.load(run / 'train_features.npz') as train, np.load(run / 'wdep_features.npz') as test:
        summary = {}
        for split, values, label_key in [('tra_cv', train, 'train_y'), ('wdep', test, 'test_y')]:
            labels = values[label_key]
            summary[split] = {name: discrete_summary(values['raw_' + name], labels)
                              for name in ['pixels', 'geometry', 'topology']}
        all_pixels = np.concatenate([train['raw_pixels'], test['raw_pixels']])
        summary['combined_pixel_unique'] = len(np.unique(all_pixels, axis=0))
        # Training-only class frequencies give a fixed null predictor on WDEP.
        prior = np.bincount(train['train_y'], minlength=10) / len(train['train_y'])
        y = test['test_y'].copy()
        null_bits = -np.log2(prior[y])
    losses, performance = {}, {}
    with np.load(run / 'predictions.npz') as predictions:
        for branch in ['geometry', 'topology']:
            logits = predictions[branch + '_logits'].astype(np.float64)
            nll = logsumexp(logits, axis=2) - np.take_along_axis(logits, y[None, :, None], axis=2)[:, :, 0]
            losses[branch] = nll / np.log(2)
            seed_ce = nll.mean(1)
            np.testing.assert_allclose(seed_ce, result['summaries'][branch]['seed_ce'], atol=5e-8, rtol=0)
            performance[branch] = dict(mean_ce_nats=float(nll.mean()), mean_logloss_bits=float(losses[branch].mean()),
                                       seed_logloss_bits=losses[branch].mean(1).tolist(),
                                       bits_saved_against_train_prior=float(null_bits.mean() - losses[branch].mean()),
                                       mean_accuracy=result['summaries'][branch]['mean_accuracy'])
    delta = losses['geometry'] - losses['topology']
    gain = dict(mean_bits_per_image=float(delta.mean()), seed_mean_bits=delta.mean(1).tolist(),
                positive_seed_means=int((delta.mean(1) > 0).sum()),
                total_bits_per_943_labels_mean_model=float(delta.mean() * len(y)),
                null_predictor_logloss_bits=float(null_bits.mean()),
                meaning='Reduction in achieved conditional label log-loss; not the difference of Shannon mutual informations or certified optimal V-information.')
    per_class = {str(c): dict(n=int((y == c).sum()), mean_bits_saved=float(delta[:, y == c].mean())) for c in range(10)}
    np.savez_compressed(out / 'logloss_bits.npz', geometry=losses['geometry'], topology=losses['topology'], difference=delta, labels=y)
    findings = dict(entropy_diagnostics=summary, model_performance=performance, logloss_gain=gain, per_class=per_class,
                    computation='Stable float64 logsumexp of saved float32 logits; archived CE validated to 5e-8 nats.',
                    sources={str(p.relative_to(repo)): sha(p) for p in [run/'result.json', run/'train_features.npz', run/'wdep_features.npz', run/'predictions.npz']})
    write(out / 'real_data.json', findings)
    return findings


def synthetic_witnesses(repo, out):
    sys.path.insert(0, str(repo / 'outputs/real_topology'))
    from optdigits_ablation import scale_features
    by_geo, by_top, found = {}, {}, {}
    started = time.perf_counter()
    for start in range(0, 65536, 1024):
        codes = np.arange(start, min(start + 1024, 65536), dtype=np.uint32)
        patches = ((codes[:, None] >> np.arange(16)) & 1).reshape(-1, 4, 4).astype(bool)
        images = np.zeros((len(codes), 32, 32), dtype=bool)
        images[:, 12:16, 12:16] = patches
        top, geo = scale_features(images)
        counts = patches.sum((1, 2))
        for code, t, g, count in zip(codes, top.astype(int), geo.astype(int), counts):
            tg, tt = (int(count), *g.tolist()), (int(count), *t.tolist())
            if 'geometry_collision' not in found and tg in by_geo:
                prior_code, prior_t = by_geo[tg]
                if tuple(t) != prior_t:
                    found['geometry_collision'] = [int(prior_code), int(code)]
            if 'topology_collision' not in found and tt in by_top:
                prior_code, prior_g = by_top[tt]
                if tuple(g) != prior_g:
                    found['topology_collision'] = [int(prior_code), int(code)]
            by_geo.setdefault(tg, (int(code), tuple(t)))
            by_top.setdefault(tt, (int(code), tuple(g)))
        if len(found) == 2:
            break
    assert len(found) == 2, 'No complete pair of witnesses found in all 4x4 masks'
    result = dict(searched_masks=start + len(codes), seconds=time.perf_counter() - started,
                  embedding='4x4 patch at rows/columns 12..15 in a zero 32x32 image; bit k is patch[k//4,k%4].',
                  feature_extractor_sha256=sha(repo / 'outputs/real_topology/optdigits_ablation.py'), pairs={})
    images_all = []
    for name, codes in found.items():
        patches = ((np.asarray(codes, dtype=np.uint32)[:, None] >> np.arange(16)) & 1).reshape(2, 4, 4).astype(bool)
        images = np.zeros((2, 32, 32), dtype=bool); images[:, 12:16, 12:16] = patches
        t, g = scale_features(images)
        x = images.reshape(-1, 8, 4, 8, 4).mean((2, 4)).reshape(-1, 64)
        np.testing.assert_array_equal(x[0], x[1])
        if name == 'geometry_collision':
            np.testing.assert_array_equal(g[0], g[1]); assert not np.array_equal(t[0], t[1])
        else:
            np.testing.assert_array_equal(t[0], t[1]); assert not np.array_equal(g[0], g[1])
        result['pairs'][name] = dict(codes=codes, patches=patches.astype(int).tolist(),
                                    topology=t.astype(int).tolist(), geometry=g.astype(int).tolist(),
                                    nonzero_pooled_pixel=dict(index=27, value=float(x[0, 27])),
                                    pooled_pixels_equal=True)
        images_all.extend(images)
    np.savez_compressed(out / 'synthetic_witnesses.npz', images=np.stack(images_all))
    write(out / 'synthetic_witnesses.json', result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(8, 7), layout='constrained')
    for row, (name, pair) in enumerate(result['pairs'].items()):
        for col in range(2):
            axes[row, col].imshow(pair['patches'][col], cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            axes[row, col].set_xticks(np.arange(-.5, 4, 1), minor=True)
            axes[row, col].set_yticks(np.arange(-.5, 4, 1), minor=True)
            axes[row, col].grid(which='minor', color='gray', linewidth=.7)
            axes[row, col].tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
            axes[row, col].set_title(f"{name.replace('_', ' ')} / {'A' if col == 0 else 'B'}\nT={pair['topology'][col]}\nG={pair['geometry'][col]}", fontsize=10)
    fig.suptitle('Exact four-scale witnesses; identical pooled pixels in each pair\nEach patch is embedded in a zero 32×32 image', fontsize=12)
    fig.savefig(out / 'synthetic_witnesses.png', dpi=180)
    fig.savefig(out / 'synthetic_witnesses.svg')
    plt.close(fig)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo, out = args.repo.resolve(), args.output.resolve()
    if out.exists():
        parser.error(f'Output already exists: {out}')
    out.mkdir(parents=True)
    real = real_data(repo, out)
    print(json.dumps(dict(real_data=real), ensure_ascii=False), flush=True)
    synthetic = synthetic_witnesses(repo, out)
    print(json.dumps(dict(synthetic=synthetic), ensure_ascii=False), flush=True)
    write(out / 'manifest.json', dict(runner_sha256=sha(Path(__file__)), source_root=str(repo),
                                     files={p.name: sha(p) for p in out.iterdir() if p.is_file()}))


if __name__ == '__main__':
    main()
