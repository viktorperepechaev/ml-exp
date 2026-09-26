"""Replay every selected checkpoint and verify all 102 development runs."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/learned_topology_matplotlib')
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import torch
from euler_layer import SmallClassifier
from single_field import SingleFieldClassifier

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
WORK = ROOT / 'work/learned_topology'
torch.set_num_threads(1)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    with np.load(WORK / 'train_cv.npz') as z:
        a = {k: torch.from_numpy(z[k].copy()) for k in z.files}
    checked, max_prob_error, max_ce_error = [], 0., 0.
    for prefix, cls, jobs in [('', SmallClassifier, 'jobs'),
                              ('single_field/', SingleFieldClassifier, 'single_field_jobs')]:
        protocol_path = OUT / prefix / 'pilot_protocol.json'
        protocol = json.loads(protocol_path.read_text())
        for name, digest in protocol['source_sha256'].items():
            assert sha(OUT / name) == digest, name
        if 'cache_sha256' in protocol:
            assert sha(WORK / 'train_cv.npz') == protocol['cache_sha256']
        results = json.loads((OUT / prefix / 'pilot_results.json').read_text())
        assert results['completed'] == results['total'] == protocol['fits']
        assert {r['id'] for r in results['runs']} == {c['id'] for c in protocol['configs']}
        for r in results['runs']:
            base = WORK / jobs / r['id']
            ckpt = torch.load(base.with_suffix('.pt'), map_location='cpu', weights_only=True)
            assert ckpt['protocol_sha256'] == r['protocol_sha256'] == sha(protocol_path)
            net = cls(r['kind'], a['mean'], a['std'], a['keep'])
            net.load_state_dict(ckpt['state_dict']); net.eval()
            assert net.trainable_parameters == r['trainable_parameters']
            assert net.stored_parameters == r['stored_parameters'] <= 512
            with torch.no_grad():
                logits = net(a['validation_x'])
                prob = logits.softmax(1).numpy()
                ce = float(torch.nn.functional.cross_entropy(logits, a['validation_y']))
            with np.load(base.with_suffix('.npz')) as z:
                np.testing.assert_allclose(prob, z['probabilities'], atol=1e-7, rtol=1e-6)
                np.testing.assert_array_equal(prob.argmax(1), z['predictions'])
                max_prob_error = max(max_prob_error, float(np.max(np.abs(prob-z['probabilities']))))
            acc = float((prob.argmax(1) == a['validation_y'].numpy()).mean())
            assert acc == r['accuracy']
            assert abs(ce-r['ce']) < 1e-7
            max_ce_error = max(max_ce_error, abs(ce-r['ce']))
            assert np.argmin(np.array(r['curve'])[:, 0])+1 == r['selected_epoch']
            for name, value in r['selected_front_parameter_change_norms'].items():
                actual = float((dict(net.named_parameters())[name].detach()-ckpt['initial_parameters'][name]).norm())
                assert abs(actual-value) < 1e-7
            if r['kind'] == 'euler_frozen':
                assert all(v == 0 for v in r['selected_front_parameter_change_norms'].values())
                assert r['selected_feature_probe_rms_change'] == 0
            elif r['kind'] == 'euler_learned':
                assert all(v > 0 for v in r['first_batch_front_gradient_norms'].values())
                assert all(v > 0 for v in r['selected_front_parameter_change_norms'].values())
                assert r['selected_feature_probe_rms_change'] > 0
            checked.append({'id': r['id'], 'accuracy': acc,
                            'hardcap_unresolved': r['hardcap_convergence_unresolved']})
    assert len(checked) == 102
    # Learned/frozen ablations start from exactly identical parameter tensors.
    for jobs, prefix in [('jobs', ''), ('single_field_jobs', 'single_')]:
        for oi in range(2):
            for seed in range(3):
                states = [torch.load(WORK/jobs/f'{prefix}euler_{kind}_o{oi}_s{seed}.pt',
                                     map_location='cpu', weights_only=True)['initial_parameters']
                          for kind in ['learned', 'frozen']]
                assert states[0].keys() == states[1].keys()
                assert all(torch.equal(states[0][k], states[1][k]) for k in states[0])
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), passed=True,
                  runs_replayed=len(checked), max_probability_error=max_prob_error,
                  max_cross_entropy_error=max_ce_error,
                  train_size=len(a['train_y']), validation_size=len(a['validation_y']),
                  parameter_caps_and_source_hashes_verified=True,
                  learned_frozen_initializations_identical=True,
                  all_learned_euler_runs_have_gradient_and_parameter_changes=True,
                  all_frozen_euler_fronts_unchanged=True,
                  hardcap_unresolved=sum(r['hardcap_unresolved'] for r in checked),
                  test_read=False, runs=checked)
    (OUT/'results_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    plot()
    print(json.dumps({k: v for k, v in report.items() if k != 'runs'}, indent=2))


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    main = json.loads((OUT/'pilot_summary.json').read_text())['selected']
    single = json.loads((OUT/'single_field/pilot_summary.json').read_text())['selected']
    names = ['MLP', 'Frozen Euler', 'Learned Euler', 'Area', 'Area + perimeter', 'CNN (2 channels)']
    rows = [main['mlp'], single['euler_frozen'], single['euler_learned'],
            single['area_learned'], single['geometry_learned'], main['cnn_c2']]
    fig, ax = plt.subplots(figsize=(9.8, 4.8), constrained_layout=True)
    colors = ['#718096', '#718096', '#137c83', '#718096', '#718096', '#394f86']
    for i, (r, c) in enumerate(zip(rows, colors)):
        values = np.array(r['seed_accuracy'])*100
        ax.scatter(values, np.full(3, i)+np.array([-.12,0,.12]), color=c, s=25, alpha=.65)
        ax.scatter(values.mean(), i, color=c, marker='D', s=65)
        ax.text(97.4, i, f"{values.mean():.2f}% | {r['trainable_parameters']} weights", va='center', fontsize=10)
    ax.set_yticks(range(6), names); ax.invert_yaxis(); ax.set_xlim(93.5,100)
    ax.set_xticks([94,95,96,97]); ax.set_xlabel('Validation accuracy (%)')
    ax.set_title('OptDigits | same 8 × 8 input | at most 512 parameters\nDots: 3 seeds; diamonds: means. Configuration selected on this validation set.', fontsize=12)
    ax.grid(axis='x', alpha=.2); ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUT/'validation_comparison.png', dpi=180); plt.close(fig)
    ckpt = torch.load(WORK/'single_field_jobs/single_euler_learned_o0_s0.pt',
                      map_location='cpu', weights_only=True)
    before, after = ckpt['initial_parameters'], ckpt['state_dict']
    fig, axes = plt.subplots(1,3,figsize=(10,3),constrained_layout=True)
    lo = min(float(before['front.conv.weight'].min()), float(after['front.conv.weight'].min()))
    hi = max(float(before['front.conv.weight'].max()), float(after['front.conv.weight'].max()))
    for ax, state, title in zip(axes[:2], [before,after], ['Initial 3 × 3 filter', 'Learned 3 × 3 filter']):
        values = state['front.conv.weight'][0,0].numpy()
        ax.imshow(values, cmap='coolwarm', vmin=lo, vmax=hi)
        for (i,j), v in np.ndenumerate(values): ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=10)
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    for k in range(4):
        axes[2].plot([0,1],[float(before['front.thresholds'][0,k]),float(after['front.thresholds'][0,k])],marker='o',label=f'Level {k+1}')
    axes[2].set_xticks([0,1],['Initial','Learned']); axes[2].set_ylabel('Threshold'); axes[2].set_title('Thresholds learn, too')
    axes[2].grid(alpha=.2)
    fig.suptitle('Fixed illustration: selected single-field configuration, seed 0',fontsize=11)
    fig.savefig(OUT/'learned_parameters.png',dpi=180); plt.close(fig)


if __name__ == '__main__': main()
