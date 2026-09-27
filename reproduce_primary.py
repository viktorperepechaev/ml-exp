"""Retrain the frozen 508-weight OptDigits comparison in a fresh run directory.

Reuses the archived model, feature extractor and optimizer loop unchanged.
This repeats the selected comparison, not the original architecture search.
"""
from __future__ import annotations

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'outputs/tiny_topology'))
import tiny_confirmation as experiment

torch.set_num_threads(1)
LABELS = ('topology', 'baseline')


def fit(run_dir, label, config, seed, digest):
    # Spawned workers have fresh module globals; set their paths explicitly.
    experiment.TRAIN = Path(run_dir) / 'train_features.npz'
    experiment.MODELS = Path(run_dir) / 'models'
    return experiment.train_one(label, config, seed, digest)


def prepare_features(run):
    ti, ty = experiment.read_original('tra')
    vi, vy = experiment.read_original('cv')
    images = np.concatenate([ti, vi])
    labels = np.concatenate([ty, vy])
    topology, geometry = experiment.scale_features(images)
    arrays = dict(labels=labels,
                  mlp_pixels=images.reshape(-1, 8, 4, 8, 4).mean((2, 4)).reshape(-1, 64),
                  mlp_global_topology=topology, mlp_global_geometry=geometry)
    assert len(labels) == 2880
    # Preserve original float64 raw features, before training-only scaling.
    for key in ['mlp_pixels', 'mlp_global_topology', 'mlp_global_geometry']:
        assert arrays[key].dtype == np.float64
    with np.load(ROOT / 'work/real_data/optdigits_complement_trainval_v1.npz') as old:
        for key in ['pixels', 'global_topology', 'global_geometry']:
            np.testing.assert_array_equal(arrays['mlp_' + key],
                                          np.concatenate([old['tr_' + key], old['va_' + key]]))
    np.savez_compressed(run / 'train_features.npz', **arrays)


def read_test(run, configs, seeds, digest):
    receipt = json.loads((run / 'weights_receipt.json').read_text())
    expected = {f'{label}_s{seed}.pt' for label in configs for seed in seeds}
    assert receipt['run_protocol_sha256'] == digest
    assert set(receipt['weights']) == expected
    for name, weight_hash in receipt['weights'].items():
        assert experiment.sha(run / 'models' / name) == weight_hash
    with zipfile.ZipFile(ROOT / 'work/real_data/optdigits.zip') as archive:
        compressed = archive.read('optdigits-orig.wdep.Z')
    raw = subprocess.run(['gzip', '-dc'], input=compressed,
                         capture_output=True, check=True).stdout.decode('ascii')
    lines = [line.strip() for line in raw.splitlines()]
    images, labels, i = [], [], 0
    while i < len(lines):
        if len(lines[i]) == 32 and set(lines[i]) <= {'0', '1'}:
            rows = lines[i:i+32]
            assert len(rows) == 32 and all(len(row) == 32 and set(row) <= {'0', '1'} for row in rows)
            label = lines[i+32]
            assert len(label) == 1 and label.isdigit()
            images.append(np.frombuffer(''.join(rows).encode(), np.uint8).reshape(32, 32) == ord('1'))
            labels.append(int(label))
            i += 33
        else:
            i += 1
    assert len(labels) == 943
    return np.stack(images), np.array(labels)


def predict(run, configs, seeds, images, digest):
    arrays = {}
    for label, config in configs.items():
        first = torch.load(run / 'models' / f'{label}_s0.pt', weights_only=False, map_location='cpu')
        x, features = experiment.eval_arrays(images, config, first['scaling'])
        probabilities = []
        for seed in seeds:
            saved = torch.load(run / 'models' / f'{label}_s{seed}.pt', weights_only=False, map_location='cpu')
            assert saved['protocol_sha256'] == digest
            assert saved['seed'] == seed and saved['config'] == config
            for key in first['scaling']:
                for field in first['scaling'][key]:
                    np.testing.assert_array_equal(saved['scaling'][key][field], first['scaling'][key][field])
            net = experiment.model(config)
            net.load_state_dict(saved['state_dict']); net.eval()
            with torch.no_grad():
                prob = torch.cat([net(x[i:i+256], features[i:i+256]).softmax(1)
                                  for i in range(0, len(x), 256)]).numpy()
            probabilities.append(prob)
        arrays[label + '_probabilities'] = np.stack(probabilities)
        arrays[label + '_predictions'] = arrays[label + '_probabilities'].argmax(2)
    path = run / 'predictions.npz'
    np.savez_compressed(path, **arrays)
    experiment.write(run / 'prediction_receipt.json', dict(
        predicted_utc=experiment.now(), run_protocol_sha256=digest,
        predictions_sha256=experiment.sha(path), metrics_computed=False,
        note='Fixed-format parser reads labels beside images; predictions do not use labels.'))


def analyze(run, labels, original_protocol, started):
    receipt = json.loads((run / 'prediction_receipt.json').read_text())
    assert receipt['predictions_sha256'] == experiment.sha(run / 'predictions.npz')
    with np.load(run / 'predictions.npz') as arrays:
        top = arrays['topology_predictions'] == labels
        geo = arrays['baseline_predictions'] == labels
    difference = top.astype(int) - geo.astype(int)
    p_value = experiment.exact_signflip(difference.sum(0))
    rng = np.random.default_rng(original_protocol['rule']['statistics']['seed'])
    patterns, counts = np.unique(difference.T, axis=0, return_counts=True)
    seeds, samples = difference.shape
    bootstrap = []
    for i in range(0, 20000, 256):
        count = min(256, 20000-i)
        image_weights = rng.multinomial(samples, counts/samples, size=count)
        seed_weights = rng.multinomial(seeds, np.ones(seeds)/seeds, size=count)
        bootstrap.extend(((image_weights @ patterns)*seed_weights).sum(1)/(seeds*samples))
    ci = np.quantile(bootstrap, [.025, .975])
    result = dict(
        completed_utc=experiment.now(), topology_accuracy=float(top.mean()),
        geometry_accuracy=float(geo.mean()), difference_percentage_points=float(100*difference.mean()),
        relative_error_reduction=float(1-(1-top.mean())/(1-geo.mean())) if geo.mean() < 1 else None,
        topology_seed_accuracy=top.mean(1).tolist(), geometry_seed_accuracy=geo.mean(1).tolist(),
        positive_seed_pairs=int((difference.mean(1) > 0).sum()),
        bootstrap95_percentage_points=(100*ci).tolist(), exact_image_signflip_p=p_value,
        historical_series_bonferroni6_p=min(1., 6*p_value),
        train_n=2880, test_n=943, trained_models=20, trainable_parameters_per_model=508,
        elapsed_seconds=time.perf_counter()-started,
        scope='Refit reproduction of frozen selected configurations; no repeated architecture search, '
              'no new independent confirmation, no ensemble. WDEP writers overlap training writers.')
    # Historical predictions are consulted only after new training and metrics.
    with np.load(ROOT / 'outputs/tiny_topology/tiny_confirmation_predictions.npz') as old, \
            np.load(run / 'predictions.npz') as fresh:
        result['historical_comparison'] = {
            label: dict(predictions_equal=bool(np.array_equal(fresh[label+'_predictions'], old[label+'_predictions'])),
                        max_probability_difference=float(np.max(np.abs(fresh[label+'_probabilities']-old[label+'_probabilities']))))
            for label in LABELS}
    experiment.write(run / 'result.json', result)
    report = (
        '# Повторное обучение основного сравнения\n\n'
        'Заново обучены 20 моделей: две заранее выбранные конфигурации, по 10 инициализаций. '
        'Обучение — TRA+CV (2880), тест — WDEP (943), 508 обучаемых параметров в каждой сети.\n\n'
        '| Модель | Средняя точность |\n|---|---:|\n'
        f'| Топологические признаки | {100*top.mean():.3f}% |\n'
        f'| Геометрические признаки | {100*geo.mean():.3f}% |\n\n'
        f'Разность: {100*difference.mean():+.3f} п.п.; '
        f'положительных пар: {result["positive_seed_pairs"]}/10.\n\n'
        'Это воспроизведение выбранного сравнения, без повторного поиска архитектур. '
        'Тест уже использовался ранее; нового независимого подтверждения здесь нет. '
        'Усреднены точности отдельных сетей, а не предсказания ансамбля.\n\n'
        'Полные метрики и сравнение со старым запуском — в `result.json`; '
        'веса — в `models/`, новые предсказания — в `predictions.npz`.\n')
    (run / 'report_ru.md').write_text(report)
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4, help='parallel CPU training processes (default: 4)')
    parser.add_argument('--output', type=Path, help='new, non-existing output directory; default: runs/primary-TIMESTAMP')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be at least 1')
    original = experiment.verify()
    configs = {label: original['pipelines'][label] for label in LABELS}
    seeds = original['seeds']
    assert seeds == list(range(10))
    assert all(c['model_family'] == 'mlp' and c['parameters'] == 508 for c in configs.values())
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run = (args.output or ROOT / 'runs' / f'primary-{timestamp}').resolve()
    if run.exists():
        parser.error(f'Output already exists; choose a new directory: {run}')
    run.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    print(f'New run: {run}', flush=True)
    print('Recomputing training features from raw TRA/CV images...', flush=True)
    prepare_features(run)
    protocol = dict(created_utc=experiment.now(), frozen_source_protocol_sha256=experiment.sha(experiment.PROTO),
                    runner_sha256=experiment.sha(__file__), data_archive_sha256=original['data_sha256'],
                    train_features_sha256=experiment.sha(run / 'train_features.npz'),
                    configs=configs, seeds=seeds, workers=args.workers,
                    python=sys.version, numpy=np.__version__, torch=torch.__version__,
                    purpose='Fresh refit of the frozen primary pair only; no stored weights used for training.')
    experiment.write(run / 'run_protocol.json', protocol)
    digest = experiment.sha(run / 'run_protocol.json')
    (run / 'models').mkdir()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
        futures = [pool.submit(fit, str(run), label, config, seed, digest)
                   for label, config in configs.items() for seed in seeds]
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            print(f'{i}/20 trained: {result["name"]} ({result["epochs"]} epochs)', flush=True)
    weights = {p.name: experiment.sha(p) for p in (run / 'models').glob('*.pt')}
    assert len(weights) == 20
    experiment.write(run / 'weights_receipt.json', dict(
        frozen_utc=experiment.now(), run_protocol_sha256=digest, weights=weights, test_read=False))
    print('All 20 new models frozen. Predicting WDEP, then calculating metrics...', flush=True)
    images, labels = read_test(run, configs, seeds, digest)
    predict(run, configs, seeds, images, digest)
    analyze(run, labels, original, started)
    print(f'Results saved to: {run}', flush=True)


if __name__ == '__main__':
    main()
