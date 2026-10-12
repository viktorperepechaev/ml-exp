"""Refit CV-selected alpha=1 OptDigits MLPs on TRA+CV and evaluate WDEP.

LR and fixed training epochs are derived only from the archived CV search.
All 20 weights are frozen before reading WDEP. WDEP was used previously;
this is a descriptive evaluation of the updated configurations, not a new
independent confirmation. WINDEP is never parsed.
"""
from __future__ import annotations

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib
import json
import multiprocessing as mp
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch

BASE = None
ARRAYS = None
BRANCHES = ('geometry', 'topology')
REFERENCE = 'runs/feature-scale-20261011'


def backend(repo):
    global BASE
    if BASE is None:
        sys.path.insert(0, str(Path(repo).resolve()))
        BASE = importlib.import_module('tune_learning_rate')
    return BASE


def cv_selection(repo):
    base = backend(repo)
    directory = repo / REFERENCE
    protocol = json.loads((directory / 'protocol.json').read_text())
    result = json.loads((directory / 'result.json').read_text())
    proof = json.loads((directory / 'audit.json').read_text())
    assert proof['passed'] and proof['checkpoint_predictions_verified'] == 600
    assert proof['protocol_sha256'] == base.sha(directory / 'protocol.json')
    assert protocol['seeds'] == list(range(10))
    assert protocol['architecture'] == 'MLP 72→6→10 ReLU, 508 parameters'
    for path, digest in protocol['source_sha256'].items():
        assert base.sha(repo / path) == digest, f'CV source changed: {path}'
    rows = [json.loads(path.read_text()) for path in (directory / 'jobs').glob('a1_*.json')]
    assert len(rows) == 300 and all(row['alpha'] == 1. for row in rows)
    groups = base.group(rows, protocol['seeds'])
    selected = base.select(groups)
    configs = {}
    for branch in BRANCHES:
        chosen = selected[branch]
        for field, value in chosen.items():
            assert result['selected']['1.0'][branch][field] == value
        median = float(np.median(chosen['selected_epochs']))
        configs[branch] = dict(branch=branch, alpha=1., parameters=508,
                               lr=chosen['lr'], epochs=int(median),
                               cv_selected_epochs=chosen['selected_epochs'],
                               cv_median_selected_epoch=median,
                               cv_mean_accuracy=chosen['mean_accuracy'])
    return configs, protocol


def raw_features(repo, images):
    _, extractor, _ = BASE.source_modules(repo)
    topology, geometry = extractor.scale_features(images)
    return dict(pixels=images.reshape(-1, 8, 4, 8, 4).mean((2, 4)).reshape(-1, 64),
                topology=topology, geometry=geometry)


def prepare_train(repo, run):
    loader, _, _ = BASE.source_modules(repo)
    ti, ty = loader.read_original('tra')
    vi, vy = loader.read_original('cv')
    images, labels = np.concatenate([ti, vi]), np.concatenate([ty, vy])
    raw = raw_features(repo, images)
    assert len(labels) == 2880
    with np.load(repo / 'work/real_data/optdigits_complement_trainval_v1.npz') as cached:
        for key, values in raw.items():
            old_key = 'pixels' if key == 'pixels' else 'global_' + key
            np.testing.assert_array_equal(values, np.concatenate([cached['tr_' + old_key], cached['va_' + old_key]]))
    arrays = dict(train_y=labels)
    for key, values in raw.items():
        assert values.dtype == np.float64
        mean, std = values.mean(0), values.std(0)
        keep = std >= 1e-10
        std = np.where(keep, std, 1)
        arrays.update({f'raw_{key}': values, f'mean_{key}': mean, f'std_{key}': std, f'keep_{key}': keep,
                       f'train_{key}': np.where(keep, (values - mean) / std, 0).astype(np.float32)})
    np.savez_compressed(run / 'train_features.npz', **arrays)


def initialize(repo, run):
    global ARRAYS
    backend(repo)
    torch.set_num_threads(1)
    _, _, model = BASE.source_modules(repo)
    BASE.Tiny = model.Tiny
    with np.load(Path(run) / 'train_features.npz') as arrays:
        ARRAYS = {key: torch.from_numpy(arrays[key].copy()) for key in arrays.files if key.startswith('train_')}


def fit(config, seed, run, digest):
    run = Path(run)
    name = f"{config['branch']}_s{seed}"
    torch.manual_seed(seed)
    model = BASE.Tiny(64, 8, 6, 1, 'early')
    assert sum(p.numel() for p in model.parameters()) == 508
    initial_sha = hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=.01)
    generator = torch.Generator().manual_seed(2026092800 + seed)
    x, f, y = ARRAYS['train_pixels'], config['alpha'] * ARRAYS['train_' + config['branch']], ARRAYS['train_y']
    curve, started = [], time.perf_counter()
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        total = 0.
        for indices in torch.randperm(len(y), generator=generator).split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x[indices], f[indices]), y[indices])
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss: {name}, epoch {epoch}')
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        curve.append(total / len(y))
    weight = run / 'models' / (name + '.pt')
    torch.save(dict(state_dict=model.state_dict(), config=config, seed=seed, protocol_sha256=digest), weight)
    record = dict(name=name, seed=seed, config=config, protocol_sha256=digest,
                  initial_parameters_sha256=initial_sha, weight_sha256=BASE.sha(weight),
                  training_loss=curve, seconds=time.perf_counter() - started, wdep_read=False)
    BASE.write(run / 'models' / (name + '.json'), record)
    return record


def prepare_wdep(repo, run, digest, configs, seeds):
    receipt = json.loads((run / 'weights_receipt.json').read_text())
    expected = {f'{branch}_s{seed}.pt' for branch in configs for seed in seeds}
    assert receipt['protocol_sha256'] == digest and set(receipt['weights']) == expected
    for name, value in receipt['weights'].items():
        assert BASE.sha(run / 'models' / name) == value
    with zipfile.ZipFile(repo / 'work/real_data/optdigits.zip') as archive:
        compressed = archive.read('optdigits-orig.wdep.Z')
    text = subprocess.run(['gzip', '-dc'], input=compressed, capture_output=True, check=True).stdout.decode('ascii')
    lines = [line.strip() for line in text.splitlines()]
    images, labels, i = [], [], 0
    while i < len(lines):
        if len(lines[i]) == 32 and set(lines[i]) <= {'0', '1'}:
            rows = lines[i:i + 32]
            assert len(rows) == 32 and all(len(row) == 32 and set(row) <= {'0', '1'} for row in rows)
            label = lines[i + 32]
            assert len(label) == 1 and label.isdigit()
            images.append(np.frombuffer(''.join(rows).encode(), np.uint8).reshape(32, 32) == ord('1'))
            labels.append(int(label))
            i += 33
        else:
            i += 1
    assert len(labels) == 943
    raw = raw_features(repo, np.stack(images))
    arrays = dict(test_y=np.asarray(labels, dtype=np.int64))
    with np.load(run / 'train_features.npz') as train:
        for key, values in raw.items():
            arrays['raw_' + key] = values
            arrays['test_' + key] = np.where(train['keep_' + key], (values - train['mean_' + key]) / train['std_' + key], 0).astype(np.float32)
    np.savez_compressed(run / 'wdep_features.npz', **arrays)
    BASE.write(run / 'wdep_read_receipt.json', dict(read_utc=BASE.now(), protocol_sha256=digest,
                weights_receipt_sha256=BASE.sha(run / 'weights_receipt.json'), member='optdigits-orig.wdep.Z',
                compressed_member_sha256=hashlib.sha256(compressed).hexdigest(), images=943,
                features_sha256=BASE.sha(run / 'wdep_features.npz'), metrics_computed=False))


def predict(run, configs, seeds, digest):
    arrays = {}
    with np.load(run / 'wdep_features.npz') as data:
        x = torch.from_numpy(data['test_pixels'].copy())
        for branch, config in configs.items():
            f = torch.from_numpy(config['alpha'] * data['test_' + branch].copy())
            logits, probabilities = [], []
            for seed in seeds:
                checkpoint = torch.load(run / 'models' / f'{branch}_s{seed}.pt', weights_only=False, map_location='cpu')
                assert checkpoint['protocol_sha256'] == digest and checkpoint['config'] == config and checkpoint['seed'] == seed
                model = BASE.Tiny(64, 8, 6, 1, 'early')
                model.load_state_dict(checkpoint['state_dict'])
                model.eval()
                with torch.no_grad():
                    values = torch.cat([model(x[i:i+256], f[i:i+256]) for i in range(0, len(x), 256)])
                    prob = values.softmax(1)
                logits.append(values.numpy()); probabilities.append(prob.numpy())
            arrays[branch + '_logits'] = np.stack(logits)
            arrays[branch + '_probabilities'] = np.stack(probabilities)
            arrays[branch + '_predictions'] = arrays[branch + '_probabilities'].argmax(2)
    np.savez_compressed(run / 'predictions.npz', **arrays)
    BASE.write(run / 'prediction_receipt.json', dict(predicted_utc=BASE.now(), protocol_sha256=digest,
                predictions_sha256=BASE.sha(run / 'predictions.npz'), metrics_computed=False,
                note='Labels are parsed beside images but are not used to compute predictions.'))


def analyze(run, configs, seeds, digest, started):
    receipt = json.loads((run / 'prediction_receipt.json').read_text())
    assert receipt['protocol_sha256'] == digest and receipt['predictions_sha256'] == BASE.sha(run / 'predictions.npz')
    with np.load(run / 'wdep_features.npz') as data:
        labels = data['test_y'].copy()
    summaries, correct = {}, {}
    with np.load(run / 'predictions.npz') as predictions:
        for branch in BRANCHES:
            correct[branch] = predictions[branch + '_predictions'] == labels
            accuracies = correct[branch].mean(1)
            ce = [float(torch.nn.functional.cross_entropy(torch.from_numpy(logits.copy()), torch.from_numpy(labels)))
                  for logits in predictions[branch + '_logits']]
            summaries[branch] = dict(config=configs[branch], mean_accuracy=float(accuracies.mean()),
                                    sd_accuracy=float(accuracies.std(ddof=1)), seed_accuracy=accuracies.tolist(),
                                    correct_counts=correct[branch].sum(1).tolist(), mean_ce=float(np.mean(ce)), seed_ce=ce)
    difference = correct['topology'].astype(int) - correct['geometry'].astype(int)
    delta = difference.mean(1) * 100
    result = dict(completed_utc=BASE.now(), protocol_sha256=digest, train_n=2880, test_n=943,
                  models=20, seeds=seeds, parameters_per_model=508, side_scale=1., summaries=summaries,
                  difference_pp=float(delta.mean()), sd_seed_difference_pp=float(delta.std(ddof=1)),
                  seed_differences_pp=delta.tolist(), positive_seed_pairs=int((delta > 0).sum()),
                  negative_seed_pairs=int((delta < 0).sum()), tied_seed_pairs=int((delta == 0).sum()),
                  relative_error_reduction=float(1 - (1 - summaries['topology']['mean_accuracy']) / (1 - summaries['geometry']['mean_accuracy'])),
                  elapsed_seconds=time.perf_counter() - started,
                  scope='Descriptive WDEP evaluation of CV-selected alpha=1 configurations; WDEP was previously seen, writers overlap TRA/CV, no ensemble or new independent confirmation.')
    BASE.write(run / 'result.json', result)
    lines = ['# OptDigits WDEP: коэффициент дополнительных признаков 1', '',
             'Заново обучены две MLP 72→6→10, по 508 параметров и по 10 инициализаций. '
             'Обучение и стандартизация — TRA+CV, 2880 изображений; оценка — WDEP, 943 изображения. '
             'Исходные признаки те же; коэффициент дополнительных признаков равен 1.', '',
             'LR выбран по средней accuracy на CV в предыдущем сравнении коэффициентов. '
             'Число эпох фиксировано как целая часть медианы выбранных по минимуму CV CE эпох десяти запусков '
             'при выбранном LR. Здесь обе медианы целые. Все 20 моделей обучены и зафиксированы до чтения WDEP; '
             'после оценки настройки не менялись.', '',
             'AdamW, weight decay 0.01, batch size 256. Инициализация seed 0–9, '
             'генератор перестановок 2026092800+seed. Усредняются accuracy отдельных моделей; ансамбля нет.', '',
             '| Модель | LR | Эпох | Средняя CV accuracy, % | Средняя WDEP accuracy, % | SD WDEP, п.п. |',
             '|---|---:|---:|---:|---:|---:|']
    names = dict(geometry='Геометрическая', topology='Топологическая')
    for branch in BRANCHES:
        value = summaries[branch]; config = configs[branch]
        lines.append(f"| {names[branch]} | {config['lr']:g} | {config['epochs']} | {100*config['cv_mean_accuracy']:.3f} | {100*value['mean_accuracy']:.3f} | {100*value['sd_accuracy']:.3f} |")
    lines += ['', f"Разность топология − геометрия: **{result['difference_pp']:+.3f} п.п.**; "
              f"положительных пар по seed {result['positive_seed_pairs']}/10, "
              f"отрицательных {result['negative_seed_pairs']}/10, равных {result['tied_seed_pairs']}/10.", '',
              'SD отражает разброс по инициализациям. WDEP уже использовалась в исследовании, '
              'и её авторы почерка встречаются в обучении. Это повторная оценка новой конфигурации; '
              'статистическая значимость и обобщение на новых авторов этим запуском не устанавливаются. '
              'WINDEP не читалась.', '',
              '## Воспроизведение', '', 'Из корня репозитория:', '', '```bash',
              'uv run --locked python evaluate_wdep_alpha1.py --workers 4', '```', '',
              'Скрипт создаёт новый каталог runs/wdep-alpha1-ДАТА и отказывается заменять существующий. '
              'Требуется сохранённый подбор runs/feature-scale-20261011. '
              'Протокол и его источники, обучающие и тестовые признаки, веса, кривые обучения, '
              'предсказания и проверка сохраняются вместе с отчётом.']
    (run / 'report_ru.md').write_text('\n'.join(lines) + '\n')
    return result


def audit(repo, run, configs, seeds, digest):
    initialize(repo, run)
    weight_receipt = json.loads((run / 'weights_receipt.json').read_text())
    initials = {}
    with np.load(run / 'train_features.npz') as train, np.load(run / 'wdep_features.npz') as test:
        x = torch.from_numpy(test['test_pixels'].copy())
        for key in ('pixels', 'geometry', 'topology'):
            raw = train['raw_' + key]
            np.testing.assert_array_equal(train['mean_' + key], raw.mean(0))
            np.testing.assert_array_equal(train['keep_' + key], raw.std(0) >= 1e-10)
            np.testing.assert_array_equal(train['std_' + key], np.where(train['keep_' + key], raw.std(0), 1))
            np.testing.assert_array_equal(test['test_' + key], np.where(train['keep_' + key],
                    (test['raw_' + key] - train['mean_' + key]) / train['std_' + key], 0).astype(np.float32))
        with np.load(run / 'predictions.npz') as saved:
            for branch, config in configs.items():
                f = torch.from_numpy(config['alpha'] * test['test_' + branch].copy())
                for index, seed in enumerate(seeds):
                    name = f'{branch}_s{seed}'
                    record = json.loads((run / 'models' / (name + '.json')).read_text())
                    assert BASE.sha(run / 'models' / (name + '.pt')) == record['weight_sha256'] == weight_receipt['weights'][name + '.pt']
                    assert record['wdep_read'] is False and len(record['training_loss']) == config['epochs']
                    checkpoint = torch.load(run / 'models' / (name + '.pt'), weights_only=False, map_location='cpu')
                    assert checkpoint['config'] == config and checkpoint['seed'] == seed and checkpoint['protocol_sha256'] == digest
                    model = BASE.Tiny(64, 8, 6, 1, 'early')
                    model.load_state_dict(checkpoint['state_dict']); model.eval()
                    with torch.no_grad():
                        logits = torch.cat([model(x[i:i+256], f[i:i+256]) for i in range(0, len(x), 256)])
                    np.testing.assert_array_equal(saved[branch + '_logits'][index], logits.numpy())
                    np.testing.assert_array_equal(saved[branch + '_probabilities'][index], logits.softmax(1).numpy())
                    np.testing.assert_array_equal(saved[branch + '_predictions'][index], logits.argmax(1).numpy())
                    initials.setdefault(seed, set()).add(record['initial_parameters_sha256'])
    assert all(len(values) == 1 for values in initials.values())
    proof = dict(passed=True, protocol_sha256=digest, verified_models=20,
                 reloaded_checkpoint_logits_probabilities_predictions_exact=True,
                 normalization_fit_to_tra_cv_only=True, identical_initial_weights_between_branches=True,
                 fixed_weights_receipt_before_wdep_read=True, windep_parsed=False)
    BASE.write(run / 'audit.json', proof)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    repo = args.repo.resolve()
    base = backend(repo)
    torch.set_num_threads(1)
    configs, cv_protocol = cv_selection(repo)
    run = (args.output or repo / 'runs' / ('wdep-alpha1-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))).resolve()
    if run.exists():
        parser.error(f'Output already exists; choose a new directory: {run}')
    run.mkdir(parents=True)
    (run / 'models').mkdir()
    protocol = dict(registered_utc=base.now(), purpose=__doc__, configs=configs, seeds=cv_protocol['seeds'],
                    architecture='MLP 72→6→10 ReLU, 508 parameters', side_scale=1.,
                    epoch_rule='int(median(CV-selected minimum-CE epochs at selected LR across all ten seeds)); no WDEP stopping',
                    optimizer='AdamW', weight_decay=.01, batch_size=256,
                    normalization='Per-feature mean/std ddof=0 refitted on TRA+CV only; std<1e-10 columns zeroed',
                    rng='torch.manual_seed(seed); independent permutation generator 2026092800+seed',
                    train_n=2880, test_n=943, previous_wdep_exposure=True,
                    parsed_archive_members=['optdigits-orig.tra.Z', 'optdigits-orig.cv.Z', 'optdigits-orig.wdep.Z'],
                    cv_reference=REFERENCE,
                    cv_reference_sha256={name: base.sha(repo / REFERENCE / name) for name in ['protocol.json', 'result.json', 'audit.json']},
                    source_sha256=cv_protocol['source_sha256'], runner_sha256=base.sha(__file__),
                    environment=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__, platform=platform.platform()))
    base.write(run / 'protocol.json', protocol)
    digest = base.sha(run / 'protocol.json')
    shutil.copy2(__file__, run / 'evaluate_wdep_alpha1.py')
    shutil.copy2(repo / 'tune_learning_rate.py', run / 'tune_learning_rate.py')
    started = time.perf_counter()
    print(json.dumps(dict(run=str(run), configs=configs, protocol_sha256=digest)), flush=True)
    prepare_train(repo, run)
    base.write(run / 'train_features_receipt.json', dict(protocol_sha256=digest, features_sha256=base.sha(run / 'train_features.npz'), wdep_read=False))
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                             initializer=initialize, initargs=(str(repo), str(run))) as pool:
        futures = [pool.submit(fit, config, seed, str(run), digest) for config in configs.values() for seed in protocol['seeds']]
        for count, future in enumerate(as_completed(futures), 1):
            result = future.result()
            base.write(run / 'status.json', dict(status='training', completed=count, total=20, wdep_read=False, updated_utc=base.now()))
            print(f"{count}/20 trained: {result['name']}", flush=True)
    weights = {path.name: base.sha(path) for path in (run / 'models').glob('*.pt')}
    assert len(weights) == 20
    base.write(run / 'weights_receipt.json', dict(frozen_utc=base.now(), protocol_sha256=digest, weights=weights, wdep_read=False))
    print('All 20 weights frozen; reading and predicting WDEP.', flush=True)
    initialize(repo, run)
    prepare_wdep(repo, run, digest, configs, protocol['seeds'])
    predict(run, configs, protocol['seeds'], digest)
    proof = audit(repo, run, configs, protocol['seeds'], digest)
    result = analyze(run, configs, protocol['seeds'], digest, started)
    base.write(run / 'status.json', dict(status='completed', completed=20, total=20, wdep_read=True,
                                        audit_passed=proof['passed'], updated_utc=base.now()))
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
