"""TRA/CV-only learning-rate search for the two fixed 508-parameter MLPs.

Run from the topological-invariants repository with uv, or pass --repo.
Every run is separate from the archived experiment. No test split is parsed.
"""
from __future__ import annotations

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/topological-invariants-matplotlib')

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
import sys
import time

import numpy as np
import torch

RATES = [.0001, .0002, .0005, .001, .002, .003, .005, .007,
         .01, .015, .02, .03, .05, .07, .1]
BRANCHES = ('geometry', 'topology')
A = None
Tiny = None


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def source_modules(repo):
    sys.path.insert(0, str(Path(repo) / 'outputs/real_topology'))
    sys.path.insert(0, str(Path(repo) / 'outputs/tiny_topology'))
    return (importlib.import_module('optdigits_pilot'),
            importlib.import_module('optdigits_ablation'),
            importlib.import_module('tiny_mlp_pilot'))


def prepare(repo, run):
    loader, features, _ = source_modules(repo)
    raw, labels = {}, {}
    with np.load(repo / 'work/real_data/optdigits_complement_trainval_v1.npz') as old:
        for split, part, prefix in [('train', 'tra', 'tr'), ('validation', 'cv', 'va')]:
            images, labels[split] = loader.read_original(part)
            topology, geometry = features.scale_features(images)
            raw[split] = dict(pixels=images.reshape(-1, 8, 4, 8, 4).mean((2, 4)).reshape(-1, 64),
                              topology=topology, geometry=geometry)
            for key, values in raw[split].items():
                old_key = 'pixels' if key == 'pixels' else 'global_' + key
                np.testing.assert_array_equal(values, old[prefix + '_' + old_key])
    arrays = {}
    for key in ['pixels', 'geometry', 'topology']:
        values = raw['train'][key]
        mean, std = values.mean(0), values.std(0)
        keep = std >= 1e-10
        std = np.where(keep, std, 1)
        arrays['mean_' + key], arrays['std_' + key], arrays['keep_' + key] = mean, std, keep
        for split in raw:
            arrays[split + '_' + key] = np.where(keep, (raw[split][key] - mean) / std, 0).astype(np.float32)
    arrays.update(train_y=labels['train'], validation_y=labels['validation'])
    assert len(arrays['train_y']) == 1934 and len(arrays['validation_y']) == 946
    np.savez_compressed(run / 'arrays.npz', **arrays)


def initialize(repo, run):
    global A, Tiny
    torch.set_num_threads(1)
    _, _, model = source_modules(repo)
    Tiny = model.Tiny
    with np.load(Path(run) / 'arrays.npz') as z:
        A = {key: torch.from_numpy(z[key].copy()) for key in z.files
             if key.startswith(('train_', 'validation_'))}


def fit(config, run_dir, digest, max_epochs, patience):
    run = Path(run_dir)
    name = config['id']
    done, weight, predfile = [run / 'jobs' / (name + ext) for ext in ['.json', '.pt', '.npz']]
    if done.exists():
        result = json.loads(done.read_text())
        assert result['protocol_sha256'] == digest
        assert result['weight_sha256'] == sha(weight) and result['predictions_sha256'] == sha(predfile)
        return result
    branch, seed, lr = config['branch'], config['seed'], config['lr']
    x, v, y, vy = [A[k] for k in ['train_pixels', 'validation_pixels', 'train_y', 'validation_y']]
    f, vf = .3 * A['train_' + branch], .3 * A['validation_' + branch]
    torch.manual_seed(seed)
    net = Tiny(64, 8, 6, 1, 'early')
    assert sum(p.numel() for p in net.parameters()) == 508
    initial_sha = hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in net.parameters())).hexdigest()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=.01)
    generator = torch.Generator().manual_seed(2026092800 + seed)
    best, stale, curve = float('inf'), 0, []
    legacy_best, legacy_stale, legacy_closed = float('inf'), 0, False
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        net.train()
        total_loss = 0.
        for indices in torch.randperm(len(y), generator=generator).split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(net(x[indices], f[indices]), y[indices])
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite training loss: {name}, epoch {epoch}')
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        net.eval()
        with torch.no_grad():
            logits = net(v, vf)
            ce = float(torch.nn.functional.cross_entropy(logits, vy))
            accuracy = float((logits.argmax(1) == vy).double().mean())
        if not np.isfinite(ce):
            raise RuntimeError(f'Non-finite validation loss: {name}, epoch {epoch}')
        curve.append([total_loss / len(y), ce, accuracy])
        if ce < best - 1e-8:
            best, best_epoch, stale = ce, epoch, 0
            state = {k: p.detach().clone() for k, p in net.state_dict().items()}
            probabilities = logits.softmax(1).numpy()
        else:
            stale += 1
        if not legacy_closed:
            if ce < legacy_best - 1e-8:
                legacy_best, legacy_epoch, legacy_stale = ce, epoch, 0
                legacy_probabilities = logits.softmax(1).numpy()
            else:
                legacy_stale += 1
            if legacy_stale >= 30 or epoch >= 250:
                legacy_closed = True
                legacy_stop = epoch
        if stale >= patience:
            break
    reason = 'patience' if stale >= patience else 'max_epochs'
    record = dict(**config, parameters=508, accuracy=float((probabilities.argmax(1) == vy.numpy()).mean()),
                  ce=best, selected_epoch=best_epoch, epochs_run=epoch, stop_reason=reason,
                  convergence_censored=reason == 'max_epochs', seconds=time.perf_counter() - started,
                  initial_parameters_sha256=initial_sha, protocol_sha256=digest,
                  legacy=dict(accuracy=float((legacy_probabilities.argmax(1) == vy.numpy()).mean()),
                              ce=legacy_best, selected_epoch=legacy_epoch,
                              epochs_run=legacy_stop if legacy_closed else epoch,
                              schedule_complete=legacy_closed))
    torch.save(dict(state_dict=state, config=config, protocol_sha256=digest, selected_epoch=best_epoch), weight)
    np.savez_compressed(predfile, probabilities=probabilities, legacy_probabilities=legacy_probabilities,
                        curve=np.asarray(curve, dtype=np.float64))
    record.update(weight_sha256=sha(weight), predictions_sha256=sha(predfile))
    write(done, record)
    return record


def group(rows, seeds, legacy=False):
    grouped = {}
    for row in rows:
        grouped.setdefault((row['branch'], row['lr']), []).append(row)
    answer = []
    for (branch, lr), items in sorted(grouped.items()):
        items.sort(key=lambda r: r['seed'])
        assert [r['seed'] for r in items] == seeds
        measurements = [r['legacy'] if legacy else r for r in items]
        acc, ce = [r['accuracy'] for r in measurements], [r['ce'] for r in measurements]
        answer.append(dict(branch=branch, lr=lr, seed_count=len(seeds), mean_accuracy=float(np.mean(acc)),
                           sd_accuracy=float(np.std(acc, ddof=1)) if len(seeds) > 1 else 0.,
                           mean_ce=float(np.mean(ce)), seed_accuracy=acc, seed_ce=ce,
                           selected_epochs=[r['selected_epoch'] for r in measurements],
                           median_selected_epoch=float(np.median([r['selected_epoch'] for r in measurements])),
                           mean_epochs_run=float(np.mean([r['epochs_run'] for r in measurements])),
                           censored_count=sum(r['convergence_censored'] for r in items) if not legacy else 0))
    return answer


def select(groups, allowed=None):
    return {branch: min([r for r in groups if r['branch'] == branch and (allowed is None or r['lr'] in allowed)],
                        key=lambda r: (-r['mean_accuracy'], r['mean_ce'], r['lr'])) for branch in BRANCHES}


def audit(repo, run, rows, digest):
    initialize(repo, run)
    proof = []
    initial = {}
    for row in sorted(rows, key=lambda r: r['id']):
        checkpoint = torch.load(run / 'jobs' / (row['id'] + '.pt'), weights_only=False, map_location='cpu')
        assert checkpoint['protocol_sha256'] == digest
        net = Tiny(64, 8, 6, 1, 'early')
        net.load_state_dict(checkpoint['state_dict'])
        net.eval()
        with torch.no_grad():
            logits = net(A['validation_pixels'], .3 * A['validation_' + row['branch']])
            ce = float(torch.nn.functional.cross_entropy(logits, A['validation_y']))
            prob = logits.softmax(1).numpy()
        with np.load(run / 'jobs' / (row['id'] + '.npz')) as saved:
            np.testing.assert_array_equal(prob, saved['probabilities'])
            assert float((prob.argmax(1) == A['validation_y'].numpy()).mean()) == row['accuracy']
            assert ce == row['ce']
            if row['seed'] in (0, 1, 2) and row['lr'] in (.003, .01) and row['legacy']['schedule_complete']:
                oi = 0 if row['lr'] == .003 else 1
                name = f"b512_pixels_{row['branch']}_early_d1_o{oi}_s{row['seed']}"
                with np.load(repo / 'work/tiny_topology' / (name + '.npz')) as historical:
                    np.testing.assert_array_equal(saved['legacy_probabilities'], historical['probabilities'])
                proof.append(name)
        initial.setdefault(row['seed'], set()).add(row['initial_parameters_sha256'])
    assert all(len(hashes) == 1 for hashes in initial.values())
    result = dict(passed=True, checkpoint_predictions_verified=len(rows),
                  identical_initial_weights_across_branches_and_rates=True,
                  historical_schedule_exact_probability_matches=proof,
                  test_partitions_parsed=[], protocol_sha256=digest)
    write(run / 'audit.json', result)
    return result


def report(run, protocol, rows, proof):
    groups = group(rows, protocol['seeds'])
    legacy = group(rows, protocol['seeds'], legacy=True)
    chosen, legacy_chosen = select(groups), select(legacy)
    old = select(groups, allowed=[r for r in [.003, .01] if r in protocol['learning_rates']]) if any(r in protocol['learning_rates'] for r in [.003, .01]) else None
    bounds = (min(protocol['learning_rates']), max(protocol['learning_rates']))
    matched = []
    for lr in protocol['learning_rates']:
        geo = next(r for r in groups if r['branch'] == 'geometry' and r['lr'] == lr)
        topo = next(r for r in groups if r['branch'] == 'topology' and r['lr'] == lr)
        differences = np.asarray(topo['seed_accuracy']) - np.asarray(geo['seed_accuracy'])
        matched.append(dict(lr=lr, mean_difference_pp=float(differences.mean() * 100),
                            positive_seed_pairs=int((differences > 0).sum())))
    result = dict(completed_utc=now(), scope=protocol['scope'], fits=len(rows), groups=groups,
                  selected=chosen, legacy_schedule_groups=legacy, legacy_schedule_selected=legacy_chosen,
                  original_two_rates_with_longer_training=old, matched_rate_comparisons=matched,
                  selected_difference_pp=100 * (chosen['topology']['mean_accuracy'] - chosen['geometry']['mean_accuracy']),
                  selected_rate_at_search_boundary={b: chosen[b]['lr'] in bounds for b in BRANCHES},
                  censored_fits=sum(r['convergence_censored'] for r in rows), audit_passed=proof['passed'],
                  limitation='Best observed on reused CV among tested rates, not a global optimum or a new test-set result.')
    write(run / 'result.json', result)
    lines = ['# Подбор learning rate для двух MLP на 508 параметров', '',
             'Только TRA (1934) и CV (946); WDEP и WINDEP не распаковывались. '
             'Это дополнительный подбор на уже использованной валидации, а не независимая оценка качества.', '',
             f"Проверено {len(protocol['learning_rates'])} значений learning rate × 2 модели × {len(protocol['seeds'])} инициализаций = {len(rows)} обучений.",
             f"Архитектура 72→6→10, AdamW, weight decay 0.01, batch size 256; максимум {protocol['max_epochs']} эпох, patience {protocol['patience']}.",
             'Чекпойнт выбирается по минимуму CV cross-entropy; learning rate — по средней CV accuracy, затем средней cross-entropy. '
             'Стандартизация обучается только на TRA. Наборы значений, инициализации, порядок батчей и правила остановки одинаковы для двух представлений.', '',
             '| Модель | Лучший проверенный LR | CV accuracy, % | SD по инициализациям, п.п. | Медиана выбранной эпохи |',
             '|---|---:|---:|---:|---:|']
    for b, title in [('geometry', 'Геометрия'), ('topology', 'Топология')]:
        v = chosen[b]
        lines.append(f"| {title} | {v['lr']:.6g} | {100*v['mean_accuracy']:.3f} | {100*v['sd_accuracy']:.3f} | {v['median_selected_epoch']:g} |")
    lines += ['', f"Разность выбранных конфигураций: {result['selected_difference_pp']:+.3f} п.п. на CV. Это оценка после выбора по той же CV, без утверждения о значимости или новом тестовом результате.", '',
              '## Вся сетка', '', '| LR | Геометрия: CV accuracy, % | Топология: CV accuracy, % | Разность, п.п. |', '|---:|---:|---:|---:|']
    for comparison in matched:
        lr = comparison['lr']
        geo = next(r for r in groups if r['branch'] == 'geometry' and r['lr'] == lr)
        topo = next(r for r in groups if r['branch'] == 'topology' and r['lr'] == lr)
        lines.append(f"| {lr:.6g} | {100*geo['mean_accuracy']:.3f} | {100*topo['mean_accuracy']:.3f} | {comparison['mean_difference_pp']:+.3f} |")
    lines += ['', '## Проверка влияния продолжительности обучения', '',
              'Для каждого свежего обучения дополнительно сохранён результат по старому правилу: максимум 250 эпох и patience 30. '
              'Это тот же начальный участок обучения с теми же весами и батчами, без отдельного повторного обучения.', '',
              '| Режим | LR геометрии | Accuracy геометрии, % | LR топологии | Accuracy топологии, % |', '|---|---:|---:|---:|---:|']
    schedules = [('Вся сетка, старое правило остановки', legacy_chosen), ('Вся сетка, длинное обучение', chosen)]
    if old is not None:
        schedules.append(('Только 0.003/0.01, длинное обучение', old))
    for title, selected in schedules:
        g, t = selected['geometry'], selected['topology']
        lines.append(f"| {title} | {g['lr']:.6g} | {100*g['mean_accuracy']:.3f} | {t['lr']:.6g} | {100*t['mean_accuracy']:.3f} |")
    lines += ['', '## Проверки и ограничения', '',
              f"- Все {proof['checkpoint_predictions_verified']} сохранённых моделей повторно загружены; вероятности и метрики пересчитаны точно.",
              f"- {len(proof['historical_schedule_exact_probability_matches'])} результатов по старому правилу побитово совпали с историческими вероятностями на CV.",
              '- Начальные веса совпадают между обеими моделями и всеми learning rate для каждого seed.',
              f"- Обучений, дошедших до ограничения числа эпох: {result['censored_fits']}.",
              f"- Выбранный LR на границе диапазона: {result['selected_rate_at_search_boundary']}.",
              '- SD характеризует разброс по инициализациям; это не доверительный интервал качества на новых данных.',
              '- WDEP и WINDEP ранее использовались в исследовании; этот запуск не делает их снова независимыми тестами.',
              '- Оптимальность установлена только среди проверенных значений на данной CV при фиксированных остальных настройках.', '',
              '![Подбор learning rate](learning_rate_search.png)', '',
              '## Воспроизведение', '', 'Из корня репозитория:', '', '```bash',
              'uv run --locked python tune_learning_rate.py --workers 4', '```', '',
              'Каждый запуск создаёт новую папку runs/lr-search-ДАТА. Для возобновления укажите тот же --output и --resume. '
              'Протокол, его исходный код, входные массивы, все чекпойнты, кривые, вероятности и аудит сохранены в папке запуска.']
    (run / 'report_ru.md').write_text('\n'.join(lines) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), layout='constrained')
    for b, label, color in [('geometry', 'Geometry', '#bb6329'), ('topology', 'Topology', '#246b91')]:
        data = sorted([r for r in groups if r['branch'] == b], key=lambda r: r['lr'])
        rates = [r['lr'] for r in data]
        axes[0].errorbar(rates, [100*r['mean_accuracy'] for r in data],
                         yerr=[100*r['sd_accuracy'] for r in data], marker='o', capsize=2,
                         linewidth=1.5, color=color, label=label)
        axes[1].plot(rates, [r['median_selected_epoch'] for r in data], 'o-', color=color, label=label)
    for ax in axes:
        ax.set_xscale('log'); ax.set_xlabel('Learning rate'); ax.grid(alpha=.2); ax.legend()
    axes[0].set_ylabel('CV accuracy, %'); axes[0].set_title(f"Mean ± SD over {len(protocol['seeds'])} seeds")
    axes[1].set_ylabel('Median selected epoch'); axes[1].set_yscale('log'); axes[1].set_title('Checkpoint selected by CV loss')
    fig.suptitle('Fixed 508-parameter MLPs · TRA/CV only · validation reused for selection')
    fig.savefig(run / 'learning_rate_search.png', dpi=180)
    fig.savefig(run / 'learning_rate_search.svg')
    plt.close(fig)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--learning-rates', type=float, nargs='+', default=RATES)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    parser.add_argument('--max-epochs', type=int, default=6000)
    parser.add_argument('--patience', type=int, default=150)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or args.max_epochs < 1 or args.patience < 1:
        parser.error('workers, max-epochs and patience must be positive')
    if any(not np.isfinite(lr) or lr <= 0 for lr in args.learning_rates):
        parser.error('learning rates must be positive and finite')
    if len(set(args.learning_rates)) != len(args.learning_rates) or len(set(args.seeds)) != len(args.seeds):
        parser.error('rates and seeds must be distinct')
    repo = args.repo.resolve()
    run = (args.output or repo / 'runs' / ('lr-search-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))).resolve()
    sources = ['outputs/real_topology/optdigits_pilot.py', 'outputs/real_topology/optdigits_ablation.py',
               'outputs/tiny_topology/tiny_mlp_pilot.py', 'work/real_data/optdigits.zip',
               'work/real_data/optdigits_complement_trainval_v1.npz']
    input_hashes = {name: sha(repo / name) for name in sources}
    code_hash = sha(__file__)
    design = dict(scope='Exploratory LR selection on reused TRA/CV; no test-set evaluation and no global-optimum claim.',
                  architecture='MLP 72→6→10 ReLU, 508 parameters', branches=list(BRANCHES),
                  learning_rates=sorted(args.learning_rates), seeds=sorted(args.seeds),
                  optimizer='AdamW', weight_decay=.01, batch_size=256, side_scale=.3,
                  max_epochs=args.max_epochs, patience=args.patience, improvement_tolerance=1e-8,
                  checkpoint_selection='Minimum validation cross-entropy; earliest tie',
                  rate_selection='Maximum mean validation accuracy over seeds, then minimum mean CE, then smaller LR',
                  legacy_schedule=dict(max_epochs=250, patience=30),
                  normalization='Per-feature mean/std ddof=0 fit only to TRA; small-variance columns zeroed',
                  rng='torch.manual_seed(seed); independent permutation generator 2026092800+seed',
                  train_n=1934, validation_n=946, parsed_archive_members=['optdigits-orig.tra.Z', 'optdigits-orig.cv.Z'],
                  source_sha256=input_hashes, runner_sha256=code_hash)
    if run.exists():
        if not args.resume:
            parser.error(f'Output already exists; use a new directory or --resume: {run}')
        protocol = json.loads((run / 'protocol.json').read_text())
        for key, value in design.items():
            assert protocol[key] == value, f'Protocol changed: {key}'
        assert protocol['arrays_sha256'] == sha(run / 'arrays.npz')
    else:
        if args.resume:
            parser.error('--resume requires an existing output directory')
        run.mkdir(parents=True)
        (run / 'jobs').mkdir()
        prepare(repo, run)
        shutil.copy2(__file__, run / 'tune_learning_rate.py')
        protocol = dict(**design, registered_utc=now(), arrays_sha256=sha(run / 'arrays.npz'),
                        environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, platform=platform.platform()))
        write(run / 'protocol.json', protocol)
    digest = sha(run / 'protocol.json')
    configs = [dict(id=f'{b}_lr{index:02d}_s{seed}', branch=b, lr=lr, seed=seed)
               for index, lr in enumerate(protocol['learning_rates']) for seed in protocol['seeds'] for b in BRANCHES]
    started = time.perf_counter()
    rows = []
    print(json.dumps(dict(output=str(run), fits=len(configs), protocol_sha256=digest)), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                             initializer=initialize, initargs=(str(repo), str(run))) as pool:
        futures = [pool.submit(fit, c, str(run), digest, args.max_epochs, args.patience) for c in configs]
        for future in as_completed(futures):
            rows.append(future.result())
            write(run / 'status.json', dict(status='running', updated_utc=now(), completed=len(rows), total=len(configs),
                                           elapsed_seconds=time.perf_counter() - started, latest=rows[-1]['id']))
            if len(rows) % 10 == 0 or len(rows) == len(configs):
                print(json.dumps(dict(completed=len(rows), total=len(configs), seconds=round(time.perf_counter() - started, 1))), flush=True)
    proof = audit(repo, run, rows, digest)
    result = report(run, protocol, rows, proof)
    write(run / 'status.json', dict(status='completed', updated_utc=now(), completed=len(rows), total=len(configs),
                                   elapsed_seconds=time.perf_counter() - started, audit_passed=True))
    print(json.dumps(dict(selected=result['selected'], censored_fits=result['censored_fits'],
                          selected_difference_pp=result['selected_difference_pp']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
