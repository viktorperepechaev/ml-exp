"""Compare side-feature scales 0.3 and 1 for the fixed 508-parameter MLPs.

TRA/CV only. Both scales receive the same full LR grid and paired seeds.
Uses the unchanged preparation/model utilities from tune_learning_rate.py.
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

BASE = None
SCALES = (.3, 1.)
BRANCHES = ('geometry', 'topology')
PREVIOUS_RATES = dict(geometry=.005, topology=.002)


def backend(repo):
    global BASE
    if BASE is None:
        sys.path.insert(0, str(Path(repo).resolve()))
        BASE = importlib.import_module('tune_learning_rate')
    return BASE


def initialize(repo, run):
    backend(repo).initialize(repo, run)


def fit(config, run_dir, digest, max_epochs, patience):
    run = Path(run_dir)
    name = config['id']
    done, weight, predfile = [run / 'jobs' / (name + ext) for ext in ['.json', '.pt', '.npz']]
    if done.exists():
        result = json.loads(done.read_text())
        assert result['protocol_sha256'] == digest
        assert result['weight_sha256'] == BASE.sha(weight) and result['predictions_sha256'] == BASE.sha(predfile)
        return result
    branch, seed, lr, alpha = [config[k] for k in ['branch', 'seed', 'lr', 'alpha']]
    x, v, y, vy = [BASE.A[k] for k in ['train_pixels', 'validation_pixels', 'train_y', 'validation_y']]
    f, vf = alpha * BASE.A['train_' + branch], alpha * BASE.A['validation_' + branch]
    torch.manual_seed(seed)
    net = BASE.Tiny(64, 8, 6, 1, 'early')
    assert sum(p.numel() for p in net.parameters()) == 508
    initial_sha = hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in net.parameters())).hexdigest()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=.01)
    generator = torch.Generator().manual_seed(2026092800 + seed)
    best, stale, curve = float('inf'), 0, []
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
        if stale >= patience:
            break
    reason = 'patience' if stale >= patience else 'max_epochs'
    record = dict(**config, parameters=508,
                  accuracy=float((probabilities.argmax(1) == vy.numpy()).mean()),
                  ce=best, selected_epoch=best_epoch, epochs_run=epoch,
                  stop_reason=reason, convergence_censored=reason == 'max_epochs',
                  seconds=time.perf_counter() - started,
                  initial_parameters_sha256=initial_sha, protocol_sha256=digest)
    torch.save(dict(state_dict=state, config=config, protocol_sha256=digest, selected_epoch=best_epoch), weight)
    np.savez_compressed(predfile, probabilities=probabilities, curve=np.asarray(curve, dtype=np.float64))
    record.update(weight_sha256=BASE.sha(weight), predictions_sha256=BASE.sha(predfile))
    BASE.write(done, record)
    return record


def audit(repo, run, rows, digest, protocol):
    initialize(repo, run)
    initial, previous_matches = {}, []
    reference = repo / protocol['reference']['path'] if protocol['reference'] else None
    previous_protocol = json.loads((reference / 'protocol.json').read_text()) if reference else None
    expected_matches = 0
    for row in sorted(rows, key=lambda r: r['id']):
        weight = run / 'jobs' / (row['id'] + '.pt')
        predfile = run / 'jobs' / (row['id'] + '.npz')
        assert BASE.sha(weight) == row['weight_sha256'] and BASE.sha(predfile) == row['predictions_sha256']
        checkpoint = torch.load(weight, weights_only=False, map_location='cpu')
        assert checkpoint['protocol_sha256'] == digest
        assert all(checkpoint['config'][k] == row[k] for k in ['alpha', 'lr', 'branch', 'seed'])
        net = BASE.Tiny(64, 8, 6, 1, 'early')
        net.load_state_dict(checkpoint['state_dict'])
        net.eval()
        with torch.no_grad():
            logits = net(BASE.A['validation_pixels'], row['alpha'] * BASE.A['validation_' + row['branch']])
            ce = float(torch.nn.functional.cross_entropy(logits, BASE.A['validation_y']))
            prob = logits.softmax(1).numpy()
        with np.load(predfile) as saved:
            np.testing.assert_array_equal(prob, saved['probabilities'])
            assert float((prob.argmax(1) == BASE.A['validation_y'].numpy()).mean()) == row['accuracy']
            assert ce == row['ce']
            if (reference and row['alpha'] == .3 and row['lr'] in previous_protocol['learning_rates']
                    and row['seed'] in previous_protocol['seeds']
                    and protocol['max_epochs'] == previous_protocol['max_epochs']
                    and protocol['patience'] == previous_protocol['patience']):
                expected_matches += 1
                index = previous_protocol['learning_rates'].index(row['lr'])
                name = f"{row['branch']}_lr{index:02d}_s{row['seed']}"
                previous_record = json.loads((reference / 'jobs' / (name + '.json')).read_text())
                assert BASE.sha(reference / 'jobs' / (name + '.npz')) == previous_record['predictions_sha256']
                with np.load(reference / 'jobs' / (name + '.npz')) as old:
                    np.testing.assert_array_equal(saved['probabilities'], old['probabilities'])
                    np.testing.assert_array_equal(saved['curve'], old['curve'])
                for field in ['accuracy', 'ce', 'selected_epoch', 'epochs_run', 'initial_parameters_sha256']:
                    assert row[field] == previous_record[field], (name, field)
                previous_matches.append(name)
        initial.setdefault(row['seed'], set()).add(row['initial_parameters_sha256'])
    assert len(previous_matches) == expected_matches
    assert all(len(hashes) == 1 for hashes in initial.values())
    result = dict(passed=True, checkpoint_predictions_verified=len(rows),
                  identical_initial_weights_across_scales_branches_rates=True,
                  alpha03_reference_exact_probability_and_curve_matches=previous_matches,
                  test_partitions_parsed=[], protocol_sha256=digest)
    BASE.write(run / 'audit.json', result)
    return result


def report(run, protocol, rows, proof):
    groups = []
    for alpha in SCALES:
        for values in BASE.group([r for r in rows if r['alpha'] == alpha], protocol['seeds']):
            groups.append(dict(alpha=alpha, **values))
    selected = {str(alpha): BASE.select([r for r in groups if r['alpha'] == alpha]) for alpha in SCALES}
    lookup = {(r['alpha'], r['branch'], r['lr']): r for r in groups}
    def difference(a, b):
        delta = np.asarray(a['seed_accuracy']) - np.asarray(b['seed_accuracy'])
        return dict(mean_difference_pp=float(delta.mean() * 100),
                    sd_difference_pp=float(delta.std(ddof=1) * 100) if len(delta) > 1 else 0.,
                    positive_seed_pairs=int((delta > 0).sum()),
                    negative_seed_pairs=int((delta < 0).sum()), ties=int((delta == 0).sum()),
                    seed_differences_pp=(100*delta).tolist())
    matched = []
    for b in BRANCHES:
        for lr in protocol['learning_rates']:
            matched.append(dict(branch=b, lr=lr,
                                **difference(lookup[(1., b, lr)], lookup[(.3, b, lr)])))
    retuned = {b: difference(selected['1.0'][b], selected['0.3'][b]) for b in BRANCHES}
    fixed_prior = {b: dict(lr=lr, alpha03=lookup[(.3, b, lr)], alpha1=lookup[(1., b, lr)],
                          **difference(lookup[(1., b, lr)], lookup[(.3, b, lr)]))
                   for b, lr in PREVIOUS_RATES.items() if lr in protocol['learning_rates']}
    topology_gap = {}
    for alpha in SCALES:
        by_rate = [dict(lr=lr, **difference(lookup[(alpha, 'topology', lr)], lookup[(alpha, 'geometry', lr)]))
                   for lr in protocol['learning_rates']]
        topology_gap[str(alpha)] = dict(
            after_separate_lr_selection=difference(selected[str(alpha)]['topology'], selected[str(alpha)]['geometry']),
            same_rate=by_rate, positive_mean_rates=sum(r['mean_difference_pp'] > 0 for r in by_rate))
    bounds = (min(protocol['learning_rates']), max(protocol['learning_rates']))
    result = dict(completed_utc=BASE.now(), scope=protocol['scope'], fits=len(rows), groups=groups,
                  selected=selected, scale_effect_after_lr_selection=retuned,
                  scale_effect_at_previous_rates=fixed_prior, matched_lr_scale_effect=matched,
                  topology_vs_geometry=topology_gap,
                  censored_fits=sum(r['convergence_censored'] for r in rows),
                  selected_rate_at_search_boundary={a: {b: values[b]['lr'] in bounds for b in BRANCHES}
                                                    for a, values in selected.items()},
                  alpha1_better_rate_count={b: sum(r['branch'] == b and r['mean_difference_pp'] > 0 for r in matched)
                                           for b in BRANCHES},
                  audit_passed=proof['passed'],
                  limitation='CV reused for selection; descriptive paired-seed comparisons, no new held-out result or significance claim.')
    BASE.write(run / 'result.json', result)
    labels = dict(geometry='Геометрическая', topology='Топологическая')
    lines = ['# Коэффициент дополнительных признаков: 0.3 против 1', '',
             'Вход: [64 стандартизованных пиксельных признака; α × 8 стандартизованных дополнительных признаков]. '
             'Сравниваются α=0.3 и α=1. α фиксирован и не обучается. Обе MLP имеют 508 параметров и архитектуру 72→6→10.', '',
             f"Свежих обучений: {len(rows)} = 2 коэффициента × 2 представления × {len(protocol['learning_rates'])} learning rate × {len(protocol['seeds'])} инициализаций.",
             'Данные: TRA 1934 изображения для обучения и стандартизации, CV 946 изображений для выбора чекпойнта и LR. '
             'WDEP и WINDEP не распаковывались. CV ранее использовалась, поэтому результаты являются дополнительным валидационным подбором.', '',
             f"AdamW, weight decay 0.01, batch size 256; максимум {protocol['max_epochs']} эпох и patience {protocol['patience']}. "
             'Чекпойнт — минимум CV cross-entropy; LR — максимум средней CV accuracy, затем минимум средней CE и меньший LR. '
             'Начальные веса, порядок батчей и правила обучения одинаковы для обеих величин α при фиксированных seed и LR.', '',
             '## Результаты после отдельного подбора LR', '',
             '| Модель | α | Лучший проверенный LR | Средняя CV accuracy, % | SD, п.п. | Медиана выбранной эпохи |',
             '|---|---:|---:|---:|---:|---:|']
    for b in BRANCHES:
        for alpha in SCALES:
            v = selected[str(alpha)][b]
            lines.append(f"| {labels[b]} | {alpha:g} | {v['lr']:.6g} | {100*v['mean_accuracy']:.3f} | {100*v['sd_accuracy']:.3f} | {v['median_selected_epoch']:g} |")
    for b in BRANCHES:
        effect = retuned[b]
        lines += ['', f"{labels[b]}: изменение accuracy при α=1 вместо α=0.3 после отдельного подбора LR — "
                  f"{effect['mean_difference_pp']:+.3f} п.п.; положительных пар по seed {effect['positive_seed_pairs']}/{len(protocol['seeds'])}."]
    lines += ['', 'SD показывает разброс по инициализациям, а не доверительный интервал качества на новых данных. '
              'Сравнения после подбора LR оценены на той же CV; отдельного утверждения о статистической значимости не делается.', '',
              '## Замена только коэффициента при прежних выбранных LR', '',
              '| Модель | Фиксированный LR | α=0.3, accuracy % | α=1, accuracy % | Разность, п.п. |',
              '|---|---:|---:|---:|---:|']
    for b, v in fixed_prior.items():
        lines.append(f"| {labels[b]} | {v['lr']:.6g} | {100*v['alpha03']['mean_accuracy']:.3f} | {100*v['alpha1']['mean_accuracy']:.3f} | {v['mean_difference_pp']:+.3f} |")
    lines += ['', 'Правило выбора эпохи одинаково, но выбранные эпохи могут различаться между коэффициентами.', '',
              '## Вся сетка', '',
              '| LR | Геометрия α=0.3 | Геометрия α=1 | Δ геометрии, п.п. | Топология α=0.3 | Топология α=1 | Δ топологии, п.п. |',
              '|---:|---:|---:|---:|---:|---:|---:|']
    for lr in protocol['learning_rates']:
        g03, g1 = [lookup[(a, 'geometry', lr)] for a in SCALES]
        t03, t1 = [lookup[(a, 'topology', lr)] for a in SCALES]
        lines.append(f"| {lr:.6g} | {100*g03['mean_accuracy']:.3f} | {100*g1['mean_accuracy']:.3f} | "
                     f"{100*(g1['mean_accuracy']-g03['mean_accuracy']):+.3f} | {100*t03['mean_accuracy']:.3f} | "
                     f"{100*t1['mean_accuracy']:.3f} | {100*(t1['mean_accuracy']-t03['mean_accuracy']):+.3f} |")
    lines += ['', '## Сохраняется ли преимущество топологии', '']
    for alpha in SCALES:
        gap = topology_gap[str(alpha)]
        lines.append(f"- При α={alpha:g}: разность топология − геометрия после отдельного подбора LR "
                     f"{gap['after_separate_lr_selection']['mean_difference_pp']:+.3f} п.п.; "
                     f"при одинаковом LR топология лучше в среднем в {gap['positive_mean_rates']}/{len(protocol['learning_rates'])} случаях.")
    lines += ['', '## Аудит', '',
              f"- Повторно загружены и проверены все {proof['checkpoint_predictions_verified']} моделей; вероятности и метрики совпадают точно.",
              '- Начальные веса совпадают для каждого seed между всеми представлениями, коэффициентами и LR.',
              f"- Для α=0.3 полностью совпали с предыдущим экспериментом {len(proof['alpha03_reference_exact_probability_and_curve_matches'])} наборов вероятностей и кривых обучения.",
              f"- Запусков, остановленных ограничением числа эпох: {result['censored_fits']}.",
              f"- Выбранный LR на границе диапазона: {result['selected_rate_at_search_boundary']}.", '',
              '![Сравнение коэффициентов](feature_scale_comparison.png)', '',
              '## Воспроизведение', '', 'Из корня репозитория:', '', '```bash',
              'uv run --locked python compare_feature_scale.py --workers 4', '```', '',
              'Новый запуск создаёт отдельный каталог runs/feature-scale-ДАТА. '
              'Повторная запись в существующий каталог запрещена без --resume. '
              'Все модели, вероятности, кривые, нормализованные входы и протокол сохранены вместе с результатами.']
    (run / 'report_ru.md').write_text('\n'.join(lines) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout='constrained', sharey=True)
    for ax, branch in zip(axes, BRANCHES):
        for alpha, color in zip(SCALES, ['#a45628', '#166c99']):
            data = sorted([r for r in groups if r['branch'] == branch and r['alpha'] == alpha], key=lambda r: r['lr'])
            ax.errorbar([r['lr'] for r in data], [100*r['mean_accuracy'] for r in data],
                        yerr=[100*r['sd_accuracy'] for r in data], marker='o', capsize=2,
                        linewidth=1.5, color=color, label=f'alpha = {alpha:g}')
        ax.set_xscale('log'); ax.set_xlabel('Learning rate'); ax.grid(alpha=.2); ax.legend()
        ax.set_title('Geometric features' if branch == 'geometry' else 'Topological features')
    axes[0].set_ylabel('CV accuracy, %')
    fig.suptitle(f"Side-feature scaling · 508 parameters · mean ± SD over {len(protocol['seeds'])} seeds\nTRA/CV only; validation reused for selection")
    fig.savefig(run / 'feature_scale_comparison.png', dpi=180)
    fig.savefig(run / 'feature_scale_comparison.svg')
    plt.close(fig)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--learning-rates', type=float, nargs='+')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    parser.add_argument('--max-epochs', type=int, default=6000)
    parser.add_argument('--patience', type=int, default=150)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    repo = args.repo.resolve()
    base = backend(repo)
    rates = sorted(args.learning_rates if args.learning_rates is not None else base.RATES)
    if args.workers < 1 or args.max_epochs < 1 or args.patience < 1:
        parser.error('workers, max-epochs and patience must be positive')
    if any(not np.isfinite(lr) or lr <= 0 for lr in rates):
        parser.error('learning rates must be positive and finite')
    if len(set(rates)) != len(rates) or len(set(args.seeds)) != len(args.seeds):
        parser.error('rates and seeds must be distinct')
    run = (args.output or repo / 'runs' / ('feature-scale-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))).resolve()
    sources = ['tune_learning_rate.py', 'outputs/real_topology/optdigits_pilot.py',
               'outputs/real_topology/optdigits_ablation.py', 'outputs/tiny_topology/tiny_mlp_pilot.py',
               'work/real_data/optdigits.zip', 'work/real_data/optdigits_complement_trainval_v1.npz']
    input_hashes = {name: base.sha(repo / name) for name in sources}
    reference_dir = repo / 'runs/lr-search-20261011'
    reference = None
    if (reference_dir / 'protocol.json').exists():
        old = json.loads((reference_dir / 'protocol.json').read_text())
        for name, value in old['source_sha256'].items():
            assert input_hashes[name] == value, f'Historical input changed: {name}'
        assert input_hashes['tune_learning_rate.py'] == old['runner_sha256']
        reference = dict(path='runs/lr-search-20261011', protocol_sha256=base.sha(reference_dir / 'protocol.json'),
                         result_sha256=base.sha(reference_dir / 'result.json'))
    design = dict(scope='Exploratory paired comparison of alpha=0.3 and alpha=1 on reused TRA/CV; no held-out test evaluation.',
                  architecture='MLP 72→6→10 ReLU, 508 parameters', branches=list(BRANCHES),
                  side_scales=list(SCALES), learning_rates=rates, seeds=sorted(args.seeds),
                  optimizer='AdamW', weight_decay=.01, batch_size=256,
                  max_epochs=args.max_epochs, patience=args.patience, improvement_tolerance=1e-8,
                  checkpoint_selection='Minimum validation CE; earliest tie',
                  rate_selection='Maximum mean validation accuracy over seeds, then minimum mean CE, then smaller LR',
                  comparisons=['Alpha=1 minus alpha=0.3 at every matched LR and branch',
                               'Best LR selected independently for each of four branch/scale combinations',
                               'Scale replacement at previously selected LR: geometry 0.005, topology 0.002',
                               'Topology minus geometry at each alpha, both matched LR and independently selected LR'],
                  previous_selected_rates=PREVIOUS_RATES,
                  normalization='Per-feature mean/std ddof=0 fit only to TRA; small-variance columns zeroed; alpha applied after normalization',
                  rng='torch.manual_seed(seed); independent permutation generator 2026092800+seed',
                  train_n=1934, validation_n=946,
                  parsed_archive_members=['optdigits-orig.tra.Z', 'optdigits-orig.cv.Z'],
                  source_sha256=input_hashes, runner_sha256=base.sha(__file__), reference=reference)
    if run.exists():
        if not args.resume:
            parser.error(f'Output already exists; use a new directory or --resume: {run}')
        protocol = json.loads((run / 'protocol.json').read_text())
        for key, value in design.items():
            assert protocol[key] == value, f'Protocol changed: {key}'
        assert protocol['arrays_sha256'] == base.sha(run / 'arrays.npz')
    else:
        if args.resume:
            parser.error('--resume requires an existing output directory')
        run.mkdir(parents=True)
        (run / 'jobs').mkdir()
        base.prepare(repo, run)
        shutil.copy2(__file__, run / 'compare_feature_scale.py')
        shutil.copy2(repo / 'tune_learning_rate.py', run / 'tune_learning_rate.py')
        protocol = dict(**design, registered_utc=base.now(), arrays_sha256=base.sha(run / 'arrays.npz'),
                        environment=dict(python=platform.python_version(), torch=torch.__version__,
                                         numpy=np.__version__, platform=platform.platform()))
        base.write(run / 'protocol.json', protocol)
    digest = base.sha(run / 'protocol.json')
    configs = [dict(id=f'a{ai}_{b}_lr{index:02d}_s{seed}', alpha=alpha, branch=b, lr=lr, seed=seed)
               for index, lr in enumerate(rates) for seed in protocol['seeds']
               for b in BRANCHES for ai, alpha in enumerate(SCALES)]
    started, rows = time.perf_counter(), []
    base.write(run / 'status.json', dict(status='running', updated_utc=base.now(), completed=0, total=len(configs)))
    print(json.dumps(dict(output=str(run), fits=len(configs), protocol_sha256=digest)), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                             initializer=initialize, initargs=(str(repo), str(run))) as pool:
        futures = [pool.submit(fit, c, str(run), digest, args.max_epochs, args.patience) for c in configs]
        for future in as_completed(futures):
            rows.append(future.result())
            base.write(run / 'status.json', dict(status='running', updated_utc=base.now(), completed=len(rows),
                                                total=len(configs), elapsed_seconds=time.perf_counter() - started,
                                                latest=rows[-1]['id']))
            if len(rows) % 20 == 0 or len(rows) == len(configs):
                print(json.dumps(dict(completed=len(rows), total=len(configs),
                                      seconds=round(time.perf_counter() - started, 1))), flush=True)
    proof = audit(repo, run, rows, digest, protocol)
    result = report(run, protocol, rows, proof)
    base.write(run / 'status.json', dict(status='completed', updated_utc=base.now(), completed=len(rows),
                                        total=len(configs), elapsed_seconds=time.perf_counter() - started,
                                        audit_passed=True))
    print(json.dumps(dict(selected=result['selected'], scale_effect=result['scale_effect_after_lr_selection'],
                          topology_vs_geometry=result['topology_vs_geometry'], censored_fits=result['censored_fits']),
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
