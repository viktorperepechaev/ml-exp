"""Freshly train and audit all 102 models of the learned Euler-layer experiment."""
from __future__ import annotations

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'outputs/learned_topology'
sys.path.insert(0, str(SOURCE))
import pilot
from euler_layer import SmallClassifier
from single_field import SingleFieldClassifier

SINGLE_KINDS = ['euler_learned', 'euler_frozen', 'area_learned',
                'perimeter_learned', 'geometry_learned']
MAIN_KINDS = list(pilot.KINDS)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2)+'\n')


def initialize_worker(run):
    pilot.CACHE = Path(run) / 'work/learned_topology/train_cv.npz'
    pilot.init()


def fit(run, family, config, digest):
    # Explicitly reset both globals: a process can alternate model families.
    pilot.SmallClassifier = SingleFieldClassifier if family == 'single_field' else SmallClassifier
    jobs = 'single_field_jobs' if family == 'single_field' else 'jobs'
    pilot.JOBS = Path(run) / 'work/learned_topology' / jobs
    return family, pilot.fit(config, digest)


def check_inputs():
    protocols = {family: json.loads((SOURCE / family / 'pilot_protocol.json').read_text())
                 for family in ['', 'single_field']}
    for protocol in protocols.values():
        for name, expected in protocol['source_sha256'].items():
            assert sha(SOURCE / name) == expected, name
    assert protocols['single_field']['inherited_training_protocol_sha256'] == sha(SOURCE / 'pilot_protocol.json')
    assert sha(ROOT / 'work/real_data/optdigits.zip') == protocols['']['source_data_sha256']
    cache = ROOT / 'work/learned_topology/train_cv.npz'
    assert sha(cache) == protocols['']['cache_sha256']
    manifest = json.loads((ROOT / 'TRANSFER_MANIFEST.json').read_text())['files']
    for name in ['outputs/learned_topology/audit_results.py', 'outputs/real_topology/optdigits_pilot.py']:
        assert sha(ROOT / name) == manifest[name], name
    # Rebuild every input/scaling array from TRA/CV, without reading a test split.
    rebuilt = {}
    for name, split in [('train', 'tra'), ('validation', 'cv')]:
        images, labels = pilot.read_original(split)
        rebuilt[name+'_x'] = images.reshape(-1, 8, 4, 8, 4).mean((2, 4)).astype(np.float32)[:, None]
        rebuilt[name+'_y'] = labels
    x = rebuilt['train_x'].reshape(-1, 64)
    mean, std = x.mean(0), x.std(0)
    keep = std >= 1e-10
    rebuilt.update(mean=mean, std=np.where(keep, std, 1), keep=keep)
    with np.load(cache) as saved:
        assert set(saved.files) == set(rebuilt)
        for key, value in rebuilt.items():
            np.testing.assert_array_equal(value, saved[key])
    assert len(protocols['']['configs']) == 72 and len(protocols['single_field']['configs']) == 30
    assert len({cfg['id'] for p in protocols.values() for cfg in p['configs']}) == 102
    return protocols


def copy_inputs(run):
    files = ['outputs/real_topology/optdigits_pilot.py', 'work/real_data/optdigits.zip',
             'work/learned_topology/train_cv.npz']
    files += ['outputs/learned_topology/'+name for name in [
        'euler_layer.py', 'single_field.py', 'pilot.py', 'single_field_pilot.py',
        'check_layer.py', 'layer_checks.json', 'audit_results.py',
        'pilot_protocol.json', 'single_field/pilot_protocol.json']]
    for name in files:
        destination = run / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    # No archived checkpoints, predictions, results or job receipts are copied.
    for jobs in ['jobs', 'single_field_jobs']:
        (run / 'work/learned_topology' / jobs).mkdir()
    (run / 'logs').mkdir()
    (run / 'math_checks').mkdir()
    for name in ['euler_layer.py', 'check_layer.py']:
        shutil.copy2(SOURCE / name, run / 'math_checks' / name)
    return {name: sha(run / name) for name in files}


def subprocess_check(run, script, logfile):
    env = os.environ.copy()
    env['MPLCONFIGDIR'] = str(run / '.cache/matplotlib')
    env['XDG_CACHE_HOME'] = str(run / '.cache')
    with (run / 'logs' / logfile).open('w') as log:
        completed = subprocess.run([sys.executable, str(run / script)], cwd=run,
                                   env=env, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f'Check failed; see {run / "logs" / logfile}')


def finish(run, rows, digests, started):
    for family in ['', 'single_field']:
        destination = run / 'outputs/learned_topology' / family
        pilot.OUT = destination
        pilot.KINDS = SINGLE_KINDS if family else MAIN_KINDS
        with (run / 'logs' / ('single-summary.log' if family else 'main-summary.log')).open('w') as log:
            with redirect_stdout(log):
                pilot.summarize(rows[family], digests[family])
        write(destination / 'pilot_status.json', dict(stage='completed', completed=len(rows[family]), total=len(rows[family])))
    print('Auditing gradients, parameter changes and all 102 new checkpoints...', flush=True)
    subprocess_check(run, 'outputs/learned_topology/audit_results.py', 'audit.log')
    audit = json.loads((run / 'outputs/learned_topology/results_audit.json').read_text())
    assert audit['passed'] and audit['runs_replayed'] == 102
    main = json.loads((run / 'outputs/learned_topology/pilot_summary.json').read_text())['selected']
    single = json.loads((run / 'outputs/learned_topology/single_field/pilot_summary.json').read_text())['selected']
    # Rank actual new results; do not assume that historical winners remain best.
    rank = lambda r: (-r['mean_accuracy'], r['mean_ce'], r['trainable_parameters'], r['id'])
    best_topology = min([main['euler_learned'], single['euler_learned']], key=rank)
    controls = [r for selection in [main, single] for key, r in selection.items() if not key.startswith('euler_')]
    best_control = min(controls, key=rank)
    best_cnn = min([r for key, r in main.items() if key.startswith('cnn_')], key=rank)
    table = [('Обычная MLP', main['mlp']),
             ('Двухполевой обучаемый эйлеров слой', main['euler_learned']),
             ('Однополевой замороженный эйлеров слой', single['euler_frozen']),
             ('Однополевой обучаемый эйлеров слой', single['euler_learned']),
             ('Однополевой обучаемый слой площади', single['area_learned']),
             ('Однополевой обучаемый слой периметра', single['perimeter_learned']),
             ('Однополевой слой площади и периметра', single['geometry_learned']),
             ('Лучшая проверенная CNN', best_cnn)]
    historical = []
    for family in ['', 'single_field']:
        jobs = 'single_field_jobs' if family else 'jobs'
        for row in rows[family]:
            relative = Path('work/learned_topology') / jobs / (row['id']+'.npz')
            with np.load(run / relative) as new, np.load(ROOT / relative) as old:
                historical.append(dict(id=row['id'],
                    predictions_equal=bool(np.array_equal(new['predictions'], old['predictions'])),
                    max_probability_difference=float(np.max(np.abs(new['probabilities']-old['probabilities'])))))
    result = dict(completed_utc=pilot.now(), fits=102, train_n=1934, validation_n=946,
                  selected_main=main, selected_single_field=single,
                  best_learned_topology=best_topology, best_nontopological_control=best_control,
                  difference_percentage_points=100*(best_topology['mean_accuracy']-best_control['mean_accuracy']),
                  all_new_checkpoint_checks_passed=True, hardcap_unresolved=audit['hardcap_unresolved'],
                  historical_comparison=historical, elapsed_seconds=time.perf_counter()-started,
                  test_read=False, scope='Fresh reproduction of the historical 102-fit development grid; '
                  'model/learning-rate/checkpoint selection on CV. No independent confirmation or significance claim.')
    write(run / 'result.json', result)
    lines = ['# Повторное обучение эксперимента с обучаемым топологическим слоем', '',
             'Все 102 модели обучены заново. TRA: 1934 изображения; CV: 946. Все получают одинаковый вход 8×8.', '',
             '| Модель | Обучаемых параметров | Средняя точность CV |', '|---|---:|---:|']
    lines += [f'| {name} | {r["trainable_parameters"]} | {100*r["mean_accuracy"]:.3f}% |' for name, r in table]
    lines += ['', 'Указаны средние трёх инициализаций для скорости обучения, выбранной по CV. '
              'На этой же CV выбраны контрольные точки. Это повтор разработки, а не независимый тест.', '',
              f'Лучший обучаемый топологический вариант минус лучший нетопологический контроль: '
              f'{result["difference_percentage_points"]:+.3f} п.п.', '',
              'Исходный результат не показывал преимущества над CNN; этот повтор не предполагает заранее положительного исхода.', '',
              'Подробности — `result.json`; новые веса и предсказания — `work/learned_topology/`; '
              'результаты проверки — `outputs/learned_topology/results_audit.json`.', '',
              'Графики в `outputs/learned_topology/` используют исторически заданные примеры '
              '(CNN с двумя каналами, иллюстрация эйлерова слоя при seed 0). Таблица выше выбирает лучшие новые результаты.']
    report = '\n'.join(lines)+'\n'
    (run / 'report_ru.md').write_text(report)
    write(run / 'status.json', dict(stage='completed', completed=102, total=102))
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4, help='parallel CPU training processes (default: 4)')
    parser.add_argument('--output', type=Path, help='new output directory; default: runs/learned-TIMESTAMP')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be at least 1')
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run = (args.output or ROOT / 'runs' / f'learned-{timestamp}').resolve()
    if run.exists():
        parser.error(f'Output already exists; choose a new directory: {run}')
    print('Verifying sources and rebuilding TRA/CV arrays from raw data...', flush=True)
    protocols = check_inputs()
    run.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    copied = copy_inputs(run)
    digests = {family: sha(SOURCE / family / 'pilot_protocol.json') for family in protocols}
    write(run / 'run_protocol.json', dict(created_utc=pilot.now(), runner_sha256=sha(__file__),
          configuration_reference_sha256=digests, input_file_sha256=copied,
          workers=args.workers, python=sys.version, numpy=np.__version__, torch=torch.__version__,
          purpose='Fresh 72+30 fits using frozen fit/summarize functions; copied protocols are historical '
                  'configuration references, not newly preregistered hypotheses.',
          raw_train_cv_arrays_recomputed_and_equal=True, old_weights_copied=False, test_read=False))
    print(f'New run: {run}', flush=True)
    print('Checking the Euler formula and numerical gradients...', flush=True)
    subprocess_check(run, 'math_checks/check_layer.py', 'math-checks.log')
    math_checks = json.loads((run / 'math_checks/layer_checks.json').read_text())
    assert math_checks['passed'] and math_checks['code_sha256'] == sha(SOURCE / 'euler_layer.py')
    rows = {'': [], 'single_field': []}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                             initializer=initialize_worker, initargs=(str(run),)) as pool:
        futures = [pool.submit(fit, str(run), family, cfg, digests[family])
                   for family, protocol in protocols.items() for cfg in protocol['configs']]
        for count, future in enumerate(as_completed(futures), 1):
            family, row = future.result()
            rows[family].append(row)
            write(run / 'outputs/learned_topology' / family / 'pilot_results.json', dict(
                completed=len(rows[family]), total=len(protocols[family]['configs']),
                protocol_sha256=digests[family], runs=rows[family], test_read=False))
            write(run / 'status.json', dict(stage='training', completed=count, total=102, last_completed=row['id']))
            print(f'{count}/102 trained: {row["id"]}; CV={100*row["accuracy"]:.3f}%', flush=True)
    finish(run, rows, digests, started)
    print(f'Results saved to: {run}', flush=True)


if __name__ == '__main__':
    main()
