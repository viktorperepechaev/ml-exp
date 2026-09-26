"""Verify the transferred bundle without changing historical result files."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((ROOT / 'TRANSFER_MANIFEST.json').read_text())
    for name, expected in manifest['files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError(f'Invalid manifest path: {name}')
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f'File changed: {name}')
    print(f"Verified hashes of {len(manifest['files'])} files.", flush=True)
    runs = []
    with tempfile.TemporaryDirectory(prefix='topological-invariants-check-') as td:
        scratch = Path(td)
        for folder in ['outputs', 'work']:
            shutil.copytree(ROOT / folder, scratch / folder)
        env = os.environ.copy()
        env['MPLCONFIGDIR'] = str(scratch / 'matplotlib-cache')
        env['XDG_CACHE_HOME'] = str(scratch / 'cache')
        for script in ['outputs/tiny_topology/tiny_final_audit.py',
                       'outputs/learned_topology/audit_results.py']:
            print(f'Checking {script}', flush=True)
            result = subprocess.run([sys.executable, str(scratch / script)],
                                    cwd=scratch, env=env, capture_output=True, text=True)
            runs.append(dict(script=script, returncode=result.returncode,
                             stdout=result.stdout, stderr=result.stderr))
            print(result.stdout, end='', flush=True)
            if result.stderr:
                print(result.stderr, file=sys.stderr, end='', flush=True)
            if result.returncode:
                break
    passed = len(runs) == 2 and all(r['returncode'] == 0 for r in runs)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  passed=passed, manifest_files_checked=len(manifest['files']),
                  python=sys.version, runs=runs,
                  scope='Replay saved results in a temporary copy; no training from scratch.')
    (ROOT / 'verification').mkdir(exist_ok=True)
    (ROOT / 'verification/latest.json').write_text(json.dumps(report, indent=2)+'\n')
    if not passed:
        raise SystemExit(1)
    print('Both experiment checks passed. See verification/latest.json.', flush=True)


if __name__ == '__main__':
    main()
