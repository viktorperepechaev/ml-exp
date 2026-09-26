"""Finish audit, export and report after the fixed statistical analysis."""
from pathlib import Path
from datetime import datetime,timezone
import json,sys,time,subprocess,traceback,os
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/tiny_topology'

def status(stage,**kw):
    p=OUT/'tiny_artifact_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(updated_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),stage=stage,**kw),indent=2)+'\n');t.replace(p)

def main():
    while not (OUT/'tiny_confirmation_test_results.json').exists():
        st=json.loads((OUT/'tiny_continuation_status.json').read_text())
        if st['stage']=='failed':raise RuntimeError(st['error'])
        status('waiting_for_frozen_test_analysis');time.sleep(15)
    with (ROOT/'work/tiny_topology/artifact_generation.log').open('a') as log:
        for stage,script,args in [('independent_final_audit','tiny_final_audit.py',[]),('exporting_fixed_seed_weights','tiny_export.py',[]),('generating_scientific_figures','tiny_figures.py',['--final']),('writing_report','tiny_report.py',[])]:
            status(stage);subprocess.run([sys.executable,str(OUT/script),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True);log.flush()
    status('ready_for_visual_review');print('Report and figures ready for visual review.',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=str(exc));traceback.print_exc();sys.exit(1)
