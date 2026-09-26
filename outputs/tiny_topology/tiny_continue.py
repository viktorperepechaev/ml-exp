"""Finite continuation of the already running tiny-network experiment."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/tiny_topology';PYTHON=ROOT/'work/real-env/bin/python';LOG=ROOT/'work/tiny_topology/confirmation_continuation.log'

def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def status(stage,**kw):
    obj=dict(updated_utc=now(),pid=os.getpid(),stage=stage,**kw)
    p=OUT/'tiny_continuation_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(p)
def call(script,*args):
    with LOG.open('a') as f:
        f.write('\n'+now()+' '+script+' '+' '.join(args)+'\n');f.flush()
        subprocess.run([str(PYTHON),str(OUT/script),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)

def main():
    plan=OUT/'tiny_continuation_plan.json'
    if not plan.exists():
        plan.write_text(json.dumps(dict(created_utc=now(),scope='Finish current authorized finite experiment, no recurring task.',steps=['Wait for all288existing CNN fits; do not restart training','Audit all744development prediction files','Replay selected checkpoints and freeze primary plus descriptive capacity curve','Train all80fixed final models before WDEPaccess','Seal predictions, then compute preregistered statistics'],code_sha256={p.name:sha(p) for p in [Path(__file__),OUT/'tiny_confirmation.py',OUT/'tiny_audit.py']},test_read=False),indent=2)+'\n')
    recorded=json.loads(plan.read_text())
    for name,digest in recorded['code_sha256'].items():assert sha(OUT/name)==digest
    while True:
        st=json.loads((OUT/'tiny_cnn_status.json').read_text())
        if st.get('status')=='completed' and st['completed']==288 and (OUT/'tiny_cnn_summary.json').exists():break
        status('waiting_for_existing_cnn_fits',completed=st['completed'],total=st['total']);time.sleep(15)
    status('auditing_development_predictions');call('tiny_audit.py','--complete')
    if not (OUT/'tiny_confirmation_protocol.json').exists():
        status('freezing_selections');call('tiny_confirmation.py','freeze')
    if not (OUT/'tiny_confirmation_weights_receipt.json').exists():
        status('refitting_registered_models');call('tiny_confirmation.py','train','--workers','4')
    if not (OUT/'tiny_confirmation_prediction_receipt.json').exists():
        status('predicting_fresh_wdep_test');call('tiny_confirmation.py','predict')
    if not (OUT/'tiny_confirmation_test_results.json').exists():
        status('analyzing_frozen_predictions');call('tiny_confirmation.py','stats')
    r=json.loads((OUT/'tiny_confirmation_test_results.json').read_text());status('completed',success_gate_passed=r['success_gate_passed'],primary=r['primary'])
    print(json.dumps(dict(stage='completed',success_gate_passed=r['success_gate_passed'],primary=r['primary'])),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        status('failed',error=str(exc));traceback.print_exc();sys.exit(1)
