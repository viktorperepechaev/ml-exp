"""Recover feasible jobs without changing any fit or selecting by outcomes."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import json
import numpy as np
import tiny_mlp_pilot as m


def main():
 p=m.freeze();audit=json.loads((m.OUT/'tiny_mlp_feasibility_audit.json').read_text());digest=m.sha(m.PROTOCOL);assert audit['protocol_sha256']==digest
 valid=audit['valid_configurations'];missing=[c for c in valid if not (m.WORK/(c['id']+'.json')).exists()]
 m.write(m.OUT/'tiny_mlp_execution_amendment.json',dict(created_utc=m.now(),protocol_sha256=digest,
  reason='Initial orchestration interrupted while feasibility of256parameter late-HOG configurations was checked. Preserve every completed feasible fit; run only missing feasible jobs. All24excluded jobs are algebraically impossible under their budget, identically for geometry and topology. No outcome-based exclusion or change in training.',
  preexisting_fits=len(valid)-len(missing),missing_ids=[c['id'] for c in missing],feasibility_audit_sha256=m.sha(m.OUT/'tiny_mlp_feasibility_audit.json')))
 if missing:
  with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=m.initialize) as pool:
   for f in as_completed([pool.submit(m.fit,c,digest) for c in missing]):f.result()
 rows=[json.loads((m.WORK/(c['id']+'.json')).read_text()) for c in valid]
 with np.load(m.CACHE) as z:y=z['validation_y']
 for r in rows:
  assert r['protocol_sha256']==digest and r['parameters']<=r['budget']
  with np.load(m.WORK/(r['id']+'.npz')) as z:
   pred=z['predictions'];assert abs((pred==y).mean()-r['accuracy'])<1e-12;np.testing.assert_array_equal(z['probabilities'].argmax(1),pred)
  assert r['ce']<=min(c[0] for c in r['curve'])+1.1e-8
 m.write(m.RESULT,dict(protocol_sha256=digest,test_read=False,completed=len(rows),total_feasible=len(valid),total_nominal=480,infeasible=24,runs=rows))
 m.summarize(rows,digest)
 m.write(m.OUT/'tiny_mlp_status.json',dict(updated_utc=m.now(),status='completed',completed=len(rows),total_feasible=len(valid),all_predictions_audited=True))


if __name__=='__main__':main()
