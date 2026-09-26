"""Package the completed experiment with data attribution and relative paths."""
from pathlib import Path
import hashlib,json,zipfile
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/tiny_topology';WORK=ROOT/'work/tiny_topology'

def main():
    assert (OUT/'tiny_networks_report_ru.md').exists()
    assert json.loads((OUT/'tiny_confirmation_final_audit.json').read_text())['passed']
    files=[]
    files += [p for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='package_manifest.json' and not p.name.endswith('.tmp')]
    files += [p for p in WORK.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in {'.npz','.pt','.json'} and not p.name.endswith('.tmp')]
    files += [ROOT/'outputs/real_topology'/n for n in ['optdigits_pilot.py','optdigits_ablation.py']]
    files += [ROOT/'outputs/topology_complement'/n for n in ['optdigits_complement_v1_protocol.json','optdigits_complement_v1.py']]
    files += [ROOT/'work/real_data'/n for n in ['optdigits.zip','optdigits_complement_trainval_v1.npz']]
    unique=sorted(set(files));manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),credit='OptDigits: E. Alpaydin and C. Kaynak (1998), UCI, DOI10.24432/C50P49, CC BY4.0. No RIM-ONE data included.',files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in unique})
    m=OUT/'package_manifest.json';m.write_text(json.dumps(manifest,indent=2)+'\n');unique.append(m)
    dest=ROOT/'outputs/tiny_topology_study.zip'
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in unique:z.write(p,str(p.relative_to(ROOT)))
    with zipfile.ZipFile(dest) as z:assert z.testzip() is None
    print(json.dumps(dict(file=str(dest),files=len(unique),bytes=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest())),flush=True)
if __name__=='__main__':main()
