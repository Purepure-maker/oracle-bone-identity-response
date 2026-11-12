"""Verify separately distributed frozen states before image inference."""
from pathlib import Path
import argparse,json,hashlib

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,default=Path(__file__).resolve().parents[1]/'assets')
    a=p.parse_args();manifest=json.loads((a.assets/'MANIFEST.json').read_text(encoding='utf8'))
    rows=[]
    for name,record in manifest['assets'].items():
        path=a.assets/name
        if not path.is_file():raise SystemExit(f'Missing {name}; extract the matching Online Resource 4 into assets/.')
        if path.stat().st_size!=record['bytes'] or sha(path)!=record['sha256']:raise SystemExit('Asset content mismatch: '+name)
        rows.append(name)
    print(json.dumps({'state':'PASS','verified_assets':rows}))

if __name__=='__main__':main()
