"""Digest final artifacts after validation. Refuses an existing freeze record."""
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('results');a=p.parse_args();root=Path(a.results);out=root/'freeze_manifest.json'
 if out.exists():raise ValueError('Final artifacts already frozen')
 audit=json.loads((root/'fairness_audit.json').read_text())
 if not audit['all_passed']:raise ValueError('Failed audit')
 hashes=[]
 for f in sorted(root.rglob('*')):
  if not f.is_file():continue
  h=hashlib.sha256()
  with f.open('rb') as stream:
   for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
  hashes.append(dict(path=str(f.relative_to(root)),bytes=f.stat().st_size,sha256=h.hexdigest()))
 out.write_text(json.dumps(dict(timestamp=datetime.now(timezone.utc).isoformat(),files=hashes,interpretation='Frozen measured study; generation refuses existing output roots. Hashes cover checkpoints, metadata, predictions, statistics and plots. Never overwrite this snapshot.'),indent=2));print(len(hashes),'artifacts frozen')
if __name__=='__main__':main()
