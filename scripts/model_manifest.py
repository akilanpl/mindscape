"""Record downloaded open-backbone file identities without bundling model weights."""
import argparse,hashlib,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.model);out=Path(a.output)
 if out.exists():raise ValueError('Refuse overwrite')
 files=[]
 for f in sorted(root.iterdir()):
  if not f.is_file():continue
  h=hashlib.sha256()
  with f.open('rb') as s:
   for b in iter(lambda:s.read(1048576),b''):h.update(b)
  files.append(dict(name=f.name,bytes=f.stat().st_size,sha256=h.hexdigest()))
 out.write_text(json.dumps(dict(model='HuggingFaceTB/SmolLM2-135M-Instruct',revision=root.name,license='Apache-2.0',source='https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct',files=files),indent=2));print(len(files),'backbone files recorded')
if __name__=='__main__':main()
