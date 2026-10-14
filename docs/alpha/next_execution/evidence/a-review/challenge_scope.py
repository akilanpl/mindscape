import importlib.util, tempfile, subprocess,json
from pathlib import Path
p=Path('work/next-sprint-audit/research/alpha/sprint_scope_audit.py').resolve()
s=importlib.util.spec_from_file_location('scope',p); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
results=[]
def git(root,*args): return subprocess.check_output(['git','-C',str(root),*args],stderr=subprocess.DEVNULL).decode().strip()
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp)/'repo';root.mkdir();git(root,'init','-q');git(root,'config','user.email','fixture@example.test');git(root,'config','user.name','Fixture')
 data=Path('work/next-r54-review/experiments/r5_4/frozen/protocol.json').read_bytes();(root/'protocol.json').write_bytes(data);(root/'f').write_text('frozen');git(root,'add','.');git(root,'commit','-qm','fixture');base=git(root,'rev-parse','HEAD')
 m.PINS['r54']=base;m.PROTOCOLS['r54']=('protocol.json',m.sha(data));m.LOCKS={}
 def check(name,want_fail):
  try: m.audit_one(root,'r54'); rejected=False
  except (ValueError,subprocess.CalledProcessError): rejected=True
  assert rejected==want_fail,(name,rejected)
  results.append({'case':name,'rejected':rejected})
 check('exact baseline',False)
 (root/'f').write_text('dirty');check('dirty baseline',True);git(root,'checkout','--','f')
 (root/'f').write_text('modified');git(root,'add','f');git(root,'commit','-qm','modify');check('committed modification',True);git(root,'reset','--hard',base)
 (root/'new.py').write_text('candidate');git(root,'add','.');git(root,'commit','-qm','add');check('committed addition',False)
 (root/'new.py').write_text('changed after commit');check('dirty added candidate currently not checked',False)
 m.PINS['r54']='0'*40;check('wrong ancestry',True);m.PINS['r54']=base
 link=Path(tmp)/'linked';link.symlink_to(root,target_is_directory=True);m.audit_one(link,'r54');results.append({'case':'supplied symlink root currently accepted','rejected':False})
print(json.dumps(results,indent=2))
