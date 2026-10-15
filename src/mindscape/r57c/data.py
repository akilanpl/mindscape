"""New final exploratory problems, generated before native fitting; no official data."""
import hashlib,json,random,shutil
from pathlib import Path
from mindscape.r57.data import identity,_math_candidates,labels

SEEDS=(17,29,43)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def row(p,scope,family):return {'identity':identity(p),'problem':p,'labels':labels(p),'scope':scope,'family':family,'split':'final'}

def prepare(output,old,legacy):
    output,old,legacy=Path(output),Path(old),Path(legacy)
    output.mkdir(exist_ok=False)
    (output/'train').mkdir();(output/'evaluator-only').mkdir()
    blocked=set();oldrows=[]
    for seed in SEEDS:
        for path in (old/f'data-v1/train/{seed}.json',old/f'data-v1/dev/{seed}.json',old/f'data-v1/evaluator-only/final-{seed}.json',legacy/f'remaining-eight-v1/shared-validation-{seed}.json'):
            previous=json.loads(path.read_text());oldrows.extend(previous)
            blocked|={identity(r['problem']) for r in previous}
        shutil.copyfile(old/f'data-v1/train/{seed}.json',output/f'train/{seed}.json')
    # Use ALL remaining contexts, not a score-selected subset. Shared across seeds.
    math=[p for p in _math_candidates() if identity(p) not in blocked]
    if len(math)<20:raise RuntimeError('Insufficient globally unseen arithmetic contexts')
    manifest={'status':'EXPLORATORY_UNTOUCHED_AT_GENERATION','files':{},'seeds':list(SEEDS),'math_contexts':len(math),
        'math_scope':'all globally unused digit/operation/incoming contexts after excluding all prior train/dev/final and original exploratory validation; identical contexts across seeds',
        'training_policy':'exact unchanged legacy128 bytes/order per seed','final_access':'evaluator only after all nine native fits complete; logical separation, not OS isolation','counts':{}}
    for seed in SEEDS:
        rng=random.Random(90171357+seed);rows=[row(p,'unseen-combination','math-'+p['operation']) for p in math]
        for family in range(12):
            if family%3==0:
                stem='identifier_'+str(seed)+'_'+str(family)+'_';names=[stem+s for s in ('aaa','aab','aba','baa')];scope='misleading-prefix-suffix'
            elif family%3==1:
                names=['alpha_'+str(seed)+'_'+str(family), 'δelta_'+str(seed)+'_'+str(family), 'λambda_'+str(seed)+'_'+str(family), '变量_'+str(seed)+'_'+str(family)];scope='novel-unicode'
            else:
                names=['n_'+''.join(rng.choices('abcdefghijklmnopqrstuvwxyz',k=k)) for k in (3,7,11,19)];scope='novel-variable-length'
            rng.shuffle(names)
            binding=f'fresh_{seed}_{family}'
            sources=[f'def f({binding}):\n return ({binding} + 3) * 2',f'{binding}=3\nvalues=[{binding}, {binding}+1]',f'def f({binding}):\n if {binding}:\n  return {binding}-1\n return 0']
            for target in range(4):
                for rotation in range(4):
                    candidates=names[rotation:]+names[:rotation]
                    p={'domain':'coding','operation':'rename','source':sources[family%3],'old_name':binding,
                       'candidates':candidates,'requested_name':names[target],'concept_ids':[3]}
                    rows.append(row(p,scope,f'coding-{seed}-{family}'))
        for index in range(96):
            ood=index>=48
            mass=rng.uniform(.25,.75) if ood else rng.uniform(1.,9.)
            dt=rng.uniform(1.1,2.) if ood else rng.uniform(.1,1.)
            force=[rng.uniform(-12.,12.),rng.uniform(-12.,12.)]
            p={'domain':'physics','operation':'force','mass':mass,'dt':dt,'force':force,
               'velocity':[rng.uniform(-5.,5.),rng.uniform(-5.,5.)],'concept_ids':[2]}
            rows.append(row(p,'ood-low-mass-long-dt' if ood else 'fresh-iid',f'physics-{seed}-{index}'))
        ids=[r['identity'] for r in rows]
        if len(ids)!=len(set(ids)) or set(ids)&blocked:raise RuntimeError('New final duplicates prior data')
        # Request identity/lexical rank AND answer position are balanced.
        coding=[r for r in rows if r['problem']['domain']=='coding']
        positions=[sum(r['labels']['coding_index']==i for r in coding) for i in range(4)]
        ranks=[sum(sorted(r['problem']['candidates']).index(r['problem']['requested_name'])==i for r in coding) for i in range(4)]
        assert positions==ranks==[48]*4
        destination=output/f'evaluator-only/final-{seed}.json';destination.write_text(json.dumps(rows,indent=2,ensure_ascii=False))
        manifest['counts'][str(seed)]={'math':len(math),'coding':len(coding),'physics':96,'coding_positions':positions,'coding_lexical_ranks':ranks,'prior_identity_overlap':0}
    for path in sorted(output.rglob('*.json')):manifest['files'][str(path.relative_to(output))]=sha(path)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2));return manifest
