"""Synthetic labelled language preparation and externally gated learning infrastructure.

Preparation and engineering loss/backward checks never fit parameters. No answer
text is generated, and trained language interpretation remains unqualified.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import string
from .language_model import ARGUMENT_SLOTS, INTENTS, FrozenVocabulary, build_language_encoder, tokenize

STATUS = 'PREPARATION ONLY — LANGUAGE GENERALIZATION UNVERIFIED'


@dataclass(frozen=True)
class LanguageExample:
    text: str
    intent: str
    spans: dict[str, tuple[int, int]]
    split: str
    template_id: str
    structure_id: str
    provenance: str = 'synthetic template annotations; not independently human-labelled language'


def labelled_form(template, values, intent, *, split, template_id, structure_id):
    """Track argument boundaries while rendering; never infer labels from answers."""
    if intent not in INTENTS: raise ValueError('Unsupported intent label')
    text = ''; spans = {}
    for literal, field, spec, conversion in string.Formatter().parse(template):
        text += literal
        if field is None: continue
        if spec or conversion or field not in ARGUMENT_SLOTS or field not in values:
            raise ValueError('Only known literal argument slots are supported')
        if field in spans: raise ValueError('Repeated slots need a separate annotation protocol')
        value = str(values[field])
        if not value or len(value)>64: raise ValueError('Bounded nonempty slot text required')
        # BOS occupies token zero. Templates keep argument token boundaries separate.
        start = 1+len(tokenize(text)); text += value
        end = len(tokenize(text))
        if end < start: raise ValueError('Argument token boundary merged with literal')
        spans[field] = (start,end)
    if len(tokenize(text)) >=128: raise ValueError('Annotated form exceeds encoder token budget')
    return LanguageExample(text,intent,spans,split,template_id,structure_id)


_TEMPLATES = {
 'train': {
  'add': 'add {left} and {right}',
  'borrow_digit': 'subtract digit {right} from {left}',
  'rename': 'rename {old_name} to {requested_name}',
  'force': 'mass {mass} force ( {force_x} , {force_y} ) velocity ( {velocity_x} , {velocity_y} ) time {dt}',
  'tool': 'inspect this program', 'unsupported': 'tell me a bedtime story'},
 'paraphrase': {
  'add':'please calculate the sum of {left} with {right}',
  'borrow_digit':'take digit {right} away from {left}',
  'rename':'change identifier {old_name} into {requested_name}',
  'force':'apply force ( {force_x} , {force_y} ) to mass {mass} moving ( {velocity_x} , {velocity_y} ) for {dt}',
  'tool':'check the following source', 'unsupported':'compose a poem about clouds'},
 'structure': {
  'add':'first operand [ {left} ] ; second operand [ {right} ] ; return their sum',
  'borrow_digit':'given minuend [ {left} ] and subtrahend [ {right} ] ; find the digit difference',
  'rename':'requested identifier [ {requested_name} ] ; original identifier [ {old_name} ] ; perform rename',
  'force':'duration [ {dt} ] ; velocity [ {velocity_x} , {velocity_y} ] ; mass [ {mass} ] ; force [ {force_x} , {force_y} ]',
  'tool':'source inspection is the requested operation', 'unsupported':'a fictional narrative is requested'},
 'vocabulary': {
  'add':'combine numerals {left} plus {right}',
  'borrow_digit':'deduct numeral {right} off {left}',
  'rename':'substitute symbol {old_name} with {requested_name}',
  'force':'inertial mass {mass} ; thrust {force_x} , {force_y} ; motion {velocity_x} , {velocity_y} ; elapsed {dt}',
  'tool':'audit executable notation', 'unsupported':'imagine an enchanted landscape'}}


def build_language_dataset():
    """Fixed synthetic train/paraphrase/structure/vocabulary-combination partitions."""
    data={name:[] for name in _TEMPLATES}
    train_pairs=[('12','34'),('56','78'),('90','12'),('34','56')]
    held_pairs=[('19','82'),('47','63'),('50','29'),('86','71')]
    for split,templates in _TEMPLATES.items():
        for intent,template in templates.items():
            for index in range(4 if intent not in ('tool','unsupported') else 1):
                left,right=(held_pairs if split in ('structure','vocabulary') else train_pairs)[index]
                names=(('x','value'),('item','count'),('size','total'),('index','result'))
                old,new=names[index]
                if split=='vocabulary': old,new=('neuron','activation') if index%2==0 else ('matrix','tensor')
                values={'left':left if intent=='add' else left[0], 'right':right if intent=='add' else right[0],
                        'mass':str(index+1), 'force_x':str(index+2), 'force_y':'0',
                        'velocity_x':'1','velocity_y':'0','dt':'2','old_name':old,'requested_name':new}
                data[split].append(asdict(labelled_form(template,values,intent,split=split,
                    template_id=f'{split}:{intent}',structure_id=f'{"reordered-clauses" if split=="structure" else split}:{intent}')))
    surfaces={}
    for split,rows in data.items():
        normalized={' '.join(tokenize(row['text'])) for row in rows}
        if any(normalized&other for other in surfaces.values()): raise ValueError('Surface split overlap')
        surfaces[split]=normalized
    return data


def training_vocabulary(dataset, max_length=128):
    return FrozenVocabulary.from_training_forms([row['text'] for row in dataset['train']],max_length=max_length)


def batch_examples(rows, vocabulary, *, device='cpu'):
    """Encode text and separate protected intent/span labels (BOS = absent slot)."""
    import torch
    if not rows or len(rows)>128: raise ValueError('Require 1..128 labelled rows')
    ids=[]; masks=[]; intents=[]; starts=[]; ends=[]
    for row in rows:
        if row['intent'] not in INTENTS or set(row['spans'])-set(ARGUMENT_SLOTS): raise ValueError('Unknown protected label')
        encoded=vocabulary.encode(row['text'])
        if encoded['truncated']: raise ValueError('Cannot supervise truncated arguments')
        start=[0]*len(ARGUMENT_SLOTS); end=[0]*len(ARGUMENT_SLOTS)
        length=sum(encoded['mask'])
        for slot,bounds in row['spans'].items():
            if len(bounds)!=2 or any(type(v)is not int for v in bounds) or not 1<=bounds[0]<=bounds[1]<length:
                raise ValueError('Protected span outside observed token sequence')
            j=ARGUMENT_SLOTS.index(slot); start[j],end[j]=bounds
        ids.append(encoded['ids']); masks.append(encoded['mask']); intents.append(INTENTS.index(row['intent'])); starts.append(start); ends.append(end)
    return {'inputs':{'token_ids':torch.tensor(ids,dtype=torch.long,device=device),'mask':torch.tensor(masks,dtype=torch.bool,device=device)},
            'labels':{'intent':torch.tensor(intents,dtype=torch.long,device=device),
                      'start':torch.tensor(starts,dtype=torch.long,device=device),'end':torch.tensor(ends,dtype=torch.long,device=device)}}


def language_loss(outputs, labels):
    """Actual multiclass intent and argument-pointer cross entropy."""
    import torch.nn.functional as F
    intent=F.cross_entropy(outputs['intent_logits'],labels['intent'])
    start=F.cross_entropy(outputs['argument_start_logits'].flatten(0,1),labels['start'].flatten())
    end=F.cross_entropy(outputs['argument_end_logits'].flatten(0,1),labels['end'].flatten())
    return {'loss':intent+(start+end)/2,'intent_loss':intent,'span_start_loss':start,'span_end_loss':end}


def validate_finite_gradients(model):
    import torch
    gradients=[parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    if not gradients or any(gradient is None or not torch.isfinite(gradient).all() for gradient in gradients):
        raise ValueError('Missing or nonfinite language gradients')


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def prepare_language_package(root,directory):
    from .learning import source_locks
    output=Path(directory); output.mkdir(parents=True,exist_ok=False)
    datasets=build_language_dataset(); vocabulary=training_vocabulary(datasets)
    manifest={'version':'r56-language-intent-spans-v1','stage':'language-intent-spans','source_locks':source_locks(root),
              'data_hash':_digest(datasets),'vocabulary':vocabulary.manifest(), 'seeds':[17], 'budgets':[len(datasets['train'])],
              'optimizer':'Adam','learning_rate':.001,'optimizer_steps':2000,'embedding_size':32,'hidden_size':48,
              'deterministic_algorithms':True, 'split_hashes':{name:_digest(rows) for name,rows in datasets.items()},
              'evaluation':{'splits':['paraphrase','structure','vocabulary'],
                  'metrics':['intent-correct-count','slot-exact-count','present-slot-exact-count','all-spans-exact-examples'],
                  'selection':'fixed final checkpoint; held-out labels never select weights, vocabulary or stopping',
                  'scientific_scope':'synthetic-template intent and argument extraction only; no semantic or open-language qualification'},
              'status':STATUS,'authorization':'UNISSUED','qualification':'UNISSUED'}
    manifest['freeze_hash']=_digest(manifest)
    for filename,value in (('datasets.json',datasets),('manifest.json',manifest)):
        (output/filename).write_text(json.dumps(value,sort_keys=True,indent=2))
    for role in ('authorization','qualification'):
        template={'role':role,'freeze_hash':manifest['freeze_hash'],'issuer':'','issued':False,
                  'scope':'native-r56-training','signature_hex':''}
        (output/f'{role}.UNISSUED.json').write_text(json.dumps(template,sort_keys=True,indent=2))
    return manifest


def train_language(root,package,authorization,qualification,trusted_keys,output):
    """Future authorized fitting path. Never called by preparation/engineering tests."""
    from .learning import validate_gate,save_checkpoint
    package=Path(package); manifest=json.loads((package/'manifest.json').read_text()); datasets=json.loads((package/'datasets.json').read_text())
    freeze_hash=validate_gate(root,manifest,datasets,authorization,qualification,trusted_keys)
    required={'src/mindscape/r56/language_model.py','src/mindscape/r56/language_learning.py'}
    if manifest.get('stage')!='language-intent-spans' or not required.issubset(manifest['source_locks']):
        raise PermissionError('Freeze does not bind language model, supervision and protocol')
    vocabulary=training_vocabulary(datasets)
    if vocabulary.manifest()!=manifest.get('vocabulary'): raise PermissionError('Train-only vocabulary freeze mismatch')
    steps=manifest.get('optimizer_steps'); rate=manifest.get('learning_rate')
    if type(steps)is not int or not 1<=steps<=50000 or type(rate)not in (int,float) or not 0<rate<=.1:
        raise PermissionError('Invalid frozen optimization budget')
    if manifest.get('deterministic_algorithms') is not True or manifest.get('split_hashes')!={name:_digest(rows) for name,rows in datasets.items()}:
        raise PermissionError('Determinism or evaluation split freeze mismatch')
    destination=Path(output)
    if destination.exists() or Path(str(destination)+'.failure.json').exists(): raise FileExistsError('Refuse to overwrite language checkpoint')
    # All neural construction and optimizer operations are strictly below the gate.
    completed_steps=0
    try:
        import torch
        torch.manual_seed(manifest['seeds'][0])
        torch.use_deterministic_algorithms(True)
        model=build_language_encoder(vocabulary,manifest['embedding_size'],manifest['hidden_size'])
        model.train()
        optimizer=torch.optim.Adam(model.parameters(),lr=rate)
        batch=batch_examples(datasets['train'],vocabulary)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            losses=language_loss(model(**batch['inputs']),batch['labels'])
            if any(not torch.isfinite(value).all() for value in losses.values()):
                raise ValueError('Nonfinite language loss')
            losses['loss'].backward()
            validate_finite_gradients(model)
            optimizer.step()
            completed_steps+=1
            if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise ValueError('Nonfinite language parameters after update')
        metadata={'trained':True,'optimizer_steps':completed_steps,'freeze_hash':freeze_hash,'stage':'language-intent-spans',
                  'vocabulary':vocabulary.manifest(),'embedding_size':manifest['embedding_size'],'hidden_size':manifest['hidden_size'],
                  'deterministic_algorithms':True,'split_hashes':manifest['split_hashes'],'evaluation':manifest['evaluation'],
                  'qualified':False,'status':'TRAINED BUT SCIENTIFICALLY UNQUALIFIED'}
        return save_checkpoint(model,destination,metadata)
    except Exception as exc:
        failure={'freeze_hash':freeze_hash,'stage':'language-intent-spans','completed_steps':completed_steps,
                 'error_type':type(exc).__name__,'status':'FAILED; NO SUCCESS OR QUALIFICATION CLAIM'}
        Path(str(destination)+'.failure.json').write_text(json.dumps(failure,sort_keys=True,indent=2))
        raise


def replay_language_checkpoint(path,expected_hash,rows):
    """Read-only replay: restore exact train-frozen vocabulary and safe JSON tensors."""
    from .learning import load_checkpoint
    path=Path(path)
    if path.stat().st_size>64_000_000: raise ValueError('Checkpoint exceeds bound')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_hash: raise ValueError('Checkpoint hash mismatch')
    metadata=json.loads(raw)['metadata']
    if metadata.get('stage')!='language-intent-spans': raise ValueError('Wrong checkpoint stage')
    v=metadata['vocabulary']
    if not isinstance(v.get('tokens'),list) or not 3<=len(v['tokens'])<=20003 or v['tokens'][:3]!=['<pad>','<unk>','<bos>'] or not 8<=v.get('max_length',0)<=512:
        raise ValueError('Invalid bounded checkpoint vocabulary')
    if any(not isinstance(token,str) or len(token)>16000 for token in v['tokens']) or len(set(v['tokens']))!=len(v['tokens']):
        raise ValueError('Invalid checkpoint vocabulary tokens')
    vocabulary=FrozenVocabulary(tuple(v['tokens']),v['training_sha256'],v['max_length'])
    if vocabulary.manifest()!=v: raise ValueError('Checkpoint vocabulary manifest differs')
    model=build_language_encoder(vocabulary,metadata['embedding_size'],metadata['hidden_size'])
    load_checkpoint(model,path,expected_hash)
    import torch
    with torch.no_grad():
        batch=batch_examples(rows,vocabulary); outputs=model(**batch['inputs']); losses=language_loss(outputs,batch['labels'])
        if any(not torch.isfinite(value).all() for value in losses.values()): raise ValueError('Nonfinite read-only replay loss')
    intent_predictions=outputs['intent_logits'].argmax(-1)
    start_predictions=outputs['argument_start_logits'].argmax(-1)
    end_predictions=outputs['argument_end_logits'].argmax(-1)
    slot_exact=(start_predictions==batch['labels']['start']) & (end_predictions==batch['labels']['end'])
    return {'intent_predictions':intent_predictions.tolist(),
            'metrics':{'examples':len(rows),'intent_correct':int((intent_predictions==batch['labels']['intent']).sum()),
                       'slot_exact_count':int(slot_exact.sum()),'slot_total':slot_exact.numel(),
                       'present_slot_exact_count':int((slot_exact & (batch['labels']['start']!=0)).sum()),
                       'present_slot_total':int((batch['labels']['start']!=0).sum()),
                       'all_spans_exact_examples':int(slot_exact.all(-1).sum())},
            'span_start_predictions':outputs['argument_start_logits'].argmax(-1).tolist(),
            'span_end_predictions':outputs['argument_end_logits'].argmax(-1).tolist(),
            'split_sha256':_digest(rows),'checkpoint_sha256':expected_hash,
            'loss':float(losses['loss']),'qualified':False,'status':'READ-ONLY REPLAY; NOT A QUALIFICATION CLAIM'}
