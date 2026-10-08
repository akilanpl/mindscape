"""Exercise real CPU neural forwards through the Cloud execution adapter."""
import importlib.util
from pathlib import Path

import torch
from transformers import BatchEncoding, Qwen2Config, Qwen2ForCausalLM


def test_cloud_cpu_forward_reports_actual_backend_and_preserves_greedy_batching(monkeypatch):
    spec=importlib.util.spec_from_file_location('cloud_runtime_test',Path('scripts/coding/cloud_runtime.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    import transformers

    from mindscape.coding import device
    monkeypatch.setattr(device,'select_device',device.select_device)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    assert module.backend()==('cpu','float32')
    torch.manual_seed(0)
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=512,
        eos_token_id=2,pad_token_id=2))
    class Tokenizer:
        pad_token_id=2
        eos_token_id=2
        eos_token='end'
        padding_side='left'
        def apply_chat_template(self,messages,**kwargs):
            return messages[-1]['content']
        def __call__(self,text,**kwargs):
            texts=[text] if isinstance(text,str) else text
            values=[[3,4] if v=='first' else [3,4,5] for v in texts]
            width=max(map(len,values))
            if kwargs.get('padding'):
                width=((width+63)//64)*64
            return BatchEncoding({'input_ids':torch.tensor([[2]*(width-len(v))+v for v in values]),
                                  'attention_mask':torch.tensor([[0]*(width-len(v))+[1]*len(v) for v in values])})
        def decode(self,tokens,**kwargs):
            return ' '.join(str(int(v)) for v in list(tokens) if int(v)!=2)
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**k:Tokenizer())
    monkeypatch.setattr(transformers.AutoModelForCausalLM,'from_pretrained',lambda *a,**k:model)
    coder=module.CloudCoder('tiny-local-fixture','cpu','float32')
    requests=[('system','first',2,11),('system','second',2,11)]
    batched=coder.generate_batch(requests)
    for request,(output,timing) in zip(requests,batched,strict=True):
        assert output==coder.generate(*request)
        assert timing['device']=='cpu' and timing['precision']=='float32'
        assert timing['actual_batch_size']==2 and not timing['cache_hit']
    assert all(p.device.type=='cpu' for p in coder.model.parameters())
    assert coder.batch_records[-1]['device']=='cpu'
    assert coder.cache_enabled is False
