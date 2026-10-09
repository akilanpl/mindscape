import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

spec = importlib.util.spec_from_file_location('coding_telemetry_v2',Path(__file__).resolve().parents[1]/'scripts/research_v2/coding_telemetry.py')
module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class TinyLM(torch.nn.Module):
    def forward(self,input_ids):
        return SimpleNamespace(logits=torch.arange(12,dtype=torch.float32).reshape(1,2,6))


def test_observer_hooks_actual_base_model_without_extra_forwards_and_restores():
    lm=TinyLM()
    class Coder:
        revision='fixture'
        model=SimpleNamespace(get_base_model=lambda:lm)
        def __init__(self):
            self.timings=[]
        def generate(self,system,prompt,max_tokens,seed):
            lm(input_ids=torch.tensor([[1,2]]))
            self.timings.append({'fixture':True})
            return 'original'
    coder=Coder();original=coder.generate;events=[]
    with module.observe(coder,events):assert coder.generate('s','p',8,0)=='original'
    assert coder.generate==original
    assert [e['kind'] for e in events]==['model_request','actual_token_input','actual_forward_output','model_response']
    assert not lm._forward_hooks and not lm._forward_pre_hooks
    assert events[2]['observed']['top5_ids']==[5,4,3,2,1]
