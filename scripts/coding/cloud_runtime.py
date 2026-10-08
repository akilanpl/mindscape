"""Execution-only Cloud backend adapter; frozen policy/source files stay unchanged."""
import os
import platform
import resource
from pathlib import Path

import torch

from mindscape.coding import device as device_module
from mindscape.coding.batching import ResidentCoder
from mindscape.coding.device import ModelDevice
from mindscape.coding.model import LocalCoder


class CloudDevice(ModelDevice):
    def synchronize(self):
        if self.name=='cuda':
            torch.cuda.synchronize(self.device)


def backend():
    name='cuda' if torch.cuda.is_available() else 'cpu'
    precision='float16' if name=='cuda' else 'float32'
    allocation=torch.ones(1,device=name)
    assert allocation.device.type==name
    return name,precision


class CloudCoder(ResidentCoder):
    def __init__(self,path,name,precision):
        def select(request=None,dtype=None,require_mps=False):
            # This explicit execution adapter never rewrites frozen device.py.
            return CloudDevice(torch.device(name),getattr(torch,precision),name,precision)
        self._original_selector=device_module.select_device
        device_module.select_device=select
        LocalCoder.__init__(self,path,cache_enabled=False,device=name,precision=precision)
        torch.set_num_threads(min(os.cpu_count() or 1,16))
        self.base_revision=self.revision
        self.loaded_adapters={}
        self.batch_records=[]
        self.deadline=None
        self.tokenizer.padding_side='left'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token=self.tokenizer.eos_token
    def generate_batch(self,requests):
        result=super().generate_batch(requests)
        for _,timing in result:
            timing['device']=self.execution.name
        self.batch_records[-1]['device']=self.execution.name
        return result


def resources():
    record={'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1024 if platform.system()=='Linux' else 1),
            'host':platform.node(),'cpu_count':os.cpu_count(),'python':platform.python_version(),
            'torch':torch.__version__,'cuda_available':torch.cuda.is_available()}
    if torch.cuda.is_available():
        record.update(cuda_allocated_bytes=torch.cuda.memory_allocated(),
                      cuda_reserved_bytes=torch.cuda.memory_reserved(),cuda_device=torch.cuda.get_device_name())
    return record


def validate_frozen():
    import hashlib
    import json
    p=Path('results/coding/emergency_mps_v1/locked_protocol.json')
    protocol=json.loads(p.read_text())
    for name,digest in protocol['source_hashes'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Frozen scientific source changed: '+name)
    preservation=json.loads(Path('experiments/coding_completion_v2/locked_preservation.json').read_text())
    for name,digest in {**preservation['checkpoint_sha256'],**preservation['scientific_data_sha256']}.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Frozen data/checkpoint changed: '+name)
    return protocol
