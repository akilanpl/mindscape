"""Local pretrained Qwen inference. No native Mindscape capability claims or training."""
from __future__ import annotations
import json
import hashlib
import os
from pathlib import Path
import threading
import time

MODEL_ID = 'Qwen/Qwen3-0.6B'
MODEL_REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'

class NeuralBackend:
    def __init__(self, model_path=None, device=None, local_files_only=True):
        self.model_path = str(model_path or os.environ.get('MINDSCAPE_MODEL_PATH') or Path(__file__).resolve().parents[2] / 'models' / 'Qwen3-0.6B')
        self.requested_device = device or os.environ.get('MINDSCAPE_DEVICE', 'auto')
        self.local_files_only = local_files_only
        self.model = None
        self.tokenizer = None
        self.stats = {}
        self._lock = threading.RLock()

    def load(self):
        with self._lock:
            if self.model is not None:
                return self
            started = time.perf_counter()
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            device = self.requested_device
            if device == 'auto':
                device = 'mps' if torch.backends.mps.is_available() else 'cpu'
            if device not in {'mps', 'cpu'}:
                raise ValueError('This local backend supports only mps or cpu')
            # CPU fallback is explicit; float32 avoids unsupported low precision CPU kernels.
            torch.set_num_threads(min(4, os.cpu_count() or 1))
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=self.local_files_only, trust_remote_code=False)
            model = AutoModelForCausalLM.from_pretrained(self.model_path, local_files_only=self.local_files_only, trust_remote_code=False, torch_dtype=torch.float16 if device == 'mps' else torch.float32, attn_implementation='sdpa')
            self.model = model.to(device).eval()
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
            self.device = device
            import psutil
            receipt_path=Path(self.model_path)/'model-receipt.json'
            if not receipt_path.exists():
                receipt_path=Path(__file__).resolve().parents[3]/'outputs'/'delivery'/'neural'/'model-receipt.json'
            receipt_sha=hashlib.sha256(receipt_path.read_bytes()).hexdigest() if receipt_path.exists() else None
            self.stats = {'model_revision':MODEL_REVISION, 'download_receipt_sha256':receipt_sha, 'rss_loaded_bytes':psutil.Process().memory_info().rss, 'model_id': MODEL_ID, 'device': device, 'load_seconds': time.perf_counter()-started, 'parameters': sum(p.numel() for p in self.model.parameters()), 'optimizer_updates': 0, 'model_path': self.model_path}
            return self

    def generate(self, prompt, max_new_tokens=192):
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt)>12000:
            raise ValueError('Question must contain 1–12000 characters')
        if not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= 512:
            raise ValueError('max_new_tokens must be 1–512')
        with self._lock:
            self.load()
            import torch
            start = time.perf_counter()
            messages = [{'role':'system','content':'You are a concise STEM assistant. Follow the requested output format. Do not claim answers are verified unless a tool verified them.'}, {'role':'user','content':prompt}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
            if inputs['input_ids'].shape[-1] > 4096:
                raise ValueError('Input exceeds local context limit')
            with torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, max_time=30.0, pad_token_id=self.tokenizer.eos_token_id)
            ids = output[0, inputs['input_ids'].shape[-1]:]
            raw = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
            self.stats['last_inference'] = {'seconds': time.perf_counter()-start, 'input_tokens': inputs['input_ids'].shape[-1], 'generated_tokens': len(ids), 'truncated': len(ids)>=max_new_tokens or (time.perf_counter()-start)>=30.0}
            return raw

    def structured_parse(self, question):
        """Advisory neural representation; callers must independently validate all values."""
        prompt = (
            'Translate the question into one JSON object for a verified STEM solver. No prose. '
            'Use subject and task. Subjects: math, physics, coding, sql, unsupported. '
            'Math tasks differentiate/integrate/solve_equation use expression and variable, e.g. '
            '{"subject":"math","task":"differentiate","expression":"x**3+2*x","variable":"x"}. '
            'determinant uses matrix: nested numeric lists. matrix uses operation inverse or transpose. '
            'Physics tasks ohm/coulomb/electric_field/lens/mirror/resistance_series/resistance_parallel use values '
            'with explicit SI units; e.g. {"subject":"physics","task":"ohm","unknown":"current",'
            '"values":{"voltage":{"value":12,"unit":"V"},"resistance":{"value":4,"unit":"ohm"}}}. '
            'Coulomb keys q1,q2 in C,r in m; electric_field q in C,r in m; lens/mirror f,u signed in m. '
            'Resistors values.resistors is a list of {value,unit:"ohm"}. '
            'Coding {"subject":"coding","task":"trace","code":"EXACT code"}. '
            'SQL {"subject":"sql","task":"select","query":"EXACT query","tables":{}} only if table data provided. '
            'Copy all numerical values faithfully. Never invent missing values. '
            'For ambiguous or unsupported questions use {"subject":"unsupported","task":"unsupported"}. '
            'Question: '+question)
        raw = self.generate(prompt, max_new_tokens=128)
        result = {'raw':raw, 'valid':False, 'method':'pretrained neural parsing; unverified'}
        try:
            cleaned=raw.strip()
            if cleaned.startswith('```'):
                cleaned=cleaned.split('\n',1)[1].rsplit('```',1)[0].strip()
            value=json.loads(cleaned)
            if isinstance(value,dict) and value.get('subject') in {'math','physics','coding','sql','unsupported'} and isinstance(value.get('task'),str):
                result.update(value, valid=True)
        except (ValueError,IndexError):
            pass
        return result

    def explain_verified(self, question, verified_answer):
        """Unverified explanatory prose, separate from the deterministic verified answer."""
        return self.generate(f'Question: {question}\nA separate verified tool returned: {verified_answer}\nExplain this result briefly. Preserve all supplied numbers and units exactly; do not introduce new calculations.', max_new_tokens=160)
