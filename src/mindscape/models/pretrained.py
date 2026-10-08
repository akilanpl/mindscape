"""Optional offline pretrained encoder backend; not a required experiment dependency."""
class LocalHFEncoder:
    def __init__(self, model_path, allow_download=False):
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install optional pretrained dependencies; no implicit download") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=not allow_download)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=not allow_download).eval()
        self.identifier = str(model_path)
        self.parameter_count = sum(p.numel() for p in self.model.parameters())

    def encode(self, texts):
        import torch
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            hidden = self.model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1)
        return pooled.cpu().numpy()


class FrozenPretrainedBackend:
    """Optional pretrained text features plus trainable task head, with offline loading default."""
    def __init__(self, model_path, groups, hidden=64, seed=0, allow_download=False):
        import numpy as np
        from mindscape.models.numpy_backend import NumpyMLP
        self.encoder = LocalHFEncoder(model_path, allow_download)
        self.head = NumpyMLP(groups, hidden, seed)
        self.model_path, self.groups, self.hidden, self.seed = str(model_path), groups, hidden, seed
        self.projection = None
        self.identifier = 'frozen_local_hf_' + str(model_path)

    @property
    def parameter_count(self): return self.encoder.parameter_count + self.head.parameter_count

    def features(self, texts):
        import numpy as np
        from mindscape.models.encoding import INPUT_SIZE
        embeddings = self.encoder.encode(texts)
        if self.projection is None:
            self.projection = np.random.default_rng(self.seed).normal(0,1/(embeddings.shape[1]**.5),
                (embeddings.shape[1],INPUT_SIZE))
        return embeddings @ self.projection

    def logits(self, texts): return self.head.logits(self.features(texts))
    def generate(self, texts): return self.head.generate(self.features(texts))

    def save(self, path):
        import json
        import numpy as np
        from pathlib import Path
        path=Path(path);path.mkdir(parents=True,exist_ok=False)
        self.head.save(path/'head')
        if self.projection is not None:np.save(path/'projection.npy',self.projection,allow_pickle=False)
        (path/'pretrained.json').write_text(json.dumps({'model_path':self.model_path,'groups':self.groups,
            'hidden':self.hidden,'seed':self.seed,'download_default':False},indent=2))

    @classmethod
    def load(cls,path):
        import json
        import numpy as np
        from pathlib import Path
        from mindscape.models.numpy_backend import NumpyMLP
        path=Path(path);record=json.loads((path/'pretrained.json').read_text())
        backend=cls(record['model_path'],record['groups'],record['hidden'],record['seed'],False)
        backend.head=NumpyMLP.load(path/'head')
        if (path/'projection.npy').exists():backend.projection=np.load(path/'projection.npy',allow_pickle=False)
        return backend
