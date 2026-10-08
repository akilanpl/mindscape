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
