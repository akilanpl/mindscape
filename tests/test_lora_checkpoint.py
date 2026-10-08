"""Actual adapter serialization/loading regression, not a benchmark measurement."""

import copy

import pytest


def test_real_lora_updated_logits_survive_checkpoint_loading(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    torch.set_num_threads(1)
    torch.manual_seed(17)
    config = transformers.Qwen2Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=16,
    )
    base = transformers.Qwen2ForCausalLM(config)
    pristine = copy.deepcopy(base)
    model = peft.get_peft_model(
        base,
        peft.LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0,
            task_type="CAUSAL_LM",
        ),
    )
    ids = torch.tensor([[1, 3, 5, 7]])
    model.eval()
    with torch.no_grad():
        before = model(ids).logits.clone()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.1)
    model.train()
    for _ in range(3):
        optimizer.zero_grad()
        model(ids, labels=ids).loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        expected = model(ids).logits.clone()
    assert not torch.allclose(before, expected, atol=1e-6)
    model.save_pretrained(tmp_path)
    restored = peft.PeftModel.from_pretrained(pristine, tmp_path, local_files_only=True).eval()
    with torch.no_grad():
        actual = restored(ids).logits
    assert torch.allclose(actual, expected, atol=1e-6)
