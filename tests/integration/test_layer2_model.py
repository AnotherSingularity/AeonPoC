"""
Integration tests for the rewritten AeonForCausalLM (Layer 2): forward, generate,
and save/load round-trip on the new PreTrainedModel base (no Qwen2 inheritance).
"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aeon.config import AeonConfig
from aeon.model import AeonForCausalLM, AeonModel
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM


def _tiny_config(tie=False):
    cfg = AeonConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, h_rec=16, tie_word_embeddings=tie,
    )
    cfg._attn_implementation = "eager"
    return cfg


def test_not_a_qwen2_subclass():
    # Layer 2 goal: Aeon owns the modeling code, no Qwen2 inheritance.
    assert not issubclass(AeonForCausalLM, Qwen2ForCausalLM)
    assert not issubclass(AeonModel, Qwen2ForCausalLM)


def test_forward_shapes_and_loss():
    cfg = _tiny_config()
    model = AeonForCausalLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    model.reset_recursion_state(batch_size=2)
    out = model(input_ids=ids, labels=ids.clone())
    assert out.logits.shape == (2, 8, cfg.vocab_size)
    assert out.loss is not None and torch.isfinite(out.loss)


def test_generate_greedy():
    cfg = _tiny_config()
    model = AeonForCausalLM(cfg).eval()
    model.reset_recursion_state(batch_size=1)
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(ids, max_new_tokens=8, do_sample=False, num_beams=1)
    assert out.shape[1] == 4 + 8


def test_recursion_gradients_flow():
    cfg = _tiny_config()
    cfg.recursion_chunk_size = 1   # training regime: within-sequence coupling
    model = AeonForCausalLM(cfg).train()
    with torch.no_grad():
        for blk in model.model.layers:
            blk.recursion_gate.fill_(0.1)
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    model.reset_recursion_state(batch_size=1)
    model(input_ids=ids, labels=ids.clone()).loss.backward()
    rec = model.model.recursion
    assert rec.A_h.grad is not None and rec.A_h.grad.abs().sum() > 0
    blk0 = model.model.layers[0]
    assert blk0.U.weight.grad is not None and blk0.U.weight.grad.abs().sum() > 0
    assert blk0.recursion_gate.grad is not None and blk0.recursion_gate.grad.abs().sum() > 0


def test_save_load_roundtrip(tmp_path):
    cfg = _tiny_config()
    model = AeonForCausalLM(cfg).eval()
    with torch.no_grad():
        for i, blk in enumerate(model.model.layers):
            blk.recursion_gate.fill_(0.02 + 0.001 * i)
    d = str(tmp_path / "ckpt")
    model.save_pretrained(d)

    reloaded = AeonForCausalLM.from_pretrained(d, torch_dtype=torch.float32).eval()
    # recursion gates round-trip (recursion_gate name dodges the HF shim)
    for i, blk in enumerate(reloaded.model.layers):
        assert abs(blk.recursion_gate.item() - (0.02 + 0.001 * i)) < 1e-6
    # a transformer weight round-trips
    a = model.model.layers[0].transformer_layer.self_attn.q_proj.weight
    b = reloaded.model.layers[0].transformer_layer.self_attn.q_proj.weight
    assert torch.allclose(a, b)
    # config model_type
    assert reloaded.config.model_type == "aeon"


def test_disable_recursion_runs_plain_transformer():
    cfg = _tiny_config()
    model = AeonForCausalLM(cfg).eval()
    model.disable_recursion()
    model.reset_recursion_state(batch_size=1)
    ids = torch.randint(0, cfg.vocab_size, (1, 5))
    out = model(input_ids=ids)
    assert out.logits.shape == (1, 5, cfg.vocab_size)
