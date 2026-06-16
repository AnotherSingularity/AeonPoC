"""tests/test_block_shapes.py — forward-pass shapes and gradient flow.

Builds a tiny Aeon model (no download) and checks:
  - forward produces correctly-shaped logits
  - with recursion enabled, gradients reach the recursion path
    (recursion.*, per-block U / D_proj / gamma)
"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aeon.config import AeonConfig
from aeon.model import AeonR1ForCausalLM


def _tiny_model():
    cfg = AeonConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        h_rec=16,
        tie_word_embeddings=True,
    )
    cfg._attn_implementation = "eager"
    return cfg, AeonR1ForCausalLM(cfg)


def test_forward_shapes():
    cfg, model = _tiny_model()
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 10))
    model.reset_recursion_state(batch_size=2)
    out = model(input_ids=ids)
    assert out.logits.shape == (2, 10, cfg.vocab_size)


def test_recursion_gradients_flow():
    cfg, model = _tiny_model()
    model.train()
    # lift the gate off zero so the recursion read contributes to the loss
    with torch.no_grad():
        for blk in model.model.layers:
            blk.recursion_gate.fill_(0.1)

    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    labels = ids.clone()
    model.reset_recursion_state(batch_size=2)
    out = model(input_ids=ids, labels=labels)
    out.loss.backward()

    rec = model.model.recursion
    assert rec.A_h.grad is not None and rec.A_h.grad.abs().sum() > 0
    assert rec.W_x.weight.grad is not None and rec.W_x.weight.grad.abs().sum() > 0

    blk0 = model.model.layers[0]
    assert blk0.U.weight.grad is not None and blk0.U.weight.grad.abs().sum() > 0
    assert blk0.D_proj.weight.grad is not None and blk0.D_proj.weight.grad.abs().sum() > 0
    assert blk0.recursion_gate.grad is not None and blk0.recursion_gate.grad.abs().sum() > 0


def test_state_persists_across_calls():
    cfg, model = _tiny_model()
    model.eval()
    model.reset_recursion_state(batch_size=1)
    ids = torch.randint(0, cfg.vocab_size, (1, 5))
    with torch.no_grad():
        model(input_ids=ids)
    r1, c1 = model.model.get_recursion_state()
    assert r1 is not None and r1.abs().sum() > 0  # state advanced off zero
