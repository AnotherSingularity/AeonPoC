"""Smoke: a brief training run completes without error (CPU, synthetic)."""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from aeon.config import AeonConfig
from aeon.model import AeonForCausalLM
from aeon.recursion import audit_certificates


def _is_recursion(name):
    return ("recursion." in name or name.endswith(".U.weight")
            or name.endswith(".D_proj.weight") or name.endswith(".recursion_gate"))


def test_short_training_run_completes():
    torch.manual_seed(0)
    cfg = AeonConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                     num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                     max_position_embeddings=128, h_rec=16, tie_word_embeddings=False)
    cfg.recursion_chunk_size = 1
    model = AeonForCausalLM(cfg).train()
    for n, p in model.named_parameters():
        p.requires_grad_(_is_recursion(n))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    losses = []
    for _ in range(15):
        ids = torch.randint(0, cfg.vocab_size, (1, 16))
        model.reset_recursion_state(1)
        loss = model(input_ids=ids, labels=ids.clone()).loss
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        assert torch.isfinite(loss)

    mean_gate = sum(abs(b.recursion_gate.item()) for b in model.model.layers) / cfg.num_hidden_layers
    assert mean_gate > 0                                  # gates lifted off zero
    assert audit_certificates(model.model.recursion)["chart_A_holds"]
