"""Regression: the contraction certificate holds across training steps."""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from aeon.config import AeonConfig
from aeon.model import AeonForCausalLM
from aeon.recursion import audit_certificates


def _model():
    cfg = AeonConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                     num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                     max_position_embeddings=32, h_rec=8, tie_word_embeddings=False)
    cfg._attn_implementation = "eager"
    cfg.recursion_chunk_size = 1   # train regime
    return cfg, AeonForCausalLM(cfg)


def test_certificate_holds_across_30_steps():
    torch.manual_seed(0)
    cfg, model = _model()
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-3)
    for step in range(30):
        ids = torch.randint(0, cfg.vocab_size, (1, 8))
        model.reset_recursion_state(1)
        out = model(input_ids=ids, labels=ids.clone())
        opt.zero_grad(); out.loss.backward(); opt.step()
        if step % 5 == 0:
            a = audit_certificates(model.model.recursion)
            assert a["chart_A_holds"], f"sigma bound broken at step {step}"
            assert a["chart_C_lyapunov_holds"], f"Lyapunov broken at step {step}"
    # final
    a = audit_certificates(model.model.recursion)
    assert a["chart_A_holds"] and a["chart_C_lyapunov_holds"]
    assert a["chart_A_sigma_Wh"] < cfg.margin_h
    assert a["chart_A_sigma_Wc"] < cfg.margin_c
