"""
Layer 3 tests — chunked-batch forward with sequentialized recursion.

Pins the two facts that define the design:
  - at gamma=0 (recursion inert) the chunk size K is invariant: every K gives
    the same logits (the batched transformer is correct).
  - the recurrent path's gradient flows at K=1 (per-token, training regime) but
    NOT at K=T (a single chunk reads only the initial state and never reads the
    advanced state back within the forward). Training must use K<T; K=T is the
    fast inference default.
"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aeon.config import AeonConfig
from aeon.model import AeonForCausalLM


def _model(tie=False):
    cfg = AeonConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        h_rec=16, tie_word_embeddings=tie,
    )
    cfg._attn_implementation = "eager"
    return cfg, AeonForCausalLM(cfg)


def test_k_invariance_at_gamma_zero():
    cfg, model = _model()
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 12))
    ref = None
    for K in (0, 1, 2, 5, 12):   # 0 == K=T (full)
        model.config.recursion_chunk_size = K
        model.disable_recursion()
        model.reset_recursion_state(1)
        with torch.no_grad():
            logits = model(input_ids=ids).logits
        if ref is None:
            ref = logits
        assert (logits - ref).abs().max().item() < 1e-4


def test_default_config_is_fully_batched():
    cfg, _ = _model()
    assert cfg.recursion_chunk_size == 0   # K = T by default


def _grad_sum(K):
    cfg, model = _model()
    model.config.recursion_chunk_size = K
    model.train()
    with torch.no_grad():
        for blk in model.model.layers:
            blk.recursion_gate.fill_(0.1)
    ids = torch.randint(0, cfg.vocab_size, (1, 10))
    model.reset_recursion_state(1)
    model(input_ids=ids, labels=ids.clone()).loss.backward()
    a = model.model.recursion.A_h.grad
    return None if a is None else a.abs().sum().item()


def test_recursion_trains_at_k1_not_at_kT():
    # K=1 (per-token): recursion is in the gradient path.
    g1 = _grad_sum(1)
    assert g1 is not None and g1 > 0
    # K=T (single chunk): the recursion state never feeds back within the
    # forward, so its parameters get no gradient. Documented, asserted property.
    gT = _grad_sum(0)
    assert gT is None or gT == 0.0


def test_k1_and_kT_differ_when_recursion_enabled():
    cfg, model = _model()
    model.eval()
    with torch.no_grad():
        for blk in model.model.layers:
            blk.recursion_gate.fill_(0.2)   # exaggerate so the effect is visible
    ids = torch.randint(0, cfg.vocab_size, (1, 8))

    model.config.recursion_chunk_size = 1
    model.enable_recursion(); model.reset_recursion_state(1)
    with torch.no_grad():
        a = model(input_ids=ids).logits

    model.config.recursion_chunk_size = 0
    model.enable_recursion(); model.reset_recursion_state(1)
    with torch.no_grad():
        b = model(input_ids=ids).logits

    # The read timing differs, so outputs differ when the gates are nonzero.
    assert (a - b).abs().max().item() > 1e-4


def test_certificate_holds_after_k1_steps():
    from aeon.recursion import audit_certificates
    cfg, model = _model()
    model.config.recursion_chunk_size = 1
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    for _ in range(5):
        ids = torch.randint(0, cfg.vocab_size, (1, 8))
        model.reset_recursion_state(1)
        out = model(input_ids=ids, labels=ids.clone())
        opt.zero_grad(); out.loss.backward(); opt.step()
    aud = audit_certificates(model.model.recursion)
    assert aud["chart_A_holds"]
