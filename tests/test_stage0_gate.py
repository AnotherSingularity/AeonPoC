"""tests/test_stage0_gate.py — the byte-identity gate.

The full Stage 0 gate (scripts/verify_stage0.py) needs the R1-Distill download.
This test verifies the identical wiring property offline by porting a small
random Qwen2 into Aeon and checking gamma=0 -> byte-identical logits (fp32).
"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from aeon.config import AeonConfig
from aeon.model import AeonR1ForCausalLM
from scripts.from_r1 import port_weights
from scripts.verify_wiring import build_small_qwen


def _check(tie):
    cfg, vanilla = build_small_qwen(seed=0, tie=tie)
    acfg = AeonConfig(**cfg.to_dict())
    acfg.h_rec = 32
    acfg._attn_implementation = "eager"
    aeon = AeonR1ForCausalLM(acfg).eval()
    port_weights(vanilla, aeon)
    aeon.disable_recursion()

    for blk in aeon.model.layers:
        assert blk.recursion_gate.item() == 0.0

    worst = 0.0
    torch.manual_seed(123)
    with torch.no_grad():
        for _ in range(4):
            ids = torch.randint(0, cfg.vocab_size, (1, 16))
            aeon.reset_recursion_state(batch_size=1)
            lv = vanilla(ids).logits[0, -1].float()
            la = aeon(ids).logits[0, -1].float()
            worst = max(worst, (lv - la).abs().max().item())
    return worst


def test_stage0_byte_identity_tied():
    worst = _check(tie=True)
    assert worst < 1e-4, f"gamma=0 forward diverges from vanilla Qwen2: {worst:.3e}"


def test_stage0_byte_identity_untied():
    worst = _check(tie=False)
    assert worst < 1e-4, f"gamma=0 forward diverges from vanilla Qwen2: {worst:.3e}"
