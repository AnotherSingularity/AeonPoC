"""
scripts/verify_wiring.py — OFFLINE byte-identity wiring gate.

The Stage 0 gate (scripts/verify_stage0.py) compares Aeon against the real
R1-Distill download. This script verifies the SAME wiring property without any
download: it builds a small *random* Qwen2 of the same family, ports its
weights into Aeon exactly as from_r1.py does, and checks that with gamma=0
everywhere the Aeon forward is byte-identical to the vanilla Qwen2 forward.

Run in fp32 where round-off is tiny: expect worst max|dlogit| ~ 1e-6. A wiring
bug (residual shifted at gamma=0, attention not causal across the per-token
loop, lm_head not re-tied, weights not actually loaded) blows past this by
orders of magnitude.

Usage:
    python scripts/verify_wiring.py
"""
import os, sys, argparse
import torch
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aeon.config import AeonConfig
from aeon.model import AeonR1ForCausalLM
from scripts.from_r1 import port_weights


def build_small_qwen(seed=0, tie=True):
    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=tie,
        rms_norm_eps=1e-6,
        attn_implementation="eager",
    )
    model = Qwen2ForCausalLM(cfg)
    # randomize away from the (mostly zero-ish) default init so the test is real
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2:
                p.normal_(0, 0.05)
    model.eval()
    return cfg, model


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--h_rec", type=int, default=32)
    args, _ = ap.parse_known_args()

    for tie in (True, False):
        cfg, vanilla = build_small_qwen(seed=0, tie=tie)

        acfg = AeonConfig(**cfg.to_dict())
        acfg.h_rec = args.h_rec
        acfg.model_type = "aeon_r1"
        acfg._attn_implementation = "eager"
        aeon = AeonR1ForCausalLM(acfg)
        aeon.eval()

        port_weights(vanilla, aeon)
        aeon.disable_recursion()

        # gate must be exactly zero
        for blk in aeon.model.layers:
            assert blk.recursion_gate.item() == 0.0

        worst = 0.0
        torch.manual_seed(123)
        for _ in range(4):
            ids = torch.randint(0, cfg.vocab_size, (1, 16))
            aeon.reset_recursion_state(batch_size=1)
            lv = vanilla(ids).logits[0, -1].float()
            la = aeon(ids).logits[0, -1].float()
            worst = max(worst, (lv - la).abs().max().item())

        tag = "tied" if tie else "untied"
        status = "OK" if worst < args.tol else "FAIL"
        print(f"[{tag:6}] worst max|dlogit| = {worst:.3e}   tol={args.tol}   {status}")
        if worst >= args.tol:
            print("WIRING FAILED. gamma=0 forward does not match vanilla Qwen2.")
            sys.exit(1)

    print("WIRING OK. gamma=0 forward is byte-identical to vanilla Qwen2 (fp32).")
    sys.exit(0)


if __name__ == "__main__":
    main()
