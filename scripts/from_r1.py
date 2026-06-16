"""
scripts/from_r1.py — port DeepSeek-R1-Distill-Qwen-1.5B weights into AeonR1.

Usage:
    python scripts/from_r1.py --out ./aeon_init/

Downloads R1-Distill-Qwen-1.5B via HuggingFace, instantiates an AeonR1,
copies all attention/MLP/embedding/norm weights, leaves recursion paths
fresh, saves checkpoint.
"""
import argparse, os, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

# Make sure aeon is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aeon.config import AeonConfig
from aeon.model import AeonR1ForCausalLM


def port_weights(r1: Qwen2ForCausalLM, aeon: AeonR1ForCausalLM):
    """Copy R1's parameters into Aeon. Recursion paths untouched."""
    # Embeddings + final norm + lm_head
    aeon.model.embed_tokens.weight.data.copy_(r1.model.embed_tokens.weight.data)
    aeon.model.norm.weight.data.copy_(r1.model.norm.weight.data)
    # lm_head: may be tied. If untied in the source, copy it explicitly.
    if id(r1.lm_head.weight) != id(r1.model.embed_tokens.weight):
        aeon.lm_head.weight.data.copy_(r1.lm_head.weight.data)
    # If the source ties them, AeonR1ForCausalLM.__init__ already re-tied
    # lm_head to the (now R1-loaded) embeddings, so nothing to do.

    # Per-layer: copy Qwen2DecoderLayer state into the wrapped qwen_block
    for l in range(aeon.config.num_hidden_layers):
        src = r1.model.layers[l]
        dst = aeon.model.layers[l].qwen_block
        dst.load_state_dict(src.state_dict(), strict=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--out", default="./aeon_init")
    ap.add_argument("--h_rec", type=int, default=256)
    args, _ = ap.parse_known_args()

    print(f"loading {args.src} ...")
    r1 = AutoModelForCausalLM.from_pretrained(args.src, torch_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(args.src)

    print("building AeonR1 with config from R1 ...")
    hcfg = AeonConfig(**r1.config.to_dict())
    hcfg.h_rec = args.h_rec
    hcfg.model_type = "aeon_r1"
    hcfg._attn_implementation = r1.config._attn_implementation  # match R1's attention kernel
    aeon = AeonR1ForCausalLM(hcfg).to(torch.bfloat16)

    print("porting weights ...")
    port_weights(r1, aeon)

    # Verify the per-block gate is exactly zero everywhere
    for l, blk in enumerate(aeon.model.layers):
        assert blk.recursion_gate.item() == 0.0, f"gate at layer {l} is not zero"

    os.makedirs(args.out, exist_ok=True)
    aeon.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
