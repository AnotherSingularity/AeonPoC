"""
scripts/verify_stage0.py — STAGE 0 BYTE-IDENTITY GATE.

Loads Aeon-R1 from `./aeon_init` (produced by from_r1.py) and the matching
vanilla R1-Distill. Runs identical input through both. With gamma=0 everywhere
and recursion_enabled=False, the outputs must match to bfloat16 precision.

If this test fails, the wiring is wrong. Do not proceed to training.
"""
import os, sys, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aeon.model import AeonR1ForCausalLM
from aeon.config import AeonConfig


PROMPTS = [
    "The capital of the country north of Mexico is",
    "Translate to French: 'The sky is blue today.' ->",
    "Q: What is 17 times 23? A: Let me think step by step.",
]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aeon", default="./aeon_init")
    ap.add_argument("--r1", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--tol", type=float, default=5e-3,
                    help="max allowed abs diff in last-token logits (bf16 is noisy)")
    args, _ = ap.parse_known_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"loading Aeon from {args.aeon} ...")
    aeon = AeonR1ForCausalLM.from_pretrained(
        args.aeon, torch_dtype=torch.bfloat16
    ).to(device).eval()
    aeon.disable_recursion()    # belt-and-suspenders: r=0 fed to every block
    tok = AutoTokenizer.from_pretrained(args.aeon)

    print(f"loading R1 from {args.r1} ...")
    r1 = AutoModelForCausalLM.from_pretrained(
        args.r1, torch_dtype=torch.bfloat16
    ).to(device).eval()

    max_diffs = []
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").to(device)
        aeon.reset_recursion_state(batch_size=ids.input_ids.shape[0])

        out_h = aeon(**ids)
        out_r = r1(**ids)
        lh = out_h.logits[0, -1].float()
        lr = out_r.logits[0, -1].float()
        diff = (lh - lr).abs().max().item()
        max_diffs.append(diff)
        argmax_match = bool(lh.argmax() == lr.argmax())
        print(f"  '{p[:60]:60}' max|dlogit|={diff:.4f}  argmax_match={argmax_match}")

    worst = max(max_diffs)
    print(f"\nworst max|dlogit|: {worst:.4f}   tol: {args.tol}")
    if worst < args.tol:
        print("STAGE 0 PASSED. Recursion path is correctly disabled at init.")
        sys.exit(0)
    else:
        print("STAGE 0 FAILED. Find the wiring bug before any training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
