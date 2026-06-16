"""STAGE 0 GATE: Aeon at gamma=0 should produce R1-equivalent predictions."""
import os, sys, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aeon.model import AeonR1ForCausalLM


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
    ap.add_argument("--tol", type=float, default=5e-1)
    args, _ = ap.parse_known_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"loading Aeon from {args.aeon} ...")
    aeon = AeonR1ForCausalLM.from_pretrained(
        args.aeon, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()
    aeon.disable_recursion()
    tok = AutoTokenizer.from_pretrained(args.aeon)

    print(f"loading R1 from {args.r1} ...")
    r1 = AutoModelForCausalLM.from_pretrained(
        args.r1, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()

    max_diffs = []
    all_match = True
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").to(device)
        aeon.reset_recursion_state(batch_size=ids.input_ids.shape[0])
        lh = aeon(**ids).logits[0, -1].float()
        lr = r1(**ids).logits[0, -1].float()
        diff = (lh - lr).abs().max().item()
        match = bool(lh.argmax() == lr.argmax())
        max_diffs.append(diff)
        if not match:
            all_match = False
        print(f"  '{p[:60]:60}' max|dlogit|={diff:.4f}  argmax_match={match}")

    worst = max(max_diffs)
    print(f"\nworst max|dlogit|: {worst:.4f}   tol: {args.tol}")
    print(f"argmax matches on all prompts: {all_match}")
    if worst < args.tol and all_match:
        print("STAGE 0 PASSED.")
        sys.exit(0)
    else:
        print("STAGE 0 FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
