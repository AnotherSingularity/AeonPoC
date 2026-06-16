"""
examples/quickstart.py — load Aeon and generate.

With a checkpoint:
    python examples/quickstart.py --ckpt <path-or-hub-id> --prompt "Hello"

Without one (no download needed), it builds a tiny randomly-initialized Aeon and
runs generation end-to-end on CPU — enough to confirm the install works:
    python examples/quickstart.py
"""
import os, sys, argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aeon import AeonConfig, AeonForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="checkpoint path or Hub id")
    ap.add_argument("--prompt", default="Hello, world.")
    ap.add_argument("--max_new_tokens", type=int, default=32)
    args, _ = ap.parse_known_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt:
        from transformers import AutoTokenizer
        model = AeonForCausalLM.from_pretrained(args.ckpt).to(device).eval()
        tok = AutoTokenizer.from_pretrained(args.ckpt)
        model.reset_recursion_state(batch_size=1)
        ids = tok(args.prompt, return_tensors="pt").to(device)
        out = model.generate(**ids, max_new_tokens=args.max_new_tokens, do_sample=False)
        print(tok.decode(out[0], skip_special_tokens=True))
    else:
        print("[no --ckpt] building a tiny random Aeon to smoke-test the API ...")
        cfg = AeonConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=4, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=128, h_rec=16)
        model = AeonForCausalLM(cfg).to(device).eval()
        model.reset_recursion_state(batch_size=1)
        ids = torch.randint(0, cfg.vocab_size, (1, 8), device=device)
        out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False)
        print(f"generated token ids: {out[0].tolist()}")
        print("OK — install works. Pass --ckpt <path> to run a real model.")


if __name__ == "__main__":
    main()
