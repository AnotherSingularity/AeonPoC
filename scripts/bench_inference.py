"""
scripts/bench_inference.py — decode-throughput benchmark (Layer 3).

Measures tokens/sec for greedy generation at different chunk sizes K. The Layer 3
target is >50 tok/s on a 3090 (from the v1 ~10 tok/s); the win comes from the
batched prefill (and batched training/eval), while decode steps are single-token
(K=1) regardless. Run on the box for the real number; on CPU it gives a relative
signal only.

Usage:
    python scripts/bench_inference.py --ckpt ./aeon_init --prompt_len 64 --new_tokens 64
    python scripts/bench_inference.py --tiny     # no checkpoint: synthetic tiny model
"""
import os, sys, time, argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aeon.config import AeonConfig
from aeon.model import AeonForCausalLM


def _tiny():
    cfg = AeonConfig(vocab_size=512, hidden_size=256, intermediate_size=512,
                     num_hidden_layers=6, num_attention_heads=8, num_key_value_heads=2,
                     max_position_embeddings=512, h_rec=64, tie_word_embeddings=False)
    cfg._attn_implementation = "eager"
    return AeonForCausalLM(cfg).eval(), cfg.vocab_size


@torch.no_grad()
def bench(model, vocab, device, prompt_len, new_tokens, chunk_sizes):
    ids = torch.randint(0, vocab, (1, prompt_len), device=device)
    for K in chunk_sizes:
        model.config.recursion_chunk_size = K
        model.reset_recursion_state(1)
        # warmup
        model.generate(ids, max_new_tokens=4, do_sample=False, num_beams=1)
        if device == "cuda":
            torch.cuda.synchronize()
        model.reset_recursion_state(1)
        t0 = time.time()
        out = model.generate(ids, max_new_tokens=new_tokens, do_sample=False, num_beams=1)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        gen = out.shape[1] - prompt_len
        label = "T(full)" if K == 0 else str(K)
        print(f"  K={label:>7}  prefill={prompt_len}  generated={gen}  "
              f"{dt:.2f}s  {gen/dt:.1f} tok/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--prompt_len", type=int, default=64)
    ap.add_argument("--new_tokens", type=int, default=64)
    ap.add_argument("--chunks", default="0,1",
                    help="comma list of K values (0 = fully batched)")
    args, _ = ap.parse_known_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    chunk_sizes = [int(x) for x in args.chunks.split(",")]

    if args.tiny or not args.ckpt:
        model, vocab = _tiny()
        model.to(device)
        print(f"[tiny synthetic model on {device}]")
    else:
        model = AeonForCausalLM.from_pretrained(
            args.ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to(device).eval()
        vocab = model.config.vocab_size
        print(f"[{args.ckpt} on {device}]")

    bench(model, vocab, device, args.prompt_len, args.new_tokens, chunk_sizes)


if __name__ == "__main__":
    main()
