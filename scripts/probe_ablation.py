"""
scripts/probe_ablation.py — is the recursion path doing anything?

Loads an Aeon checkpoint and, for each probe prompt, generates a reply twice:
  ON  : enable_recursion()  + fresh recursion state
  OFF : disable_recursion() + fresh recursion state  (runs as plain backbone)

Greedy decode (do_sample=False), so any ON/OFF difference is the recursion
path, not sampling noise. Reports IDENTICAL vs DIFFERENT per prompt and the
norms of the global state (r, c) after the ON run.

Usage:
    python scripts/probe_ablation.py --ckpt ./aeon_stage1

NOTE: this is the first thing in the pipeline that *reloads* trained per-block
gates. If mean|recursion_gate| prints ~0 here on a pre-rename checkpoint, the
gates did not survive the save/load round-trip (see docs/STAGE1_REPORT.md,
"gate serialization"); run scripts/fix_gate_keys.py on the checkpoint first.
"""
import os, sys, argparse
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aeon.model import AeonR1ForCausalLM
from aeon.recursion import audit_certificates


# A mix of factual recall, creative generation, and in-context memory.
PROMPTS = [
    "Q: What is the capital of France? A:",
    "Q: What is 13 times 12? A:",
    "Write a two-line poem about the sea.",
    "The secret word is 'lighthouse'. Ignore distractions. The secret word is:",
    "Alice put a red key in the drawer. Bob took a blue key. The key in the drawer is:",
]


def gen(model, tok, prompt, device, max_new_tokens, temperature):
    ids = tok(prompt, return_tensors="pt").to(device)
    # Deterministic sampling: reseed before every generate so the ON and OFF
    # runs draw the same random numbers. The recursion path is then the only
    # thing that can make the two outputs differ.
    torch.manual_seed(0)
    out = model.generate(
        **ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        num_beams=1,
        pad_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.7)
    args, _ = ap.parse_known_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"loading {args.ckpt} ...")
    model = AeonR1ForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.ckpt)

    # --- gate / certificate readout (did the trained recursion actually load?) ---
    gates = [blk.recursion_gate.item() for blk in model.model.layers]
    mean_abs_gate = sum(abs(g) for g in gates) / len(gates)
    aud = audit_certificates(model.model.recursion)
    print(f"\nmean|recursion_gate| across {len(gates)} blocks: {mean_abs_gate:.4f}")
    print(f"sigma(Wh)={aud['chart_A_sigma_Wh']:.4f}  sigma(Wc)={aud['chart_A_sigma_Wc']:.4f}  "
          f"certificate_holds={aud['chart_A_holds']}")
    if mean_abs_gate < 1e-6:
        print("WARNING: gates are ~zero. The recursion path is inert regardless of the\n"
              "         enable/disable flag, so ON and OFF WILL be identical. If this\n"
              "         checkpoint was trained to nonzero gates, they did not survive\n"
              "         reload -- see docs/STAGE1_REPORT.md ('gate serialization').")
    print("=" * 78)

    n_diff = 0
    for p in PROMPTS:
        model.enable_recursion()
        model.reset_recursion_state(batch_size=1)
        reply_on = gen(model, tok, p, device, args.max_new_tokens, args.temperature)
        r_on, c_on = model.model.get_recursion_state()
        r_norm = float(r_on.norm()) if r_on is not None else float("nan")
        c_norm = float(c_on.norm()) if c_on is not None else float("nan")

        model.disable_recursion()
        model.reset_recursion_state(batch_size=1)
        reply_off = gen(model, tok, p, device, args.max_new_tokens, args.temperature)

        verdict = "IDENTICAL" if reply_on == reply_off else "DIFFERENT"
        if verdict == "DIFFERENT":
            n_diff += 1

        print(f"\nPROMPT: {p}")
        print(f"  [ON ]  ||r||={r_norm:.3f}  ||c||={c_norm:.3f}")
        print(f"         {reply_on}")
        print(f"  [OFF]  {reply_off}")
        print(f"  -> {verdict}")

    print("=" * 78)
    print(f"{n_diff}/{len(PROMPTS)} prompts changed when the recursion path was enabled.")


if __name__ == "__main__":
    main()
