"""
scripts/chat.py — multi-turn chat REPL.

Usage:
    python scripts/chat.py --ckpt ./aeon_stage1
"""
import os, sys, argparse
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aeon.model import AeonR1ForCausalLM


SYSTEM_PROMPT = (
    "You are Aeon, a small language model. Respond as Aeon. Be concise and "
    "direct. Do not narrate your reasoning unless asked."
)


def split_response(text: str) -> str:
    """Separate any internal dialog from the final reply.

    Best-effort: R1-Distill-Qwen-1.5B does not reliably emit </think> tags at
    this scale, so we fall back to a paragraph heuristic.
      - if the output has a </think>, the reply is everything after it
      - else if it has a blank line, take everything after the LAST one
        (the final paragraph is usually the actual answer)
      - else return the whole thing
    """
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    if "\n\n" in text:
        return text.rsplit("\n\n", 1)[-1].strip()
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    args, _ = ap.parse_known_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AeonR1ForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=torch.bfloat16
    ).to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.ckpt)

    # Persistent recursion state across the whole chat
    model.reset_recursion_state(batch_size=1)

    print("Aeon chat. Type 'reset' to clear the recursion state, "
          "'quit' to exit.")
    history = []
    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not user: continue
        if user == "quit": break
        if user == "reset":
            model.reset_recursion_state(batch_size=1)
            history = []
            print("[state cleared]")
            continue

        history.append({"role": "user", "content": user})
        # Inject the Aeon system message at the start of every turn, then the
        # running conversation. Use the tokenizer's chat template if available
        # (R1-Distill has one).
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
        try:
            prompt = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = (SYSTEM_PROMPT + "\n"
                      + "\n".join(f"{m['role']}: {m['content']}" for m in history)
                      + "\nassistant:")

        ids = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.temperature > 0,
                pad_token_id=tok.eos_token_id,
            )
        raw = tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)
        reply = split_response(raw)
        print(f"aeon> {reply}")
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
