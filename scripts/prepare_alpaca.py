"""
scripts/prepare_alpaca.py — build the Stage 1 training file from tatsu-lab/alpaca.

Writes a .jsonl with one {"text": ...} row per example, formatted with the
checkpoint tokenizer's chat template when available (falls back to the classic
Alpaca prompt format otherwise).

Usage:
    python scripts/prepare_alpaca.py --tokenizer ./aeon_init --out ./alpaca.jsonl

Requires the `datasets` package and network access to HuggingFace.
"""
import os, sys, argparse, json

ALPACA_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
)
ALPACA_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n{output}"
)


def format_row(ex, tok):
    instruction = ex.get("instruction", "").strip()
    inp = ex.get("input", "").strip()
    output = ex.get("output", "").strip()
    if tok is not None and getattr(tok, "chat_template", None):
        user = instruction if not inp else f"{instruction}\n\n{inp}"
        msgs = [{"role": "user", "content": user},
                {"role": "assistant", "content": output}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False)
        except Exception:
            pass
    tmpl = ALPACA_WITH_INPUT if inp else ALPACA_NO_INPUT
    return tmpl.format(instruction=instruction, input=inp, output=output)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tatsu-lab/alpaca")
    ap.add_argument("--split", default="train")
    ap.add_argument("--tokenizer", default=None,
                    help="optional tokenizer path/name for chat-template formatting")
    ap.add_argument("--out", default="./alpaca.jsonl")
    ap.add_argument("--limit", type=int, default=0,
                    help="if >0, only write this many rows")
    args, _ = ap.parse_known_args()

    from datasets import load_dataset
    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"loading {args.dataset}:{args.split} ...")
    ds = load_dataset(args.dataset, split=args.split)

    n = 0
    with open(args.out, "w") as f:
        for ex in ds:
            text = format_row(ex, tok)
            f.write(json.dumps({"text": text}) + "\n")
            n += 1
            if args.limit and n >= args.limit:
                break
    print(f"wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
