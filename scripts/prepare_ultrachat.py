"""
scripts/prepare_ultrachat.py — build the Stage 2 training file from UltraChat.

Source: HuggingFaceH4/ultrachat_200k, split `train_sft`. Each example has a
`messages` list ([{role, content}, ...]) of a multi-turn dialogue.

Filtering (see buildbook 2.1):
  - keep conversations with >= --min_user_turns user turns (default 3)
  - drop if total token length < --min_tokens (default 200)
  - drop if total token length > --max_tokens (default 4096)
  - target ~--target rows (default 30000); if more pass, randomly subsample
    (seeded) for reproducibility.

Output JSONL, one row per conversation:
  {"text": "<chat-templated conversation>", "n_turns": <user turns>, "n_tokens": <int>}

n_turns / n_tokens are precomputed so the curriculum scheduler can filter on
them without re-tokenizing.

Usage:
    python scripts/prepare_ultrachat.py --tokenizer /workspace/aeon_init \
        --out /workspace/ultrachat.jsonl
Requires the `datasets` package and HF access.
"""
import os, sys, json, random, argparse


def n_user_turns(messages):
    return sum(1 for m in messages if m.get("role") == "user")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    ap.add_argument("--split", default="train_sft")
    ap.add_argument("--tokenizer", default="/workspace/aeon_init",
                    help="tokenizer path/name for chat templating + token counts")
    ap.add_argument("--out", default="/workspace/ultrachat.jsonl")
    ap.add_argument("--target", type=int, default=30000)
    ap.add_argument("--min_tokens", type=int, default=200)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--min_user_turns", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_scan", type=int, default=0,
                    help="if >0, stop scanning the source after this many examples (debug)")
    args, _ = ap.parse_known_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"loading {args.dataset}:{args.split} ...")
    ds = load_dataset(args.dataset, split=args.split)

    kept = []
    scanned = 0
    for ex in ds:
        scanned += 1
        if args.max_scan and scanned > args.max_scan:
            break
        messages = ex.get("messages")
        if not messages:
            continue
        if n_user_turns(messages) < args.min_user_turns:
            continue
        try:
            text = tok.apply_chat_template(messages, tokenize=False)
        except Exception:
            continue
        n_tokens = len(tok(text, add_special_tokens=False).input_ids)
        if n_tokens < args.min_tokens or n_tokens > args.max_tokens:
            continue
        kept.append({"text": text,
                     "n_turns": n_user_turns(messages),
                     "n_tokens": n_tokens})
        if scanned % 5000 == 0:
            print(f"  scanned {scanned}, kept {len(kept)}")

    print(f"scanned {scanned}, kept {len(kept)} after filtering")

    if len(kept) > args.target:
        random.Random(args.seed).shuffle(kept)
        kept = kept[:args.target]
        print(f"subsampled to {args.target} (seed {args.seed})")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")

    # quick distribution summary
    if kept:
        toks = sorted(r["n_tokens"] for r in kept)
        p = lambda q: toks[min(len(toks) - 1, int(q * len(toks)))]
        print(f"wrote {len(kept)} rows to {args.out}")
        print(f"n_tokens: min={toks[0]} p50={p(0.5)} p90={p(0.9)} max={toks[-1]}")
    else:
        print("WARNING: no rows written — check the dataset/filters.")


if __name__ == "__main__":
    main()
