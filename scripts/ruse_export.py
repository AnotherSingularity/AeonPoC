"""
scripts/ruse_export.py — turn match transcripts into Stage 2 training rows.

Reads one or more transcripts written by scripts/play_ruse.py --out, and
writes a .jsonl with one {"text": ...} row per (faction, match): the system
prompt, then every turn's war table as a user message and the faction's
accepted orders as the assistant reply. One match is one long multi-turn
sequence, which is exactly the length curriculum the Stage 2 notes ask for:
the recursion state only earns its keep when the right move on turn 15
depends on what the table showed on turn 3.

Usage:
    python scripts/ruse_export.py --tokenizer ./aeon_stage1 --out ruse.jsonl match1.json match2.json
    python scripts/ruse_export.py --out ruse.jsonl --winner-only matches/*.json

Also usable without transcripts: --selfplay N plays N heuristic-vs-heuristic
matches in-process first.
"""
import os, sys, argparse, json, glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aeon.ruse import Match, RuseConfig, default_board, HeuristicPlayer
from aeon.ruse.players import SYSTEM_PROMPT


def rows_from_records(records, factions, winner=None, winner_only=False):
    for f in factions:
        if winner_only and winner != f:
            continue
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for r in records:
            obs = r["observation_text"].get(f)
            if obs is None:
                continue
            orders = "\n".join(r["accepted"].get(f, [])) or "PASS"
            msgs.append({"role": "user", "content": obs})
            msgs.append({"role": "assistant", "content": orders})
        if len(msgs) > 1:
            yield f, msgs


def format_messages(msgs, tok):
    if tok is not None and getattr(tok, "chat_template", None):
        try:
            return tok.apply_chat_template(msgs, tokenize=False)
        except Exception:
            pass
    return "\n".join(f"{m['role']}: {m['content']}" for m in msgs)


def selfplay(n, seed, turns):
    out = []
    for i in range(n):
        m = Match(default_board(), seed=seed + i, config=RuseConfig(max_turns=turns))
        res = m.run({"CAPITAL": HeuristicPlayer("CAPITAL", seed=seed + i),
                     "INDUSTRIAL": HeuristicPlayer("INDUSTRIAL", seed=seed + 1000 + i)})
        out.append({
            "winner": res.winner,
            "records": [{"observation_text": r.observation_text, "accepted": r.accepted}
                        for r in res.records],
            "factions": list(m.board.factions),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("transcripts", nargs="*", help="JSON files from play_ruse.py --out")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--out", default="./ruse.jsonl")
    ap.add_argument("--winner-only", action="store_true")
    ap.add_argument("--selfplay", type=int, default=0)
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args, _ = ap.parse_known_args()

    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)

    matches = []
    for pattern in args.transcripts:
        for path in sorted(glob.glob(pattern)) or [path for path in [pattern]]:
            with open(path) as f:
                d = json.load(f)
            d.setdefault("factions", list(d["final_scores"]))
            matches.append(d)
    if args.selfplay:
        matches += selfplay(args.selfplay, args.seed, args.turns)

    n = 0
    with open(args.out, "w") as f:
        for d in matches:
            for faction, msgs in rows_from_records(d["records"], d["factions"],
                                                   d.get("winner"), args.winner_only):
                f.write(json.dumps({"text": format_messages(msgs, tok),
                                    "faction": faction}) + "\n")
                n += 1
    print(f"wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
