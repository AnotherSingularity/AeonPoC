"""
scripts/play_ruse.py — run a match on the power board.

Usage:
    # heuristic vs random, print the war table every turn
    python scripts/play_ruse.py --p1 heuristic --p2 random --turns 12 -v

    # Aeon (from a checkpoint) as the challenger against the scripted incumbent
    python scripts/play_ruse.py --p1 heuristic --p2 aeon --ckpt ./aeon_stage1 --seat2 INDUSTRIAL

    # you at the table (type orders, one per line, blank line to submit)
    python scripts/play_ruse.py --p1 human --p2 heuristic

    # save the full transcript (for scripts/ruse_export.py)
    python scripts/play_ruse.py --p1 heuristic --p2 heuristic --out match.json

Factions: CAPITAL (incumbent hub), INDUSTRIAL, SECURITY, CONTINENTAL, MOVEMENT.
"""
import os, sys, argparse, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aeon.ruse import Match, RuseConfig, default_board, make_player, ORDER_HELP
from aeon.ruse.board import DEFAULT_FACTIONS


class HumanPlayer:
    def __init__(self, faction):
        self.faction = faction

    def act(self, obs, text):
        print(text)
        print(ORDER_HELP)
        print(f"[{self.faction}] orders (blank line to submit):")
        lines = []
        while True:
            try:
                line = input("> ")
            except EOFError:
                break
            if not line.strip():
                break
            lines.append(line)
        return "\n".join(lines) or "PASS"


def build_player(spec, faction, seed, args):
    if spec == "human":
        return HumanPlayer(faction)
    kw = {}
    if spec == "aeon":
        kw = dict(ckpt=args.ckpt, max_new_tokens=args.max_new_tokens,
                  temperature=args.temperature)
    return make_player(spec, faction, seed=seed, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1", default="heuristic", help="random|heuristic|aeon|human")
    ap.add_argument("--p2", default="random")
    ap.add_argument("--seat1", default="CAPITAL", choices=sorted(DEFAULT_FACTIONS))
    ap.add_argument("--seat2", default="INDUSTRIAL", choices=sorted(DEFAULT_FACTIONS))
    ap.add_argument("--ckpt", default=None, help="Aeon checkpoint for aeon players")
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the transcript as JSON")
    ap.add_argument("-v", "--verbose", action="store_true")
    args, _ = ap.parse_known_args()
    if args.seat1 == args.seat2:
        ap.error("seat1 and seat2 must differ")

    board = default_board([args.seat1, args.seat2])
    match = Match(board, seed=args.seed, config=RuseConfig(max_turns=args.turns))
    players = {
        args.seat1: build_player(args.p1, args.seat1, args.seed, args),
        args.seat2: build_player(args.p2, args.seat2, args.seed + 1, args),
    }
    result = match.run(players, verbose=args.verbose)
    print(result.summary())
    for f, p in players.items():
        if hasattr(p, "recursion_norms"):
            print(f"{f} recursion state norms: {p.recursion_norms()}")
        if getattr(p, "parse_errors", None):
            print(f"{f} parse errors: {len(p.parse_errors)} (first: {p.parse_errors[0]})")

    if args.out:
        payload = {
            "seed": args.seed, "seats": {args.seat1: args.p1, args.seat2: args.p2},
            "winner": result.winner, "reason": result.reason, "turns": result.turns,
            "final_scores": result.final_scores, "score_history": result.score_history,
            "epistemic": result.epistemic,
            "records": [{
                "turn": r.turn, "observation_text": r.observation_text,
                "orders_text": r.orders_text, "accepted": r.accepted,
                "rejected": r.rejected, "scores": r.scores, "lambda_max": r.lambda_max,
            } for r in result.records],
        }
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"wrote transcript to {args.out}")


if __name__ == "__main__":
    main()
