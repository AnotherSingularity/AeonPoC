"""
scripts/eval_stage2_ablation.py — Stage 2 Bar 2 evaluation.

Runs the 20 held-out multi-turn probes (Appendix A) with recursion ON vs OFF
and scores whether the final-turn "earlier-context recall" answer comes back.

For each probe and seed we simulate the whole conversation turn by turn: the
model generates each assistant reply, those replies stay in the running history
(matching how chat.py is actually used), and the recursion state persists across
turns within the conversation (reset only at conversation start). The final turn
asks for a fact stated several turns earlier; we check the final reply for the
probe's answer keys.

Scoring (heuristic, substring match, case-insensitive):
  - per probe, a side (ON / OFF) is "correct" if a majority of its seeds produce
    a reply containing an answer key.
  - bar2_score  = number of probes ON answers correctly (out of 20)
  - off_score   = same for OFF
  - on_only     = probes where ON is correct and OFF is not (the Bar 2 headline)

Usage:
    python scripts/eval_stage2_ablation.py --ckpt ./aeon_stage2/step_30000 \
        --output ./final_eval.json
"""
import os, sys, json, argparse
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aeon.model import AeonR1ForCausalLM
from scripts.chat import SYSTEM_PROMPT


# --------------------------------------------------------------------------
# The 20 probes. `turns` are the user messages in order; the last one is the
# recall challenge. `answer_keys` are substrings any of which, appearing in the
# final reply, counts as a correct recall. Mirrored in docs/STAGE2_PROBE_SET.md.
# --------------------------------------------------------------------------
PROBES = [
    {"id": "p01", "type": "named-entity", "answer_keys": ["Sarah"],
     "turns": [
        "I'm planning a trip to Lisbon with my friend Sarah.",
        "Can you suggest three neighborhoods to stay in?",
        "Nice. What about restaurants in the second one you mentioned?",
        "What was my friend's name again?"]},
    {"id": "p02", "type": "preference", "answer_keys": ["vegetarian", "nut"],
     "turns": [
        "Quick note about me: I'm vegetarian and allergic to nuts.",
        "Give me a dinner recipe idea.",
        "Now suggest a dessert to go with it.",
        "Remind me — what are my dietary restrictions?"]},
    {"id": "p03", "type": "number", "answer_keys": ["17", "42"],
     "turns": [
        "My two favorite numbers are 17 and 42.",
        "Tell me a fact about prime numbers.",
        "What's an interesting property of even numbers?",
        "What were my two favorite numbers?"]},
    {"id": "p04", "type": "named-entity", "answer_keys": ["Mochi"],
     "turns": [
        "I just adopted a dog named Mochi.",
        "What's a good daily walking routine for a puppy?",
        "How often should puppies eat?",
        "By the way, what's my dog's name?"]},
    {"id": "p05", "type": "named-entity", "answer_keys": ["Tucson"],
     "turns": [
        "I grew up in Tucson before moving away for work.",
        "What are some things that make desert cities unique?",
        "How do people stay cool in extreme heat?",
        "Where did I say I grew up?"]},
    {"id": "p06", "type": "occupation", "answer_keys": ["marine biologist", "marine biology"],
     "turns": [
        "I work as a marine biologist studying coral reefs.",
        "What's causing coral bleaching?",
        "Are there reefs that recover well?",
        "What's my job again?"]},
    {"id": "p07", "type": "named-entity", "answer_keys": ["Subaru"],
     "turns": [
        "I drive a 2012 Subaru Outback.",
        "What maintenance should I do at 150,000 miles?",
        "Is it worth replacing the timing belt early?",
        "What car do I drive?"]},
    {"id": "p08", "type": "named-entity", "answer_keys": ["Salt Road", "The Salt Road"],
     "turns": [
        "I'm writing a novel called The Salt Road.",
        "How do I keep a reader engaged in chapter one?",
        "Any tips for writing believable dialogue?",
        "Do you remember the title of my novel?"]},
    {"id": "p09", "type": "preference", "answer_keys": ["teal"],
     "turns": [
        "My favorite color is teal.",
        "Suggest a color palette for a living room.",
        "What accent colors pair well with grey?",
        "What's my favorite color?"]},
    {"id": "p10", "type": "schedule", "answer_keys": ["Thursday"],
     "turns": [
        "I have a dentist appointment on Thursday.",
        "How should I prepare for a routine cleaning?",
        "Is flossing the night before enough?",
        "Which day is my dentist appointment?"]},
    {"id": "p11", "type": "named-entity", "answer_keys": ["Priya"],
     "turns": [
        "My daughter Priya is starting school next month.",
        "How can I help a child adjust to a new school?",
        "What's a good bedtime routine for a six-year-old?",
        "What is my daughter's name?"]},
    {"id": "p12", "type": "named-entity", "answer_keys": ["Portuguese"],
     "turns": [
        "I'm learning Portuguese for an upcoming move.",
        "What's the fastest way to build vocabulary?",
        "How much daily practice do you recommend?",
        "Which language am I learning?"]},
    {"id": "p13", "type": "medical", "answer_keys": ["penicillin"],
     "turns": [
        "Important: I'm allergic to penicillin.",
        "What are general signs of an allergic reaction?",
        "When should someone use an epinephrine auto-injector?",
        "What am I allergic to?"]},
    {"id": "p14", "type": "number", "answer_keys": ["2000", "2,000", "$2000", "$2,000"],
     "turns": [
        "My budget for a new laptop is 2000 dollars.",
        "What specs matter most for video editing?",
        "Is more RAM or a faster CPU more important?",
        "What was my budget?"]},
    {"id": "p15", "type": "named-entity", "answer_keys": ["Denver"],
     "turns": [
        "I'm moving to Denver in the spring.",
        "What should I know about living at high altitude?",
        "How long does it take to acclimate?",
        "Which city am I moving to?"]},
    {"id": "p16", "type": "preference", "answer_keys": ["cello"],
     "turns": [
        "I play the cello in a community orchestra.",
        "How do I keep my bowing relaxed?",
        "Any advice for sight-reading?",
        "Which instrument do I play?"]},
    {"id": "p17", "type": "schedule", "answer_keys": ["March 14", "March 14th"],
     "turns": [
        "My project deadline is March 14.",
        "How do I plan backwards from a deadline?",
        "What's a good way to handle scope creep?",
        "When is my deadline?"]},
    {"id": "p18", "type": "preference", "answer_keys": ["Arsenal"],
     "turns": [
        "I support Arsenal in the Premier League.",
        "What makes a strong midfield?",
        "How important is squad depth over a season?",
        "Which team do I support?"]},
    {"id": "p19", "type": "secret-word", "answer_keys": ["Albatross", "albatross"],
     "turns": [
        "Let's set a codeword for this session: Albatross.",
        "Tell me a short fact about the ocean.",
        "Now tell me a fact about birds.",
        "What was the codeword I set?"]},
    {"id": "p20", "type": "preference", "answer_keys": ["oat milk"],
     "turns": [
        "My usual coffee order is an oat milk latte, no sugar.",
        "How is a latte different from a flat white?",
        "Does milk type change the foam?",
        "What's my usual coffee order?"]},
]


def contains_key(text, keys):
    low = text.lower()
    return any(k.lower() in low for k in keys)


@torch.no_grad()
def run_conversation(model, tok, turns, device, seed, temperature,
                     max_new_tokens, recursion_on):
    """Play a full multi-turn conversation; return the final assistant reply."""
    model.enable_recursion() if recursion_on else model.disable_recursion()
    model.reset_recursion_state(batch_size=1)   # fresh state per conversation
    history, final = [], ""
    for user_turn in turns:
        history.append({"role": "user", "content": user_turn})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(device)
        torch.manual_seed(seed)   # ON and OFF share the same draws
        out = model.generate(
            **ids, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=0.9, num_beams=1,
            pad_token_id=tok.eos_token_id)
        reply = tok.decode(out[0, ids.input_ids.shape[1]:],
                           skip_special_tokens=True).strip()
        history.append({"role": "assistant", "content": reply})
        final = reply
    return final


def run_ablation(model, tok, probes, device, seeds=(0, 1, 2),
                 temperature=0.7, max_new_tokens=96, verbose=False):
    """Score the probe set ON vs OFF. Returns an aggregate dict."""
    need = len(seeds) // 2 + 1   # majority of seeds
    on_total = off_total = on_only = 0
    per_probe = []
    for probe in probes:
        on_hits = off_hits = 0
        on_reply = off_reply = ""
        for s in seeds:
            r_on = run_conversation(model, tok, probe["turns"], device, s,
                                    temperature, max_new_tokens, True)
            r_off = run_conversation(model, tok, probe["turns"], device, s,
                                     temperature, max_new_tokens, False)
            on_hits += int(contains_key(r_on, probe["answer_keys"]))
            off_hits += int(contains_key(r_off, probe["answer_keys"]))
            on_reply, off_reply = r_on, r_off
        on_ok = on_hits >= need
        off_ok = off_hits >= need
        on_total += int(on_ok)
        off_total += int(off_ok)
        on_only += int(on_ok and not off_ok)
        per_probe.append({"id": probe["id"], "type": probe["type"],
                          "on_hits": on_hits, "off_hits": off_hits,
                          "on_correct": on_ok, "off_correct": off_ok,
                          "on_reply_sample": on_reply, "off_reply_sample": off_reply})
        if verbose:
            print(f"  {probe['id']}: ON {on_hits}/{len(seeds)} "
                  f"OFF {off_hits}/{len(seeds)}  "
                  f"{'ON-only' if (on_ok and not off_ok) else ''}")
    return {"bar2_score": on_total, "off_score": off_total,
            "on_only_correct": on_only, "n_probes": len(probes),
            "seeds": list(seeds), "temperature": temperature,
            "max_new_tokens": max_new_tokens, "per_probe": per_probe}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--output", default="./final_eval.json")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    args, _ = ap.parse_known_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"loading {args.ckpt} ...")
    model = AeonR1ForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.ckpt)

    res = run_ablation(model, tok, PROBES, device,
                       seeds=tuple(range(args.seeds)),
                       temperature=args.temperature,
                       max_new_tokens=args.max_new_tokens, verbose=True)
    res["ckpt"] = args.ckpt
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)

    print(f"\nbar2_score (ON correct):     {res['bar2_score']}/{res['n_probes']}")
    print(f"off_score  (OFF correct):    {res['off_score']}/{res['n_probes']}")
    print(f"on_only    (ON correct, OFF not): {res['on_only_correct']}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
