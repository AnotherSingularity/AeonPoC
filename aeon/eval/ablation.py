"""
aeon.eval.ablation — recursion ON/OFF ablation over multi-turn probes.

For each probe (a list of user turns whose final turn requires recalling an
earlier fact) the conversation is played out turn by turn with the recursion
path ON and OFF, and the final reply is checked for the probe's answer keys.
Generation is seeded per turn so ON/OFF share the same draws — the recursion is
the only variable.
"""
import torch

DEFAULT_SYSTEM_PROMPT = (
    "You are Aeon, a small language model. Respond as Aeon. Be concise and "
    "direct. Do not narrate your reasoning unless asked."
)


def contains_key(text, keys):
    low = text.lower()
    return any(k.lower() in low for k in keys)


@torch.no_grad()
def run_conversation(model, tok, turns, device, seed, temperature,
                     max_new_tokens, recursion_on, system_prompt=DEFAULT_SYSTEM_PROMPT):
    model.enable_recursion() if recursion_on else model.disable_recursion()
    model.reset_recursion_state(batch_size=1)
    history, final = [], ""
    for user_turn in turns:
        history.append({"role": "user", "content": user_turn})
        messages = ([{"role": "system", "content": system_prompt}] + history
                    if system_prompt else history)
        try:
            prompt = tok.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        except Exception:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
        ids = tok(prompt, return_tensors="pt").to(device)
        torch.manual_seed(seed)
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=True,
                             temperature=temperature, top_p=0.9, num_beams=1,
                             pad_token_id=tok.eos_token_id)
        reply = tok.decode(out[0, ids.input_ids.shape[1]:],
                           skip_special_tokens=True).strip()
        history.append({"role": "assistant", "content": reply})
        final = reply
    return final


def run_ablation(model, tok, probes, device, seeds=(0, 1, 2),
                 temperature=0.7, max_new_tokens=96):
    """Score `probes` ON vs OFF. Returns an aggregate dict.

    Each probe is a dict with `turns` (list of user strings) and `answer_keys`
    (list of acceptable substrings). A side is "correct" if a majority of seeds
    produce a reply containing an answer key.
    """
    need = len(seeds) // 2 + 1
    on_total = off_total = on_only = 0
    per_probe = []
    for probe in probes:
        on_hits = off_hits = 0
        for s in seeds:
            r_on = run_conversation(model, tok, probe["turns"], device, s,
                                    temperature, max_new_tokens, True)
            r_off = run_conversation(model, tok, probe["turns"], device, s,
                                     temperature, max_new_tokens, False)
            on_hits += int(contains_key(r_on, probe["answer_keys"]))
            off_hits += int(contains_key(r_off, probe["answer_keys"]))
        on_ok, off_ok = on_hits >= need, off_hits >= need
        on_total += int(on_ok)
        off_total += int(off_ok)
        on_only += int(on_ok and not off_ok)
        per_probe.append({"id": probe.get("id"), "on_hits": on_hits,
                          "off_hits": off_hits, "on_correct": on_ok,
                          "off_correct": off_ok})
    return {"bar2_score": on_total, "off_score": off_total,
            "on_only_correct": on_only, "n_probes": len(probes),
            "seeds": list(seeds), "per_probe": per_probe}
