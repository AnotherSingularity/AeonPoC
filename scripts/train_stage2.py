"""
scripts/train_stage2.py — Stage 2 study run (frozen backbone, long horizon).

Trains the recursion path only (cell params + per-block U / D_proj /
recursion_gate, plus optional r_init/c_init). The entire backbone stays frozen.
This is the characterization study from docs/STAGE2_BUILDBOOK.md, NOT co-training
(that is Stage 2-B) and NOT a release run.

Features required by the buildbook:
  - length curriculum over the run (3 phases, hard cutoffs, scaled to --steps)
  - gradient accumulation, cosine LR schedule with warmup, grad clipping
  - per-audit contraction check; abort cleanly (with alert) if it ever fails
  - per-layer gate history for Bar 3 differentiation analysis
  - Telegram heartbeat / alerts (optional, env-configured)
  - resume-from-checkpoint with optimizer + scheduler + sampler + RNG state
  - periodic inline ablation eval (Bar 2) + held-out eval loss (Bar 1)
  - a status JSON for scripts/aeon_status.py

Default --steps is 30000 (buildbook Option Y). Pass --steps 60000 for the full
study; the curriculum boundaries scale proportionally either way.

No architecture changes. model.py / recursion.py are untouched.
"""
import os, sys, json, time, math, random, argparse, traceback
from collections import deque

import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aeon.model import AeonR1ForCausalLM
from aeon.recursion import audit_certificates
from aeon.audit import gate_summary
from scripts.telegram_notify import (
    send_telegram, telegram_configured, format_heartbeat, format_alert, fmt_eta)
from scripts.eval_stage2_ablation import run_ablation, PROBES


# --------------------------------------------------------------------------
# Curriculum
# --------------------------------------------------------------------------
# Phase boundaries as fractions of the total step budget. For the buildbook's
# 60k schedule these give 15k / 35k; for Option Y (30k) they give 7500 / 17500,
# matching the 7500 / 10000 / 12500 phase budgets.
PHASE_BOUNDARY_FRACS = (15 / 60, 35 / 60)
# Phase 3 is capped at 1536 tokens (not 4096/2048) to fit a 3090's 24 GB:
# backprop through the per-token loop scales with sequence length. The Phase 3
# token window is narrowed to 1024-1536 to match.
PHASE_FILTERS = {1: (0, 768), 2: (512, 1536), 3: (1024, 1536)}
PHASE_BUDGET = {1: 768, 2: 1536, 3: 1536}      # effective seq_len budget per phase
PHASE_NAMES = {1: "warm-up", 2: "medium", 3: "long"}


def phase_boundaries(total_steps):
    return (round(total_steps * PHASE_BOUNDARY_FRACS[0]),
            round(total_steps * PHASE_BOUNDARY_FRACS[1]))


def phase_for_step(step, total_steps):
    b1, b2 = phase_boundaries(total_steps)
    return 1 if step < b1 else (2 if step < b2 else 3)


def phase_str(step, total_steps):
    b1, b2 = phase_boundaries(total_steps)
    ph = phase_for_step(step, total_steps)
    if ph == 1:
        return f"{PHASE_NAMES[1]} -> {PHASE_NAMES[2]} (next at step {b1})"
    if ph == 2:
        return f"{PHASE_NAMES[2]} -> {PHASE_NAMES[3]} (next at step {b2})"
    return f"{PHASE_NAMES[3]} (final phase)"


class CurriculumSampler:
    """Deterministic, resumable sampler. Phase is a function of the optimizer
    step; within a phase, rows passing the phase's token filter are visited in a
    shuffled order seeded by (seed, phase, epoch)."""

    def __init__(self, rows, total_steps, seed):
        self.rows = rows
        self.total = total_steps
        self.seed = seed
        self.phase = None
        self.epoch = 0
        self.order = []
        self.pos = 0

    def _build_order(self, phase):
        lo, hi = PHASE_FILTERS[phase]
        idx = [i for i, r in enumerate(self.rows) if lo <= r["n_tokens"] <= hi]
        if not idx:                       # never starve the loader
            idx = list(range(len(self.rows)))
        g = torch.Generator().manual_seed(self.seed * 100 + phase * 17 + self.epoch)
        perm = torch.randperm(len(idx), generator=g).tolist()
        self.order = [idx[p] for p in perm]
        self.pos = 0

    def next(self, step):
        ph = phase_for_step(step, self.total)
        if ph != self.phase:              # hard phase transition -> reshuffle
            self.phase, self.epoch = ph, 0
            self._build_order(ph)
        if self.pos >= len(self.order):   # epoch boundary -> reshuffle
            self.epoch += 1
            self._build_order(self.phase)
        row = self.rows[self.order[self.pos]]
        self.pos += 1
        return row

    def state(self):
        return {"phase": self.phase, "epoch": self.epoch, "pos": self.pos}

    def load_state(self, s):
        self.phase, self.epoch = s["phase"], s["epoch"]
        if self.phase is not None:
            self._build_order(self.phase)   # deterministic rebuild
            self.pos = s["pos"]


# --------------------------------------------------------------------------
# Param selection (same set Stage 1 trained)
# --------------------------------------------------------------------------
def is_recursion_param(name):
    return (
        "recursion." in name
        or name.endswith(".U.weight")
        or name.endswith(".D_proj.weight")
        or name.endswith(".recursion_gate")
        or "r_init" in name
        or "c_init" in name
    )


def freeze_backbone(model):
    for name, p in model.named_parameters():
        p.requires_grad_(is_recursion_param(name))
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in trainable)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"trainable: {n_tr:,} / {n_all:,} ({100*n_tr/n_all:.2f}%)  [backbone frozen]")
    return trainable


# --------------------------------------------------------------------------
# Checkpoint / resume
# --------------------------------------------------------------------------
def save_checkpoint(out_dir, step, model, tok, opt, sched, sampler, args, mean_gate_start):
    d = os.path.join(out_dir, f"step_{step}")
    os.makedirs(d, exist_ok=True)
    model.save_pretrained(d)
    tok.save_pretrained(d)
    state = {
        "step": step,
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "sampler": sampler.state(),
        "mean_gate_start": mean_gate_start,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "args": vars(args),
    }
    torch.save(state, os.path.join(d, "training_state.pt"))
    # maintain a `latest` pointer for exfil (symlink; rclone needs --copy-links)
    latest = os.path.join(out_dir, "latest")
    try:
        if os.path.islink(latest) or os.path.exists(latest):
            os.remove(latest)
        os.symlink(os.path.abspath(d), latest)
    except OSError:
        pass
    print(f"checkpoint saved: {d}")
    return d


def restore_training_state(resume_dir, opt, sched, sampler):
    # weights_only=False: this is our own training-state pickle (optimizer,
    # sampler, RNG, numpy state), not an untrusted tensor file.
    state = torch.load(os.path.join(resume_dir, "training_state.pt"),
                       map_location="cpu", weights_only=False)
    opt.load_state_dict(state["optimizer"])
    sched.load_state_dict(state["scheduler"])
    sampler.load_state(state["sampler"])
    torch.set_rng_state(state["torch_rng"])
    if state.get("cuda_rng") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        except Exception:
            pass
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    return state["step"], state.get("mean_gate_start")


# --------------------------------------------------------------------------
# Eval helpers
# --------------------------------------------------------------------------
@torch.no_grad()
def eval_loss(model, tok, rows, device, max_len):
    if not rows:
        return float("nan")
    model.eval()
    tot, n = 0.0, 0
    for r in rows:
        enc = tok(r["text"], return_tensors="pt", truncation=True, max_length=max_len)
        ids = enc.input_ids.to(device)
        attn = enc.attention_mask.to(device)
        model.reset_recursion_state(batch_size=1)
        out = model(input_ids=ids, attention_mask=attn, labels=ids.clone())
        tot += out.loss.item()
        n += 1
    return tot / max(1, n)


def write_status(path, payload):
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
    except OSError as e:
        print(f"[status] could not write {path}: {e}")


def append_gamma_history(path, step, summary):
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps({"step": step,
                                "gamma_per_layer": summary["gamma_per_layer"],
                                "mean_abs": summary["mean_abs"],
                                "stdev_abs": summary["stdev_abs"]}) + "\n")
    except OSError as e:
        print(f"[gamma_history] could not write {path}: {e}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="/workspace/aeon_init",
                    help="Stage 1 fixed checkpoint to start from")
    ap.add_argument("--data", required=True, help="ultrachat.jsonl from prepare_ultrachat.py")
    ap.add_argument("--out", default="/workspace/aeon_stage2")
    ap.add_argument("--resume", default=None, help="a step_<N> dir to resume from")
    ap.add_argument("--steps", type=int, default=30000)        # Option Y default
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_seq_len", type=int, default=1536)   # 3090 VRAM cap
    ap.add_argument("--lr_recursion", type=float, default=5e-5)
    ap.add_argument("--lr_backbone", type=float, default=0.0)  # frozen; informational
    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--audit_every", type=int, default=200)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--heartbeat_every", type=int, default=1000)
    ap.add_argument("--checkpoint_every", type=int, default=1000)
    ap.add_argument("--gamma_history_every", type=int, default=500)
    ap.add_argument("--eval_every", type=int, default=5000)
    ap.add_argument("--eval_seeds", type=int, default=3)
    ap.add_argument("--eval_max_new_tokens", type=int, default=96)
    ap.add_argument("--eval_loss_rows", type=int, default=200)
    ap.add_argument("--no_eval", action="store_true", help="skip inline eval (smoke runs)")
    ap.add_argument("--gamma_history_file", default="/workspace/gamma_history.jsonl")
    ap.add_argument("--status_file", default="/workspace/aeon_status.json")
    args, _ = ap.parse_known_args()
    return args


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    init_dir = args.resume or args.init
    print(f"loading model from {init_dir} ...")
    model = AeonR1ForCausalLM.from_pretrained(
        init_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    tok = AutoTokenizer.from_pretrained(init_dir)
    trainable = freeze_backbone(model)

    opt = torch.optim.AdamW(trainable, lr=args.lr_recursion, weight_decay=args.weight_decay)
    sched = get_cosine_schedule_with_warmup(opt, args.warmup_steps, args.steps)

    # Data: hold out a fixed eval-loss slice, sample the rest by curriculum.
    with open(args.data) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for r in rows:
        r.setdefault("n_tokens", len(tok(r["text"], add_special_tokens=False).input_ids))
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    n_hold = min(args.eval_loss_rows, len(rows) // 10)
    holdout = [rows[i] for i in order[:n_hold]]
    train_rows = [rows[i] for i in order[n_hold:]]
    print(f"{len(train_rows)} train rows, {len(holdout)} held-out for eval-loss")
    sampler = CurriculumSampler(train_rows, args.steps, args.seed)

    start_step, mean_gate_start = 0, None
    if args.resume:
        start_step, mean_gate_start = restore_training_state(args.resume, opt, sched, sampler)
        print(f"resumed at step {start_step}")
    if mean_gate_start is None:
        mean_gate_start = gate_summary(model)["mean_abs"]   # ~0.029 from Stage 1

    if not telegram_configured():
        print("[telegram] env vars unset — heartbeats disabled, logging locally only.")

    model.train()
    recent = deque(maxlen=max(1, 1000 // max(1, args.grad_accum)))
    step_times = deque(maxlen=200)
    last_audit_step, last_holds = start_step, True
    last_eval = {"step": None, "bar2": None, "eval_loss": None}
    start_time = time.time()

    def status_payload(step, loss, gs):
        lo = min(recent) if recent else loss
        hi = max(recent) if recent else loss
        avg_t = sum(step_times) / len(step_times) if step_times else None
        eta = avg_t * (args.steps - step) if avg_t else None
        return {
            "pid": os.getpid(), "start_time": start_time, "last_update_time": time.time(),
            "step": step, "total_steps": args.steps, "loss": loss,
            "loss_lo": lo, "loss_hi": hi,
            "mean_gate": gs["mean_abs"], "mean_gate_start": mean_gate_start,
            "gate_stdev": gs["stdev_abs"], "holds": last_holds,
            "last_audit_step": last_audit_step,
            "phase": PHASE_NAMES[phase_for_step(step, args.steps)],
            "phase_str": phase_str(step, args.steps),
            "phase_boundaries": list(phase_boundaries(args.steps)),
            "last_eval_step": last_eval["step"], "last_eval_bar2": last_eval["bar2"],
            "last_eval_loss": last_eval["eval_loss"],
            "avg_step_time": avg_t, "eta_seconds": eta,
        }

    try:
        for step in range(start_step, args.steps):
            t0 = time.time()
            budget = min(args.max_seq_len, PHASE_BUDGET[phase_for_step(step, args.steps)])
            opt.zero_grad(set_to_none=True)
            micro_loss = 0.0
            for _ in range(args.grad_accum):
                row = sampler.next(step)
                enc = tok(row["text"], return_tensors="pt",
                          truncation=True, max_length=budget)
                ids = enc.input_ids.to(device)
                attn = enc.attention_mask.to(device)
                model.reset_recursion_state(batch_size=1)
                out = model(input_ids=ids, attention_mask=attn, labels=ids.clone())
                (out.loss / args.grad_accum).backward()
                micro_loss += out.loss.item()
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            opt.step()
            sched.step()

            step_done = step + 1
            avg_loss = micro_loss / args.grad_accum
            recent.append(avg_loss)
            step_times.append(time.time() - t0)
            gs = gate_summary(model)

            if step_done % args.log_every == 0 or step_done == args.steps:
                print(f"step {step_done}  loss {avg_loss:.4f}  mean|g|={gs['mean_abs']:.4f}")
                write_status(args.status_file, status_payload(step_done, avg_loss, gs))

            if step_done % args.audit_every == 0:
                a = audit_certificates(model.model.recursion)
                last_holds = a["chart_A_holds"]
                last_audit_step = step_done
                print(f"audit: sigma(Wh)={a['chart_A_sigma_Wh']:.4f} "
                      f"sigma(Wc)={a['chart_A_sigma_Wc']:.4f} holds={last_holds}")
                if not last_holds:
                    msg = (f"CERT VIOLATION at step {step_done}: "
                           f"sigma(Wh)={a['chart_A_sigma_Wh']:.4f} "
                           f"sigma(Wc)={a['chart_A_sigma_Wc']:.4f}")
                    print(msg)
                    send_telegram(format_alert(step_done, msg))
                    save_checkpoint(args.out, step_done, model, tok, opt, sched,
                                    sampler, args, mean_gate_start)
                    sys.exit(2)

            if step_done % args.gamma_history_every == 0:
                append_gamma_history(args.gamma_history_file, step_done, gs)

            if step_done % args.heartbeat_every == 0:
                avg_t = sum(step_times) / len(step_times) if step_times else None
                eta = avg_t * (args.steps - step_done) if avg_t else None
                send_telegram(format_heartbeat(
                    step_done, args.steps, avg_loss,
                    min(recent), max(recent), gs["mean_abs"], gs["stdev_abs"],
                    last_holds, last_audit_step, phase_str(step_done, args.steps),
                    eta, gate_delta=gs["mean_abs"] - mean_gate_start))

            if step_done % args.checkpoint_every == 0 or step_done == args.steps:
                save_checkpoint(args.out, step_done, model, tok, opt, sched,
                                sampler, args, mean_gate_start)

            if not args.no_eval and step_done % args.eval_every == 0:
                el = eval_loss(model, tok, holdout, device, args.max_seq_len)
                res = run_ablation(model, tok, PROBES, device,
                                   seeds=tuple(range(args.eval_seeds)),
                                   temperature=0.7,
                                   max_new_tokens=args.eval_max_new_tokens)
                last_eval = {"step": step_done, "bar2": res["bar2_score"], "eval_loss": el}
                print(f"eval @ step {step_done}: bar2_score={res['bar2_score']}/{res['n_probes']} "
                      f"on_only={res['on_only_correct']} eval_loss={el:.4f}")
                model.train()

        # ---- end of run ----
        final = gate_summary(model)
        bar1 = final["mean_abs"] > 0.05 and last_holds
        bar3 = final["stdev_abs"] > 0.02
        bar2_score = last_eval["bar2"]
        def verdict(ok): return "PASS" if ok else "PARTIAL/FAIL"
        summary = (
            f"Aeon Stage 2 complete (step {args.steps}).\n"
            f"  mean|gate|: {final['mean_abs']:.3f} (Bar 1: {verdict(bar1)})\n"
            f"  gate stdev: {final['stdev_abs']:.3f} (Bar 3: {verdict(bar3)})\n"
            f"  bar2 score: {bar2_score}/{len(PROBES) if bar2_score is not None else '-'} "
            f"(Bar 2: {'see final eval' if bar2_score is None else verdict(bar2_score > 10)})\n"
            f"Outputs under {args.out}")
        print(summary)
        send_telegram(summary)

    except SystemExit:
        raise
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        send_telegram(format_alert(locals().get("step_done", start_step),
                                   "training crashed:\n" + tb[-1500:]))
        raise


if __name__ == "__main__":
    main()
