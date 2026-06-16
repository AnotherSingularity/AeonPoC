"""
scripts/aeon_status.py — one-shot Stage 2 health/progress dashboard.

Reads the status JSON written by train_stage2.py (default
/workspace/aeon_status.json), checks the training process and GPU, and prints a
summary. Loud about the failure modes that matter on a multi-week run: dead
process, stale/hung log, certificate violation.

Usage:
    python scripts/aeon_status.py
    python scripts/aeon_status.py --status /workspace/aeon_status.json
"""
import os, sys, json, time, argparse, subprocess


def human_dt(seconds):
    seconds = int(max(0, seconds))
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d:
        return f"{d}d {h}h"
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def proc_info(pid):
    """Return (alive, detail) for a pid, using psutil if available."""
    if pid is None:
        return False, "no pid recorded"
    try:
        import psutil
        if not psutil.pid_exists(pid):
            return False, f"PID {pid} not found"
        p = psutil.Process(pid)
        cpu = p.cpu_percent(interval=0.2)
        started = time.time() - p.create_time()
        return True, f"PID {pid}, CPU {cpu:.0f}%, started {human_dt(started)} ago"
    except ImportError:
        alive = os.path.isdir(f"/proc/{pid}")
        return alive, (f"PID {pid} ({'alive' if alive else 'dead'}); "
                       f"install psutil for detail")
    except Exception as e:
        return False, f"PID {pid} check failed: {e}"


def gpu_info():
    try:
        q = ("utilization.gpu,memory.used,memory.total,temperature.gpu")
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return "nvidia-smi error"
        util, used, total, temp = [x.strip() for x in out.stdout.strip().split(",")]
        return f"{util}% util, {float(used)/1024:.1f} / {float(total)/1024:.1f} GB VRAM, {temp}°C"
    except Exception:
        return "n/a (no nvidia-smi)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", default="/workspace/aeon_status.json")
    ap.add_argument("--stale_secs", type=int, default=300)
    args, _ = ap.parse_known_args()

    print("== AEON STAGE 2 STATUS ==")
    if not os.path.isfile(args.status):
        print(f"NO STATUS FILE at {args.status}.")
        print("Training has not written status yet, or the path is wrong.")
        print("==")
        sys.exit(1)

    with open(args.status) as f:
        s = json.load(f)

    alive, detail = proc_info(s.get("pid"))
    last_write = time.time() - s.get("last_update_time", 0)
    stale = alive and last_write > args.stale_secs

    if alive and stale:
        print(f"Process:        HUNG?  {detail} — no status update for {human_dt(last_write)}")
    elif alive:
        print(f"Process:        RUNNING ({detail})")
    else:
        print(f"Process:        DEAD — {detail}")

    print(f"GPU:            {gpu_info()}")
    print(f"Last status:    {human_dt(last_write)} ago")

    step, total = s.get("step", 0), s.get("total_steps", 0)
    pct = 100.0 * step / total if total else 0.0
    eta = s.get("eta_seconds")
    print(f"Last step:      {step} / {total} ({pct:.1f}%, ETA "
          f"{human_dt(eta) if eta else '?'})")
    print(f"Last loss:      {s.get('loss', float('nan')):.3f} "
          f"(range {s.get('loss_lo', 0):.1f}-{s.get('loss_hi', 0):.1f} recent)")
    mg, mg0 = s.get("mean_gate", 0.0), s.get("mean_gate_start", 0.0)
    print(f"mean|gate|:     {mg:.3f} (up from {mg0:.3f} at start of Stage 2)")
    gsd = s.get("gate_stdev", 0.0)
    print(f"gate stdev:     {gsd:.3f} (Bar 3 target: > 0.02){'  <-- met' if gsd > 0.02 else ''}")

    holds = s.get("holds", None)
    if holds is False:
        print(f"Certificate:    CERT VIOLATION (last audit step {s.get('last_audit_step')})")
    else:
        print(f"Certificate:    holds={holds} (last audit step {s.get('last_audit_step')})")

    le_step, le_bar2 = s.get("last_eval_step"), s.get("last_eval_bar2")
    if le_step is not None:
        print(f"Last eval:      step {le_step} — bar2 {le_bar2}/20 "
              f"(Bar 2 target: > 10/20){'  <-- met' if (le_bar2 or 0) > 10 else ''}")
    else:
        print("Last eval:      none yet")
    print(f"Phase:          {s.get('phase_str', s.get('phase', '?'))}")
    print("==")

    if not alive:
        print("!! TRAINING PROCESS IS NOT RUNNING !!")
        sys.exit(1)
    if stale:
        print("!! LOG IS STALE — POSSIBLE HANG !!")
        sys.exit(1)


if __name__ == "__main__":
    main()
