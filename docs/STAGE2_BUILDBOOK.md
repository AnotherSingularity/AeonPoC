# Aeon Stage 2 Study Buildbook

**Purpose:** Characterize long-horizon behavior of the Aeon hybrid architecture under realistic training. Not a release. Not a capability run. A *study* designed to produce evidence about whether the architecture develops three named signatures over extended training.

**Hand this entire document to Claude Code.** He produces the data prep script, training script, eval script, status script, and documentation, then commits. Dylan reviews, then launches.

---

## 1. Study Design

### 1.1 What we are measuring

After Stage 1 (2000 steps, alpaca, frozen backbone) we know:

- The architecture trains without breaking the contraction certificate.
- γ stabilized at ~0.029 mean magnitude with stdev ~0.030 (signed).
- On a 5-prompt ablation probe (greedy, post-key-fix), 4/5 prompts diverged between ON and OFF runs, but only one (capital-of-France) was clearly *better* with recursion than without; the others were "different degenerate loops."

What we do not yet know:

- Whether γ continues to grow past 0.05 with more training.
- Whether per-layer γ values *differentiate* (some layers high, some low) as the recursion finds specialized roles.
- Whether the recursion produces visible long-context recall benefits on data that actually requires multi-turn memory.

Stage 2 is the experiment designed to answer those three questions over a single long training run.

### 1.2 The three bars

Stage 2 succeeds (architecture is characterized; Dylan can decide what to do next) if all three bars are cleared by the end of training. Partial clearance still produces useful data but is not "the architecture is understood."

**Bar 1 — Architectural validation.**
- γ mean magnitude > 0.05 by end of training
- Contractive certificate `holds=True` at every audit through the entire run
- Eval-set loss trends downward across training (not just train loss)

**Bar 2 — Capability evidence.**
- Multi-turn ablation probes show Aeon-ON producing materially more coherent responses than Aeon-OFF on conversations with ≥3 user turns
- At least one held-out prompt where ON correctly references something from 3+ turns earlier and OFF does not
- This is evaluated at end-of-training on a fixed probe set of 20 multi-turn prompts

**Bar 3 — Functional differentiation.**
- Per-layer γ values show non-trivial structure (not all clustered at the same value)
- Stdev of per-layer γ magnitudes > 0.02 by end of training
- Visually inspectable per-layer γ histogram shows distinguishable groups

### 1.3 What this study is NOT

- Not Stage 2-B (co-training). Backbone stays frozen.
- Not a release. Aeon-v0.2 designation is reserved for a model that clears all three bars.
- Not optimized for capability. If the recursion develops weakly, that is data, not failure.
- Not a benchmark run. No external benchmark scores. The probe set is internal.

---

## 2. Data

### 2.1 Dataset: UltraChat (multi-turn dialogue)

Single dataset. UltraChat is the cleanest signal available for "did the recursion learn to use persistent state across conversational context." Mixing in OpenOrca / LongAlpaca / raw text was considered and rejected for Stage 2 because attribution becomes impossible if results vary across datasets.

**Source:** `HuggingFaceH4/ultrachat_200k` on HF. Use the `train_sft` split.

**Filtering:**
- Keep only conversations with ≥3 user turns (we want recurrence to matter).
- Drop any conversation where total token length (across all turns combined, after tokenization) exceeds 4096.
- Drop any conversation where total token length is < 200 (too short to exercise recurrence).
- Target: ~30,000 filtered conversations. If filtering yields more, randomly subsample to 30,000 for reproducibility.

**Output:** One JSONL file with one row per conversation. Each row:
```json
{"text": "<formatted conversation with chat template applied>", "n_turns": 4, "n_tokens": 1247}
```

The `n_turns` and `n_tokens` fields are precomputed so the curriculum scheduler can filter by them without re-tokenizing.

### 2.2 Curriculum

Length curriculum over 60,000 training steps, three phases:

| Phase | Steps | Filter | Effective seq_len budget |
|-------|-------|--------|--------------------------|
| Phase 1 (warm-up) | 0 - 15,000 | n_tokens ≤ 768 | up to 768 |
| Phase 2 (medium) | 15,000 - 35,000 | n_tokens 512 - 1536 | up to 1536 |
| Phase 3 (long) | 35,000 - 60,000 | n_tokens 1024 - 4096 | up to 2048 (truncate longer) |

Phase transitions are hard cutoffs. At step 15,001 the data loader switches to the Phase 2 filter and reshuffles.

### 2.3 Eval probe set

Fixed set of 20 multi-turn prompts held out from training, evaluated periodically. These are written by hand (see Appendix A for the list). Each probe is a multi-turn conversation with a verifiable "earlier-context recall" challenge — the final turn requires referencing something stated 2-5 turns earlier.

Eval ablation runs every 5,000 steps. For each of the 20 probes:
- Run with recursion ON (3 seeds, sampling temperature 0.7)
- Run with recursion OFF (3 seeds, same temperature)
- Score: does ON correctly reference the earlier-context fact more often than OFF?

Aggregate score across the 20 probes is logged. This is the primary Bar 2 evidence.

---

## 3. Training Configuration

```python
# scripts/train_stage2.py defaults
batch_size            = 1
max_seq_len           = 2048   # truncate Phase 3 examples to this
gradient_accumulation = 4      # effective batch = 4
lr_recursion          = 5e-5   # lower than Stage 1's 1e-4 since training is much longer
lr_backbone           = 0      # frozen
total_steps           = 60000
warmup_steps          = 1000
lr_schedule           = "cosine"
weight_decay          = 0.01
grad_clip             = 1.0
seed                  = 42
audit_every           = 200
log_every             = 50
heartbeat_every       = 1000
checkpoint_every      = 1000
eval_every            = 5000
```

### 3.1 What gets trained

Same parameter set as Stage 1:
- All recursion cell parameters (Cayley-D matrices, decay, projections)
- Per-block `U`, `D_proj`, `recursion_gate`
- Optional learned initial states `r_init`, `c_init` if enabled in config

Everything else (the entire R1-Distill backbone) stays frozen with `requires_grad_(False)`.

### 3.2 Mixed precision

bf16 for forward pass, fp32 for the recursion cell internals (cayley solve, sigma_max readoff). The patches from Stage 1 are already in the repo. Verify `aeon/recursion.py` has the `orig_dtype` cast pattern before launching.

### 3.3 Resume-from-checkpoint

Training script must support `--resume <ckpt_dir>` to restart from a saved checkpoint and continue at the saved step number. Optimizer state, scheduler state, RNG state must all be saved. Without this, a 14-day run is one hardware blip away from being lost.

---

## 4. Telemetry & Monitoring

### 4.1 Local log

Standard text log written to `/workspace/train.log` via `tee`. Every line is one of:

```
step <N>  loss <L>  mean|g|=<M>
audit: sigma(Wh)=<X> sigma(Wc)=<Y> holds=<T/F>
checkpoint saved: ./aeon_stage2/step_<N>
eval @ step <N>: bar2_score=<S>/20
```

### 4.2 Per-layer γ histogram

Every 500 steps, write a separate file `/workspace/gamma_history.jsonl` with one row:

```json
{"step": 1500, "gamma_per_layer": [-0.009, 0.024, -0.025, ..., 0.038], "mean_abs": 0.029, "stdev_abs": 0.012}
```

This is the Bar 3 evidence trail. At end of training the file lets you plot per-layer trajectories and inspect differentiation.

### 4.3 Telegram heartbeat

Every 1000 steps, POST to Telegram via bot:

```
Aeon Stage 2 — step 14000/60000 (23.3%, ETA 11d 7h)
loss: 1.847 (range 1.6-2.4 last 1000 steps)
mean|gate|: 0.044 (Δ +0.015 since Stage 1 start)
gate stdev across layers: 0.018
certs: holds=True (last audit step 13800)
phase: warm-up → medium (next at step 15000)
```

On any traceback or `holds=False`, send an immediate alert message and exit cleanly (don't keep running with a broken certificate).

Token and chat_id are read from environment variables `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`. Heartbeats are optional — if the env vars are unset, the script logs locally only and prints a notice.

### 4.4 Status script

`scripts/aeon_status.py` — single command to check whether training is alive and progressing. Prints:

```
== AEON STAGE 2 STATUS ==
Process:        RUNNING (PID 12345, CPU 99%, started 4d 12h ago)
GPU:            34% util, 18.4 / 24 GB VRAM, 67°C
Last log write: 18 seconds ago
Last step:      14823 / 60000 (24.7%, ETA 16d 4h)
Last loss:      1.847 (range 1.6-2.4 last 100 steps)
mean|gate|:     0.044 (up from 0.029 at start of Stage 2)
gate stdev:     0.018 (Bar 3 target: > 0.02)
Certificate:    holds=True (last audit at step 14800)
Last eval:      step 10000 — bar2 score 7/20 (Bar 2 target: > 10/20)
Phase:          warm-up (15000 step transition in 177 steps)
==
```

If process is dead, says so loudly. If log is stale (> 5 min since last write while process is alive), says "HUNG?" loudly. If certs ever showed `holds=False`, says "CERT VIOLATION at step N" loudly.

Run it from SSH any time: `python scripts/aeon_status.py`.

---

## 5. Hardware & Cost

### 5.1 Box

Vast 3090 24GB, on-demand, 50GB disk, PyTorch image. Same workflow as Stage 1. **Pin reliability ≥99%** when picking the box; over 14-21 days, 99% means ~3 hours of expected downtime which checkpoints absorb. 98% means 7+ hours which is more annoying.

### 5.2 Duration projection

At batch=1 grad_accum=4, with seq_len growing across curriculum phases, average step time on a 3090:

- Phase 1 (≤768 tokens): ~25 sec/step → 15k steps = ~104 hours
- Phase 2 (≤1536 tokens): ~50 sec/step → 20k steps = ~278 hours
- Phase 3 (≤2048 tokens): ~70 sec/step → 25k steps = ~486 hours

Total: ~868 hours = **~36 days**.

At $0.143/hr: **~$124**.

That's significantly more than the "cheap pass" you originally wanted. Three options:

**Option X — accept ~$120 / 5 weeks.** Get the real study.

**Option Y — halve total steps to 30k (phase budgets 7500/10000/12500).** Cuts cost to ~$60, cuts duration to ~18 days. Risk: 30k may not be enough for Bar 3 (per-layer differentiation) to emerge. Bars 1 and 2 likely still achievable.

**Option Z — cut Phase 3 entirely.** Train 35k steps over Phases 1+2 only (no >1536 token sequences). Cuts cost to ~$55, cuts duration to ~16 days. Trade: never tests the architecture on its longest-context use case.

**Recommend Option Y as default.** Drops cost meaningfully, keeps all three phases, still gives ~3x more training than Stage 1. If end-of-Stage-2 results are promising but Bar 3 is marginal, continuation to the full 60k is justifiable as Stage 2-extended.

Dylan's call. If this document says "Y" we proceed at 30k steps.

### 5.3 Checkpoint exfil

Every 1000 steps, save to `/workspace/aeon_stage2/step_<N>/`. Once per day (or on phase transitions), the training script also runs:

```bash
rclone copy /workspace/aeon_stage2/latest gdrive:aeon_stage2_runs/checkpoint_<N> -P
```

This way checkpoints are durable across the inevitable Vast instance issues. If the box dies mid-training, the latest exfiled checkpoint is on Drive and the run can be restarted on a fresh box with `--resume`.

---

## 6. Repository Changes

Branch off main with `stage2-study`. New files:

```
scripts/prepare_ultrachat.py     # data prep, writes ultrachat.jsonl
scripts/train_stage2.py          # full Stage 2 training script (replaces existing stub)
scripts/eval_stage2_ablation.py  # 20-probe eval, ablation, logging
scripts/aeon_status.py           # status dashboard
docs/STAGE2_PROBE_SET.md         # the 20 held-out probes
docs/STAGE2_BUILDBOOK.md         # this document, committed for reference
```

Modify:
```
aeon/audit.py                    # add gate_stdev_across_layers() helper
```

Tests:
```
tests/test_stage2_curriculum.py  # smoke test for curriculum scheduler
tests/test_stage2_resume.py      # smoke test for resume-from-checkpoint
tests/test_telegram.py           # mock-based test for heartbeat function
```

No changes to architecture. No changes to recursion code. No changes to model.py.

---

## 7. Launch Sequence

### 7.1 Pre-launch checklist (on Dylan's laptop)

- [ ] Stage 2 branch committed and pushed by Claude Code
- [ ] Local `aeon_check_dir` pulled with new branch checked out
- [ ] Stage 1 fixed checkpoint (`final_fixed/`) on Drive, accessible via rclone
- [ ] Telegram bot token + chat ID saved to password manager
- [ ] GitHub PAT created with 30-day expiry, repo read-only scope
- [ ] rclone config block saved separately (will paste onto box)

### 7.2 On the Vast box

```bash
# Setup
cd /workspace
git clone "https://<PAT>@github.com/AnotherSingularity/-.-.git" aeon-proj
cd aeon-proj
git checkout stage2-study
pip install -e ".[dev,data]"

# Verify offline gates one more time
python aeon/recursion.py | tail -3
python scripts/verify_wiring.py | tail -3

# Pull Stage 1 fixed checkpoint as init
mkdir -p /workspace/aeon_init
rclone copy gdrive:aeon_runs/stage1_20260615_fixed /workspace/aeon_init
ls -la /workspace/aeon_init/    # should show model.safetensors + configs

# Prepare data
python scripts/prepare_ultrachat.py --out /workspace/ultrachat.jsonl
# expect: ~30k rows, ~5-10 min

# Set env vars (one session)
export TELEGRAM_BOT_TOKEN="<your_token>"
export TELEGRAM_CHAT_ID="<your_chat_id>"
export HF_TOKEN="<optional_hf_token_for_higher_rate_limits>"

# Test telegram once
python -c "
import os, urllib.request, urllib.parse
url = f'https://api.telegram.org/bot{os.environ[\"TELEGRAM_BOT_TOKEN\"]}/sendMessage'
data = urllib.parse.urlencode({'chat_id': os.environ['TELEGRAM_CHAT_ID'], 'text': 'Aeon Stage 2 launch test'}).encode()
print(urllib.request.urlopen(url, data=data).status)
"
# expect: 200, and a Telegram message lands on Dylan's phone

# Launch
apt-get install -y tmux 2>/dev/null || true
tmux new-session -d -s train \
  "python scripts/train_stage2.py \
     --init /workspace/aeon_init \
     --data /workspace/ultrachat.jsonl \
     --out /workspace/aeon_stage2 \
     --steps 30000 \
     --batch_size 1 \
     --grad_accum 4 \
     2>&1 | tee /workspace/train.log"

# Verify it's running
sleep 60
tail -30 /workspace/train.log
python scripts/aeon_status.py
```

After that the box is autonomous. Dylan checks via Telegram pings or by SSH-ing in and running `python scripts/aeon_status.py`.

### 7.3 Mid-training observations to expect

By step 5000:
- Loss should be trending down on train, eval-loss first datapoint logged
- mean|gate| should have grown from 0.029 to ~0.035-0.040
- Per-layer gate values still relatively bunched (stdev ~0.015)

By step 15000 (Phase 1 → 2 transition):
- mean|gate| 0.040-0.050
- First per-layer differentiation visible in the histogram
- Eval-set bar2 score 5-10/20

By step 30000 (end of cheap run):
- mean|gate| target 0.05+ (Bar 1)
- gate stdev target > 0.02 (Bar 3)
- bar2 score target > 10/20 (Bar 2)

If any of those targets are clearly out of reach by step 25000 (e.g. mean|gate| stuck at 0.035), abort early. The certificate-violated case aborts automatically.

### 7.4 End-of-training procedure

```bash
# Confirm final checkpoint exists
ls -la /workspace/aeon_stage2/step_30000/

# Final eval
python scripts/eval_stage2_ablation.py \
  --ckpt /workspace/aeon_stage2/step_30000 \
  --output /workspace/final_eval.json

# Exfil everything
rclone copy /workspace/aeon_stage2/step_30000 \
  gdrive:aeon_stage2_runs/final_$(date +%Y%m%d) -P
rclone copy /workspace/train.log \
  gdrive:aeon_stage2_runs/final_$(date +%Y%m%d)/
rclone copy /workspace/gamma_history.jsonl \
  gdrive:aeon_stage2_runs/final_$(date +%Y%m%d)/
rclone copy /workspace/final_eval.json \
  gdrive:aeon_stage2_runs/final_$(date +%Y%m%d)/

# Destroy instance from Vast dashboard
```

Final telegram message from the training script:

```
Aeon Stage 2 complete.
Final stats:
  mean|gate|: 0.052 (Bar 1: PASS)
  gate stdev: 0.024 (Bar 3: PASS)
  bar2 score: 12/20 (Bar 2: PASS)
Outputs at gdrive:aeon_stage2_runs/final_<date>/
```

(or PARTIAL / FAIL on each bar as appropriate)

---

## 8. Success Criteria & Decision Tree

After Stage 2 completes, Dylan inspects results against the three bars:

**All three bars cleared:** Architecture characterized. Aeon-v0.2 designation earned. Next decision is whether to do Stage 2-B (co-training), scale up to a larger base model, or pause to write up findings.

**Bars 1 and 2 cleared, Bar 3 marginal:** Architecture works, recursion is learning specialized roles weakly. Consider extending training to 60k steps before moving on. Likely worth the additional ~$60.

**Bar 1 cleared, Bars 2 and 3 not:** Architecture is structurally sound but recursion isn't developing strong capability behavior. Investigate: is the data exercising recurrence sufficiently? Is H_rec=256 too small? Don't proceed to Stage 2-B until this is resolved.

**Bar 1 not cleared (γ plateaued below 0.05 or certificate violated):** Architecture has a problem at this scale. Halt. Diagnose. Don't move forward.

Document outcome in `docs/STAGE2_RESULTS.md` regardless.

---

## Appendix A: The 20 multi-turn probes

(To be authored by Claude Code, drawing from these patterns)

Each probe is a 3-5 turn conversation where the final user turn requires referencing earlier context. Examples:

**Probe type 1 — named entity recall**
```
turn 1: I'm planning a trip to Lisbon with my friend Sarah.
turn 2: Can you suggest three neighborhoods?
turn 3: Great. What about restaurants in the second one you mentioned?
turn 4: What was my friend's name again?
```

**Probe type 2 — preference recall**
```
turn 1: I'm vegetarian and allergic to nuts.
turn 2: Give me a dinner recipe.
turn 3: Now suggest a dessert.
turn 4: What are my dietary restrictions?
```

**Probe type 3 — number recall**
```
turn 1: My favorite numbers are 17 and 42.
turn 2: Tell me a fact about prime numbers.
turn 3: What's an interesting property of even numbers?
turn 4: What were my two favorite numbers?
```

Twenty of these, varied in topic and difficulty. Authored by Claude Code, committed to `docs/STAGE2_PROBE_SET.md`.

---

## Appendix B: What is NOT in scope for Stage 2

To prevent scope creep, the following are explicit non-goals:

- Architecture changes (no H_rec resize, no margin changes, no new chart)
- Backbone co-training (that's Stage 2-B)
- Scaling to a larger base model (that's Stage 3)
- Public release or external publication
- Benchmark scoring (HELM, MMLU, etc.)
- Adding more datasets
- Adding RLHF or DPO
- Fine-tuning chat formatting beyond what UltraChat naturally provides

If any of these come up during Stage 2 execution, Dylan defers them to Stage 3 planning.

---

End of buildbook.
