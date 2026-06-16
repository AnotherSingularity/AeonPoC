# Layer 3 Brief — Batched Attention with Sequentialized Recursion

**Status:** brief only. No implementation until Dylan approves this document.
**Branch:** `aeon-v2` (never touch `stage2-study`).
**This is the riskiest layer.** It changes *when* the recurrent state feeds back
into the transformer — an architectural-semantics change, not just a speed
refactor. Treat behavior preservation as the deliverable; speed is secondary.

---

## 1. Objective

Eliminate the per-token Python loop's inference penalty (currently ~10 tok/s on
a 3090, ~20-30x slower than a vanilla transformer) while preserving the
persistent-state semantics that make Aeon Aeon.

Current (slow): for each token, run the full L-block stack, then advance the
recurrent state. The recurrent **read** (the `gamma * U * r` residual shift)
happens *before* each block, so the state at token t modulates token t's own
attention.

Target (fast): run batched attention over the whole sequence per layer; advance
the recurrent state in a small sequential scan over precomputed per-token
writes.

## 2. The semantic change (why this is risky)

In the per-token loop, the state read at token t reflects tokens 0..t-1 and
modulates token t's attention. In a batched design the state advances *after*
the block stack, so — depending on the concrete formulation — it influences only
the **next** token, not the current one's attention. **That is a real change to
the architecture's dynamics, and it only matters when gamma != 0.** At gamma = 0
the recurrent path contributes nothing, so the batched forward is exactly the
bare transformer (see §4, gate 1). The risk lives entirely in the gamma != 0
regime, i.e. in whether Stage 1 behavior survives.

Hypothesis (must be tested, not assumed): at the Stage 1 plateau gamma ~ 0.029
the recurrent influence is small enough that the read-timing change does not
materially alter training dynamics or ablation behavior. If that hypothesis
fails, see §6 (chunked-batch fallback).

## 3. Hard constraints

- **Do NOT modify `aeon/recursion.py`.** The cell is canonical.
- **Do NOT touch `stage2-study`.**
- **Preserve the `_is_hf_initialized` protection.** `AeonModel.__init__` marks
  the recurrent cell's submodules and each block's `U`/`D_proj` as
  `_is_hf_initialized = True` *before* `post_init()`, so the generic
  `_init_weights` pass does not clobber the canonical cell init or the small
  U/D init. Any restructuring of `AeonModel.__init__` (very likely in Layer 3)
  MUST keep that protection intact, or fresh models silently get the wrong init
  and `init_from_pretrained` produces a subtly wrong cell.
- Keep `recursion_gate` naming and the v2 key layout (`transformer_layer.*`); do
  not reintroduce a serialization shim collision.

## 4. Validation gates (all must hold)

**Gate 1 — Stage 0 byte-identity at gamma = 0 (NO REGRESSION from Layer 2).**
`scripts/verify_wiring.py` must still report worst `max|dlogit|` ~1e-6 in fp32
(random reference ported into Aeon, gamma = 0). On the box, the real Stage 0
(`verify_stage0.py`) must still argmax-match on all three prompts. Because the
recurrent path is inert at gamma = 0, the batched forward must equal the bare
transformer exactly — this gate is the regression check on the new batched
transformer path itself.

**Gate 2 — Stage 1 checkpoint still loads and is non-trivial.** After
`fix_gate_keys.py` + `migrate_to_v2.py`, the June-15 Stage 1 checkpoint must load
into the Layer-3 `AeonForCausalLM` and report `mean|recursion_gate| ~ 0.029`
(via `probe_ablation.py`). If Layer 3 changes any module/attribute names, extend
`migrate_to_v2.py` and document it. (Dylan runs this — no checkpoint in the
sandbox; implement the load path and document the exact commands.)

**Gate 3 — Stage 1 behavior parity (THE risk gate).** Re-train Stage 1 for
~500 steps with the batched architecture and compare against the original
Stage 1 run: gamma trajectory, certificate `holds=True` at every audit, loss
curve, and ablation (`probe_ablation.py`) ON/OFF divergence. "Comparable within
noise" passes; "materially different" triggers the STOP in §5. (Dylan runs this
on the box; the sandbox cannot — document the comparison procedure and the
specific quantities to diff.)

**Gate 4 — Recursion self-test.** `python aeon/recursion.py` still prints atlas
equivalence < 1e-4 and certificate holds at init and after 50 steps.

**Gate 5 — Full test suite green**, plus new tests for the batched forward
(equivalence to the per-token forward at gamma = 0; per-token-vs-batched parity
on a tiny model; generate still works).

## 5. STOP-and-ping condition (explicit)

**If Gate 3 shows meaningful Stage 1 behavior degradation under fully-batched
recursion — gamma trajectory diverging, certificate instability, loss
regression, or ablation divergence collapsing — STOP. Do not commit the
fully-batched design as the new default. Ping Dylan with the comparison data and
the proposed chunked-batch fallback (§6) before continuing.** This is a human
decision point. A real speedup that quietly changes the architecture's behavior
is a failure, not a win.

Gates 1, 2, 4, 5 failing are also stop-and-report conditions, but Gate 3 is the
one that needs a human judgment call rather than a bug fix.

## 6. Sanctioned fallback — chunked-batch (K)

If fully-batched degrades Stage 1 behavior, the implementer is **explicitly
allowed to propose and implement chunked-batch** instead of forcing the
fully-batched design:

- Process the sequence in chunks of K tokens. Within a chunk, attention is
  batched and the recurrent read uses a fixed state (the state from the end of
  the previous chunk). The state advances once per chunk from that chunk's
  aggregated writes.
- `K = 1` recovers the exact per-token semantics (v1); `K = T` is fully batched.
  K becomes a config knob trading speed for per-token expressiveness.
- Pick the smallest K that both clears Gate 3 and delivers a worthwhile speedup,
  and report the chosen K with the supporting comparison.

Propose the concrete formulation (and which of fully-batched vs chunked you are
committing) in the Layer 3 report; do not silently pick one.

## 7. Inference benchmark (nice-to-have, NOT the gate)

Target: **>50 tok/s on a 3090** (from current ~10), measured by a small
`scripts/bench_inference.py` (greedy decode, fixed prompt, warm cache, tok/s
averaged over a run). This is the motivation for Layer 3 and a useful signal,
but **architectural correctness (Gates 1-5) is what determines mergeability.** A
design that hits 100 tok/s but fails Gate 3 does not ship; a design that clears
all gates at 40 tok/s is preferable to one that clears them at 0.

## 8. Sandbox vs on-box split

Runnable in the sandbox (no GPU/HF/checkpoint): Gate 1 (`verify_wiring`, fp32
~1e-6), Gate 4 (recursion self-test), Gate 5 (pytest), a CPU forward-timing
micro-benchmark (per-token vs batched, relative speedup sanity), and the
batched-vs-per-token gamma=0 equivalence test. Document these results.

Runnable only on the box (Dylan): real Stage 0 on the warm-start reference
(Gate 1 argmax), Stage 1 checkpoint load (Gate 2), Stage 1 behavior parity
re-train (Gate 3), and the real tok/s benchmark (§7). Implement the paths and
document the exact commands and the quantities to compare.

---

End of brief.

---

## Addendum — what was delivered (2026-06-16, design approved)

Implemented as **chunked-batch with config knob `recursion_chunk_size` (K)**:
`0` (default) = K=T fully batched; `1` = exact per-token (byte-identical to v1);
`k` = chunks of k. K is exposed precisely so the speed-vs-fidelity tradeoff at
intermediate K can be characterized later.

**Key empirical finding (the §2 risk, made concrete).** At **K=T the recurrent
path gets zero gradient in a single-forward training pass** — the one chunk reads
only the chunk-start (initial) state and never reads the advanced state back,
so cell/U/D/`recursion_gate` are disconnected from the loss. Measured directly:
K=T -> `A_h.grad = None`; K=1 -> `A_h.grad = 7.8e-2`. Caught by gradient
measurement, which is what the STOP condition was written for.

**Resolution (no STOP needed — chunked-K was pre-approved).**
- K=T stays the **inference** default (generation decode steps are single-token,
  so the recursion is live token-to-token during decode; the win is batched
  prefill).
- **Training** must use K < seq_len; `train_stage1.py` / `train_stage2.py`
  default to **K=1** (exact v1 semantics) and warn if K would be a single chunk.
- The risky fully-batched/`K>1` *training* regime is therefore opt-in, not the
  default — Stage 1 re-training at K=1 is the v1 architecture exactly.

**Speed.** Inference speedup is real (batched prefill); on a tiny CPU model the
prefill-dominated bench showed K=T ~6.5x faster than K=1. **Training speedup is
NOT delivered by K=T** (it can't train); a real training speedup requires
chunked `K>1`, which trades fidelity and must be behavior-validated per K.

**Gate 3 instruction.** Run the Stage 1 parity re-train at
`--recursion_chunk_size 1`. Do NOT run it at the K=T default — it will show the
gate frozen (no gradient), which is expected, not a regression.

