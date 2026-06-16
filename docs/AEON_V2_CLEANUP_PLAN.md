# Aeon V2 Cleanup Plan

**Purpose:** Transform the repository from "Aeon as an architectural modification of R1-Distill-Qwen-1.5B" into "Aeon, an authentic end-to-end small language model architecture." This is the engineering work that makes Aeon viable for public release.

**Scope:** This document specifies the engineering. The IP, licensing, and "is public release the right move" decisions are made elsewhere (after Stage 2 evidence). This plan is what gets executed *in parallel with Stage 2* on a separate branch.

**Branch:** `aeon-v2` (off `main`, never touches `stage2-study` until merge).

**Owner:** Claude Code, ~6-10 weeks of focused work.

**Validation:** The Stage 1 checkpoint (trained Recursion weights from June 15 2026) must load and behave equivalently in the new code. That's the proof the rewrite is correct.

---

## Why This Matters

Right now the repo is half-Aeon and half-borrowed. `aeon/block.py` wraps `transformers.Qwen2DecoderLayer`. `aeon/model.py` subclasses `Qwen2ForCausalLM`. Class names include "R1" in them. The repo is named `-.-` for privacy. README is locked to a single sentence so as not to leak the architecture.

This was fine for Stage 1 — get a working architecture quickly, ship a real artifact in a weekend. It's not fine for public release because:

1. **Identity:** Aeon-the-public-LLM cannot be advertised as "DeepSeek-R1-Distill-Qwen-1.5B + a recurrent path." It has to be its own thing with its own name and its own implementation.

2. **Stability:** Every time `transformers` library updates Qwen2's internals, Aeon breaks. We already burned hours debugging the eager-vs-sdpa attention mismatch, the bf16 cusolver issues, the `gamma` rename shim. Owning the transformer layer eliminates this entire failure class.

3. **Speed:** The current per-token loop is ~20-30x slower than vanilla transformer inference. No public model with that performance tax gets adopted. The architectural refactor to batched-attention-with-sequentialized-recursion-state lives in this cleanup.

4. **Future scaling:** Aeon-7B-from-scratch (Stage 3) requires owning the transformer layer — you can't train from random init with someone else's pretrained-weights-shaped wrapper code.

5. **IP boundary:** Publishing weights requires publishing modeling code. Owning the modeling code means the code you publish is yours, line by line, without "wraps Qwen2DecoderLayer" appearing in any public file.

---

## Five Layers of Work

### Layer 1 — Surface and Naming Cleanup

**Goal:** Remove all R1/Qwen/DeepSeek language from Aeon's public surface. Internal modules can mention them in implementation comments, but no class name, file name, configuration key, or user-facing string should reference the origin model.

**Specific changes:**

- `AeonR1ForCausalLM` → `AeonForCausalLM` (and propagate everywhere)
- `AeonR1Config` (if any) → `AeonConfig` (already named this — verify)
- `model_type = "aeon_r1"` → `model_type = "aeon"` in config
- Class docstrings rewritten to describe the Aeon architecture without origin attribution
- Remove all `r1` from variable names (e.g., `r1_block` → `transformer_block` or similar)
- `from_r1.py` → `init_from_pretrained.py` (still functional but reframed as "warm-start from any compatible pretrained transformer" rather than "R1-specific")
- Repository description in `pyproject.toml` updated (still terse, but neutral)
- README.md rewritten for public audience (architecture overview at appropriate detail level, install, usage, citation block)
- Add `LICENSE` file (license selection deferred but file structure ready)
- Add `CITATION.cff` for academic citation
- Add `MODEL_CARD.md` template (HuggingFace standard)

**Files modified:** ~15. **Files created:** 3-4. **Files deleted:** 0.

**Time estimate:** 1-2 weeks.

**Validation:** All existing tests still pass. `grep -ri "r1\|qwen\|deepseek" aeon/ tests/` should return zero results in user-facing code (implementation comments are OK to leave).

> **Layer 1 completion note (2026-06-16):** Layer 1 is complete and the README
> architecture overview is intentionally kept locked (a comment marker holds its
> eventual place; disclosure level is decided after Stage 2 + license). The grep
> above is intentionally NOT zero after Layer 1: the remaining `Qwen2*` references
> in `aeon/model.py`, `aeon/block.py`, and `aeon/config.py` are base-class/import
> *implementation*, not public surface. They are removed in **Layer 2**, which
> replaces those base classes with Aeon's own transformer. Layer 1's surface goal
> (class names, `model_type`, file names, docstrings, exports) is met; the
> residual references resolve in Layer 2, by design.

---

### Layer 2 — Own the Transformer Implementation

**Goal:** Replace `transformers.Qwen2DecoderLayer` and `Qwen2Model` and `Qwen2ForCausalLM` with Aeon's own implementations. The architecture stays compatible with Qwen2-shaped pretrained weights for warm-start purposes, but the code is Aeon's, not borrowed.

**New files:**

- `aeon/transformer.py` — `AeonDecoderLayer` (replaces `Qwen2DecoderLayer`)
- `aeon/attention.py` — `AeonAttention` with GQA, RoPE, KV cache support
- `aeon/feedforward.py` — `AeonMLP` (SwiGLU pattern matching Qwen2's MLP shape)
- `aeon/embedding.py` — `AeonEmbedding`, `AeonRMSNorm`, RoPE utilities
- `aeon/cache.py` — Aeon's own KV cache (or shim around HF's DynamicCache, but Aeon-controlled)

**Modified files:**

- `aeon/model.py` — `AeonForCausalLM` extends `nn.Module` (not `Qwen2ForCausalLM`), uses Aeon's own components
- `aeon/block.py` — `AeonBlock` wraps `AeonDecoderLayer` (not `Qwen2DecoderLayer`)
- `scripts/init_from_pretrained.py` — loads weights from any compatible transformer checkpoint (R1, Llama, Mistral) into Aeon's native architecture

**Reference implementations to study:**
- nanoGPT (Karpathy) — simplest end-to-end transformer
- Llama's reference implementation in `transformers` source
- Mistral's reference implementation
- Qwen2's implementation in `transformers/models/qwen2/modeling_qwen2.py` (study for compatibility, then write your own version)

**Critical requirements:**
- Must be numerically identical to vanilla Qwen2 forward when initialized from R1 weights with γ=0 (the Stage 0 byte-identity gate must still pass)
- Must support SDPA, eager, and flash attention paths (attention implementation is selectable via config)
- Must support fp32, bf16, fp16 dtypes cleanly
- Must support CUDA and CPU execution (and ideally MPS for Apple)
- Must support gradient checkpointing (for training larger models)

**Time estimate:** 4-6 weeks.

**Validation:** 
1. `from_pretrained` Aeon from existing R1 weights, run Stage 0 gate, confirm pass (argmax matches on all 3 prompts).
2. Load Stage 1 checkpoint into new code, run probe_ablation.py, confirm behavior matches Stage 1 results.
3. Add tests for each new module.

---

### Layer 3 — Batched Attention with Sequentialized Recursion

**Goal:** Eliminate the per-token loop's inference penalty while preserving the persistent-state semantics. This is the change that makes Aeon's inference competitive with vanilla transformers.

**Architecture change:**

Current (slow):
```python
for t in range(T):
    h_t = embedding(token_t)
    for layer in self.layers:
        h_t = layer(h_t, recursion_state)
        write_t += recursion_write(h_t)
    recursion_state = recursion.step(write_t, recursion_state)
```

New (fast):
```python
H = embedding(tokens)  # full sequence
for layer in self.layers:
    H = layer(H, attention_mask)  # batched attention over full sequence
    writes_per_token = recursion_write_per_token(H)

# Recursion state advances sequentially, but only this small loop
for t in range(T):
    recursion_state = recursion.step(writes_per_token[t], recursion_state)
```

**Semantic implications:**

In the old loop, recursion state at token t influenced attention computation for token t (via the residual shift before the block). In the new design, recursion state advances *after* the full block stack completes, so it influences only the *next token*, not the current one's attention.

This is a real semantic change. It needs validation that the Stage 1 results still hold. Hypothesis: at γ ~ 0.03 the recursion's influence is small enough that this change doesn't materially affect training dynamics. Hypothesis must be tested.

**Implementation steps:**

1. Build the new batched forward in `aeon/model.py`
2. Re-train Stage 1 with the new architecture for ~500 steps as a smoke test
3. Compare γ trajectory, certificate behavior, loss curve against original Stage 1 results
4. If results materially differ: investigate, possibly compromise (e.g., partial batching with shorter per-token-loops every K tokens)
5. If results match within noise: commit, this is the new default

**Expected speedup:** 20-30x for inference, similar for training.

**Time estimate:** 2-4 weeks (including validation).

**Validation:** 
1. Inference benchmark: tokens/sec on a 3090 should be 100+ instead of 5-15
2. Stage 1 re-training shows equivalent γ plateau and certificate stability
3. ABLation probe results comparable to Stage 1 (qualitative if not quantitative match)

**Critical caveat:** This may not work cleanly. The per-token loop was a deliberate design choice to let recursion state modulate attention. If batching breaks that and degrades the architecture meaningfully, this layer needs a different approach (e.g., chunked-batch where recursion state advances every K tokens, K being a tunable parameter that trades speed for expressiveness).

---

### Layer 4 — Training and Inference Infrastructure

**Goal:** Make Aeon usable by people who aren't Dylan. Working install, working examples, working tests, working integration with the standard ML ecosystem.

**New files:**

- `scripts/train.py` — unified training script taking a YAML config (replaces train_stage1.py and train_stage2.py as the primary entry point; old scripts stay as historical references)
- `configs/stage1.yaml`, `configs/stage2_study.yaml`, `configs/stage3_from_scratch.yaml` — reproducible training configurations
- `aeon/data/` — dataset utilities (DataLoader, formatting, tokenization, length curriculum scheduler)
- `aeon/eval/` — eval utilities (ablation probes, perplexity, downstream task helpers)
- `aeon/utils/` — common utilities (config loading, logging, checkpointing, distributed setup helpers)
- `examples/quickstart.py` — minimal "load Aeon and chat" example
- `examples/finetune.py` — minimal fine-tuning example
- `examples/notebook.ipynb` — Jupyter walkthrough for first-time users

**Modified files:**

- `aeon/model.py` — add HuggingFace integration hooks (`AutoModelForCausalLM.register(AeonConfig, AeonForCausalLM)`)
- Existing scripts updated to call into the new `aeon.data` and `aeon.eval` modules
- Tests reorganized into `tests/unit/`, `tests/integration/`, `tests/regression/`

**HuggingFace Hub integration:**

- Aeon-v2 architecture should `push_to_hub` and `from_pretrained` work seamlessly via HF Hub
- Model card template includes architecture description, training details, intended use, limitations, ethical considerations
- License file referenced in HF Hub model card
- Examples in the model card actually run when copy-pasted

**Time estimate:** 4-6 weeks.

**Validation:** A new user (someone who isn't us) should be able to install the repo and chat with Aeon using ≤10 minutes of work from `git clone`. Quickstart example must run end-to-end on CPU.

---

### Layer 5 — Test Suite

**Goal:** Confidence that future changes don't silently break things, and reproducibility of the engineering claims.

**Test categories:**

1. **Unit tests** (`tests/unit/`):
   - Each module (attention, MLP, normalization, embedding, recursion cell) tested in isolation
   - Forward pass shape correctness
   - Gradient flow correctness
   - Numerical stability across dtypes
   - ~20-30 tests, runs in <30 sec

2. **Integration tests** (`tests/integration/`):
   - End-to-end forward through AeonForCausalLM
   - Generate API works (greedy, sampling, beam)
   - Save/load round-trip preserves all parameters
   - HuggingFace integration (push_to_hub, from_pretrained) works
   - ~10-15 tests, runs in <2 min

3. **Regression tests** (`tests/regression/`):
   - Stage 0 byte-identity gate (against vanilla Qwen2 when warm-started from R1)
   - Stage 1 checkpoint loads and produces expected gate values
   - Stage 1 probe ablation produces expected on/off divergence
   - Recursion self-test (certificate holds at init and after training)
   - ~5-10 tests, runs in <5 min

4. **Smoke tests** (`tests/smoke/`):
   - A 50-step training run completes without error
   - Inference benchmark produces tokens/sec within expected range
   - ~3-5 tests, runs in <10 min

**Test infrastructure:**

- CI configured to run unit + integration tests on every commit
- Regression + smoke tests run nightly or on tagged releases
- Pytest fixtures for common setups (tiny model, mock data, etc.)

**Time estimate:** Ongoing, but ~2 weeks of focused effort to write the initial suite.

**Validation:** All 30-50+ tests pass on the `aeon-v2` branch before merging to main.

---

## Validation Procedure (Critical)

Before `aeon-v2` is allowed to replace the current Aeon code, the following must hold:

1. **Stage 0 byte-identity gate:** AeonForCausalLM loaded from R1 weights with γ=0 must produce argmax-matching outputs on the three Stage 0 prompts. Per-token logit divergence may be present (bf16 noise) but argmax must match.

2. **Stage 1 checkpoint compatibility:** The trained Stage 1 checkpoint (gamma values from June 15 2026) must load into AeonForCausalLM (with appropriate key remapping if names changed) and produce non-zero `mean|recursion_gate|` ≈ 0.029.

3. **Stage 1 probe ablation:** Running probe_ablation.py against the Stage 1 checkpoint in AeonForCausalLM must produce ON/OFF divergence on at least 3/5 of the original 5 prompts (matching or close to original Stage 1 result).

4. **Recursion self-test:** `python aeon/recursion.py` produces atlas equivalence < 1e-4 and certificate holds at init and after 50 training steps.

5. **Inference benchmark:** AeonForCausalLM with batched attention runs at >50 tokens/sec on a 3090 (vs. current ~10 tokens/sec). Confirms the Layer 3 refactor delivered the promised speedup.

6. **Full test suite green:** All 30-50 tests pass.

If any of these fail, `aeon-v2` is not ready to merge. The pre-existing `stage2-study` branch (and `main`) remain authoritative until aeon-v2 passes all validation.

---

## What Stays the Same

To avoid scope creep, the following are *out of scope* for aeon-v2:

- The Recursion cell mathematics (`aeon/recursion.py`) — already canonical, do not modify
- The architectural thesis (bounded recurrent path with contraction certificate)
- The contraction parameterization details (Cayley + diag(tanh) + sigmoid margin scaling)
- The two-state design (r and c)
- The Margin H = 0.98, Margin C = 0.95 defaults
- The H_rec = 256 default
- The set of trainable parameters in Stage 1 (recursion cell + per-block U/D/gate)

These are the architecture. They've been validated. They don't change. This document specifies a *re-implementation around* them, not a redesign of them.

---

## Handoff Brief for Claude Code

```
Implement docs/AEON_V2_CLEANUP_PLAN.md on a new branch `aeon-v2` 
off main. Work through the five layers in order. Commit by layer 
(or by sub-section if a layer is large) with clear commit messages.

DO NOT modify the stage2-study branch. Do not modify aeon/recursion.py. 
Do not change the Recursion architecture in any way — only re-implement 
the surrounding transformer/infrastructure code as Aeon's own.

The validation procedure in section "Validation Procedure (Critical)" 
is the gate. All six items must pass before aeon-v2 can be considered 
mergeable. If any item fails, report it explicitly and stop work on 
that layer until we discuss.

I do NOT have the Stage 1 trained checkpoint in this sandbox. Where 
the validation procedure requires loading the Stage 1 checkpoint 
(items 2 and 3), implement the load path correctly but skip the 
runtime validation — that's Dylan's job after I push. Document what 
I would have run.

For Layer 3 (batched attention), if validation step 5 fails (speedup 
real but Stage 1 behavior degrades meaningfully), STOP and ping me 
before continuing. That's the riskiest layer and needs a human 
decision before committing to a design that changes architecture 
semantics.

Time estimate: 6-10 weeks of work. Expected commit cadence: ~2-3 
commits per week per layer, more during initial setup.

When done with each layer: report back with summary of what changed, 
what tests now pass, what's still open. Don't try to do all five 
layers before reporting.
```

---

## Decision Points That Are NOT in This Plan

These are decisions Dylan and I make together, separately from this implementation:

- **Final license choice** (Apache 2.0 vs AGPL vs custom community license vs dual). This document just makes sure the file structure is ready; the content of the LICENSE is decided after Stage 2 results.

- **Public release timing.** This document prepares aeon-v2 for public release but doesn't commit to release. That decision happens after Stage 2 evidence + Marty consultation.

- **Patent strategy.** Whether to file a provisional patent on the bounded-recurrent-contractive-parameterization mechanism before public release. Independent of this implementation work but should happen on a parallel track.

- **Repo rename and ownership.** Currently `AnotherSingularity/-.-`. Public release would need a real name. Independent of implementation.

- **HuggingFace organization.** For public release, weights would live under an HF org account (e.g., `horizon-tech/aeon-7b`). Set up the org account independently.

- **Whether to integrate Aeon's architecture into mainline `transformers`.** Long-term goal, requires its own engineering effort, post-public-release decision.

---

End of plan.
