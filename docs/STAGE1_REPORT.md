# Stage 1 Report

Internal record of the Stage 1 recursion warm-up run. Technical detail is fine
here; this file is repo-internal, not public-facing.

## Summary

Stage 1 trained the recursion path (Cayley-D cell params + per-block `U`,
`D_proj`, `gamma`) on top of a frozen R1-Distill-Qwen-1.5B backbone for 2000
steps. The contraction certificate held at every audit. Per-block gates lifted
off zero to a small plateau, as expected for Stage 1.

## Hardware & cost

| | |
|---|---|
| GPU | 1× NVIDIA RTX 3090, 24 GB (Vast.ai) |
| Price | $0.143 / hr |
| Wall time | ~24 h (2000 steps) |
| Total cost | ~$5 including debugging |

## Training config

| param | value |
|---|---|
| data | `tatsu-lab/alpaca` (chat-template formatted) |
| batch_size | 1 |
| seq_len | 512 |
| lr | 1e-4 |
| steps | 2000 |
| optimizer | AdamW, grad-clip 1.0 |
| trainable | recursion cell + per-block `U` / `D_proj` / `gamma` (backbone frozen) |
| dtype | bfloat16 |

Model config (from the Stage 1 `config.json`): `num_hidden_layers=28`,
`hidden_size=1536`, `h_rec=256`, `margin_h=0.98`, `margin_c=0.95`,
`tie_word_embeddings=false`, `vocab_size=151936`.

## Stage 0 gate result

Run with the patched `verify_stage0.py` (both models forced to `sdpa`):

```
argmax_match = True on all 3 prompts
worst max|dlogit| = 0.2812
```

Argmax matches everywhere; the absolute logit gap is the per-token-loop bf16
noise floor (see "Architectural note"). Gate passes on the argmax criterion.

## Stage 1 final metrics

| metric | value | buildbook expectation (§14) |
|---|---|---|
| mean \|gamma\| plateau | ~0.0293 | 0.05–0.20 |
| sigma(Wh) | ~0.754 | < MARGIN_H = 0.98 ✓ |
| sigma(Wc) | ~0.665 | < MARGIN_C = 0.95 ✓ |
| certificate (`chart_A_holds`) | True through all 40 audits | True at every checkpoint ✓ |
| training loss | dropped from R1 baseline | ~2.0 → 1.5–1.7 |

Note: the gamma plateau (~0.029) came in **below** the buildbook's anticipated
0.05–0.20 band. That is consistent with Stage 1 being a gentle warm-up at a
frozen backbone — the gates only need to be slightly nonzero to start
contributing — but it is also the kind of signal that says "the recursion is
contributing only weakly so far." Stage 2 (unfrozen backbone) is where the gates
are expected to grow into a load-bearing role.

## Bugs encountered and fixed

1. **`cayley()` bf16 on CUDA.** `torch.linalg.solve` (cuSOLVER `lu_factor`)
   supports only fp32/fp64. Fix: cast the skew-symmetric system to fp32 for the
   solve, cast the orthogonal result back to the original dtype. Same matrix,
   same orthogonality, precision swap is invisible to the forward pass.

2. **`sigma_max()` bf16 on CUDA.** Same cuSOLVER limitation for
   `torch.linalg.svdvals`. Fix: cast the (detached) input to fp32 before the
   SVD. This path is audit-only/read-only, so the cast is free.

3. **`_attn_implementation` eager/sdpa mismatch.** `AeonConfig(**r1.config.to_dict())`
   does not carry `_attn_implementation` (HF sets it during `from_pretrained`),
   so Aeon defaulted to `eager` while R1 defaulted to `sdpa`. The two kernels
   round differently in bf16, producing a ~0.27 max|dlogit| gap with *correct
   argmax*. Fix: copy `r1.config._attn_implementation` onto the Aeon config when
   porting, and load both models with `attn_implementation="sdpa"` in the gate.

4. **Stage 0 tolerance recalibration.** The buildbook's `tol=5e-3` assumed a
   batched forward. Aeon's per-token loop has an intrinsic bf16 noise floor of
   ~0.2–0.3 vs batched attention (reproducible even on vanilla R1 with no Aeon
   code in the path). The meaningful invariant at gamma=0 is **argmax match**,
   not raw logit distance. Gate updated to `tol=5e-1` **and** an argmax
   assertion across all prompts.

### Also discovered (now fixed) — gate serialization

The per-block gate is a parameter literally named `gamma`. transformers'
`save_pretrained`/`from_pretrained` apply a legacy key rewrite (`gamma`→`weight`,
`beta`→`bias`, from old LayerNorm naming). Observed directly in the Stage 0 load
warnings: the checkpoint carried `model.layers.N.weight` while the model expected
`model.layers.N.gamma`, so the gate was **not** loaded and fell back to its
zero-initialized value.

- At init (gate=0) this is harmless — which is why Stage 0 still passed.
- For a **trained** checkpoint it means the learned gates (~0.029) are written to
  the file under the `weight` key but do **not** load back into the `gamma`
  parameter. `probe_ablation.py` prints `mean|gamma|` on load specifically to
  surface this; if it reads ~0, the gates didn't survive reload.

The trained values are not destroyed — they live in the checkpoint under
`model.layers.N.weight`.

**Resolution.** Two changes:
1. The gate parameter was renamed `gamma` → `recursion_gate` (matches no HF
   shim), so all checkpoints saved by current code round-trip correctly.
2. `scripts/fix_gate_keys.py` recovers a pre-rename checkpoint by remapping the
   stray `model.layers.N.weight` keys to `model.layers.N.recursion_gate`. The
   Stage 1 checkpoint must be passed through it once before it will load with
   current code (or run in `probe_ablation.py` / Stage 2).

Note the remap target is `recursion_gate`, **not** `gamma`: the HF shim also
fires on *load*, so a recovered `.gamma` key would be rewritten back to
`.weight` and still fail to match. Only a shim-immune name round-trips.

## Architectural note

The per-token recursion loop runs attention one token at a time (threaded KV
cache) instead of one batched TxT call. Mathematically identical — verified
byte-identical at gamma=0 in **fp32** (`scripts/verify_wiring.py`, ~1e-6) — but
in **bf16** the per-token kernel path diverges from the batched path by ~0.2–0.3
in logit space. This is a property of the execution order, not a wiring bug.
Aeon's design absorbs this: the contraction certificate bounds the recurrent
state regardless, so the noise cannot compound into instability. The Stage 0
gate was recalibrated to argmax-equivalence accordingly.

## Checkpoint location

Stage 1 checkpoint lives on Google Drive at `aeon_runs/stage1_20260615/`.
It is **not** committed or uploaded anywhere in this repo by design (weights are
large and live out-of-band). Point `--ckpt` at a local copy when running
`probe_ablation.py` or Stage 2.

## Reproduction

```bash
pip install -e ".[dev,data]"
python scripts/from_r1.py --out ./aeon_init
python scripts/verify_stage0.py --aeon ./aeon_init        # argmax gate
python scripts/prepare_alpaca.py --tokenizer ./aeon_init --out ./alpaca.jsonl
python scripts/train_stage1.py --init ./aeon_init --data ./alpaca.jsonl \
    --out ./aeon_stage1 --batch_size 1 --seq_len 512 --lr 1e-4 --steps 2000
```
