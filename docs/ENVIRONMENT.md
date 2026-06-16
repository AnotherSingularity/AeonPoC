# Environment — known-good stack

Aeon is sensitive to the torch/CUDA build because the per-token training forward
holds the whole sequence's activation graph. Use the pinned stack below; do not
let `pip` pull a default-PyPI torch newer than what's pinned.

## Known-good stack

| component | version | notes |
|---|---|---|
| Python | 3.10–3.12 | |
| torch | **2.5.1** | install from the **cu124** index, NOT default PyPI |
| CUDA | 12.x (cu124 wheels) | |
| transformers | **4.46.3** | |
| GPU | RTX 3090 / 4090, 24 GB | Stage 1 and the Stage 2 plan target this card |

Install:

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev,data]"
```

`pyproject.toml` pins `torch==2.5.1` and `transformers==4.46.3`. Stage 1 (June 15
2026) ran on torch ~2.4/2.5 + CUDA 12.x — this stack reproduces it.

## The regression we hit (cu130 baseline-memory OOM)

**Symptom.** The Stage 2 smoke OOM'd on a 24 GB 3090 *before the first forward
pass*: the process sat at ~22–23 GiB at startup, and peak memory was a roughly
**fixed ~24108 MiB independent of `seq_len`** (lowering `--max_seq_len` did not
help — the tell-tale sign this is baseline memory, not activation memory).

**Cause.** `torch` 2.11+ changed the default PyPI wheel to **cu130**, and
torch 2.11/2.12 + cu130 has a documented baseline-memory regression on 24 GB
consumer cards (3090/4090) that consumes most of VRAM before any workload runs.
Nothing in Aeon's code changed — the box had been rebuilt with a newer default
torch.

**References.** PyTorch issues #175666, #182941; vLLM #42049.

**Fix.** Pin torch to 2.5.1 from the cu124 index (above). If you ever see a fixed
~24 GiB peak that ignores `seq_len`, suspect the torch/CUDA build first — not the
model or the training script.

## Related, queued (not yet implemented)

Gradient checkpointing for the per-token loop is queued as a **Layer 4**
follow-up (see `docs/AEON_V2_CLEANUP_PLAN.md`): wrap the 28-layer-per-token block
in `torch.utils.checkpoint.checkpoint(use_reentrant=False)`, thread the recurrent
state as explicit input/output (not via `self`), and keep `DynamicCache.update()`
outside the checkpoint boundary so the backward recompute does not double-append.
Reference: arXiv 2008.07027. Expected ~50–70% activation-memory reduction at
~20–30% throughput cost. This reduces *activation* memory and is orthogonal to
the cu130 *baseline*-memory regression above — pinning torch is the actual fix
for the OOM we hit.
