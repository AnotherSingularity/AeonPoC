# Aeon

A small, efficient language model.

## Install

Install the pinned torch build from the cu124 index first (newer default
torch wheels are cu130 and OOM 24 GB GPUs — see `docs/ENVIRONMENT.md`):

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

For development tools and tests:

```bash
pip install -e ".[dev]"
```

## Usage

```python
from aeon import AeonForCausalLM
from transformers import AutoTokenizer

model = AeonForCausalLM.from_pretrained("<checkpoint>")
tok = AutoTokenizer.from_pretrained("<checkpoint>")

ids = tok("Hello", return_tensors="pt")
out = model.generate(**ids, max_new_tokens=64)
print(tok.decode(out[0], skip_special_tokens=True))
```

## Tests

```bash
pytest -q
```

## Status

In active development. Not yet released. See `MODEL_CARD.md` for the model card
(in progress) and `CITATION.cff` for citation metadata.

<!--
Architecture overview intentionally omitted pending a decision on public
disclosure level. See docs/AEON_V2_CLEANUP_PLAN.md (Layer 1) — the overview is
held until the detail level is confirmed.
-->

## License

See `LICENSE` (license selection pending).
