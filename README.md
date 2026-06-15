# Aeon

Aeon is a small, efficient language model project. It pairs a standard
transformer backbone with a lightweight recurrent path that carries a compact
persistent state across a conversation. The goal is a compact model you can
train and run on a single machine.

## Install

```bash
pip install -e .
```

For development tools and tests:

```bash
pip install -e ".[dev]"
```

## Tests

```bash
pytest -q
```

## Project layout

- `aeon/` — the model package (configuration, blocks, the recurrent cell, and
  the full model).
- `scripts/` — utilities for initialization, verification, training, and an
  interactive chat loop.
- `tests/` — self-tests and shape/gradient checks.

## Status

Early-stage. The recurrent path is designed to stay bounded and stable during
training, and the model is built so it can be initialized from a known-good
backbone and improved from there.
