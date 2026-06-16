# Aeon

A small, efficient language model project.

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

## Tests

```bash
pytest -q
```

## Status

In active development. Not yet released.

## License

See LICENSE.
