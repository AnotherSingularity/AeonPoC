# Aeon

A small, efficient language model project.

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

## Aeon inside R.U.S.E.

`aeon/ruse` is a deception strategy game played on a multiplex map of
institutional power, with Aeon seated as a faction. Scripted opponents, an
order language, and a transcript exporter for Stage 2 data are included.
See `docs/RUSE_BOARD.md`.

```bash
python scripts/play_ruse.py --p1 heuristic --p2 random -v
python scripts/play_ruse.py --p1 heuristic --p2 aeon --ckpt ./aeon_stage1 --seat2 INDUSTRIAL
```

## Status

In active development. Not yet released.

## License

See LICENSE.
