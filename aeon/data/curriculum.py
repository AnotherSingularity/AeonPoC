"""
aeon.data.curriculum — a deterministic, resumable length-curriculum sampler.

Phase is a function of the optimizer step (hard cutoffs at fractions of the total
budget). Within a phase, rows passing the phase's token-length filter are visited
in a shuffled order seeded by (seed, phase, epoch), so a run resumes exactly.
"""
import torch

# Default 3-phase schedule (boundaries as fractions of total steps). Phase 3 is
# capped at 1536 tokens to fit 24 GB cards (see docs/ENVIRONMENT.md).
DEFAULT_PHASES = [
    {"until_frac": 15 / 60, "min_tokens": 0,    "max_tokens": 768,  "budget": 768},
    {"until_frac": 35 / 60, "min_tokens": 512,  "max_tokens": 1536, "budget": 1536},
    {"until_frac": 1.0,     "min_tokens": 1024, "max_tokens": 1536, "budget": 1536},
]


def phase_boundaries(total_steps, phases=DEFAULT_PHASES):
    """Absolute step boundaries (end of each phase)."""
    return [round(total_steps * p["until_frac"]) for p in phases]


def phase_for_step(step, total_steps, phases=DEFAULT_PHASES):
    """0-based phase index for a given optimizer step."""
    for i, b in enumerate(phase_boundaries(total_steps, phases)):
        if step < b:
            return i
    return len(phases) - 1


class CurriculumSampler:
    def __init__(self, rows, total_steps, phases=DEFAULT_PHASES, seed: int = 42,
                 token_field: str = "n_tokens"):
        self.rows = rows
        self.total = total_steps
        self.phases = phases
        self.seed = seed
        self.token_field = token_field
        self.phase = None
        self.epoch = 0
        self.order = []
        self.pos = 0

    def budget(self, step):
        return self.phases[phase_for_step(step, self.total, self.phases)]["budget"]

    def _build_order(self, phase):
        spec = self.phases[phase]
        lo, hi = spec["min_tokens"], spec["max_tokens"]
        idx = [i for i, r in enumerate(self.rows)
               if lo <= r.get(self.token_field, 0) <= hi]
        if not idx:                      # never starve the loader
            idx = list(range(len(self.rows)))
        g = torch.Generator().manual_seed(self.seed * 100 + phase * 17 + self.epoch)
        perm = torch.randperm(len(idx), generator=g).tolist()
        self.order = [idx[p] for p in perm]
        self.pos = 0

    def next(self, step):
        ph = phase_for_step(step, self.total, self.phases)
        if ph != self.phase:             # hard phase transition -> reshuffle
            self.phase, self.epoch = ph, 0
            self._build_order(ph)
        if self.pos >= len(self.order):  # epoch boundary -> reshuffle
            self.epoch += 1
            self._build_order(self.phase)
        row = self.rows[self.order[self.pos]]
        self.pos += 1
        return row

    def state(self):
        return {"phase": self.phase, "epoch": self.epoch, "pos": self.pos}

    def load_state(self, s):
        self.phase, self.epoch = s["phase"], s["epoch"]
        if self.phase is not None:
            self._build_order(self.phase)
            self.pos = s["pos"]
