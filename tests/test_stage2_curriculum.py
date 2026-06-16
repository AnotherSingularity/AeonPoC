"""Smoke test for the Stage 2 curriculum scheduler."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_stage2 import (
    phase_boundaries, phase_for_step, PHASE_FILTERS, CurriculumSampler)


def test_boundaries_scale_with_steps():
    assert phase_boundaries(30000) == (7500, 17500)   # Option Y
    assert phase_boundaries(60000) == (15000, 35000)  # full study


def test_phase_for_step():
    b1, b2 = phase_boundaries(30000)
    assert phase_for_step(0, 30000) == 1
    assert phase_for_step(b1 - 1, 30000) == 1
    assert phase_for_step(b1, 30000) == 2
    assert phase_for_step(b2 - 1, 30000) == 2
    assert phase_for_step(b2, 30000) == 3
    assert phase_for_step(29999, 30000) == 3


def _rows():
    return [{"n_tokens": n} for n in
            (250, 600, 700, 768, 900, 1100, 1500, 1536, 2000, 3000, 4096)]


def test_filters_respect_phase_token_window():
    s = CurriculumSampler(_rows(), 30000, seed=42)
    for n in (s.next(0)["n_tokens"] for _ in range(30)):
        assert PHASE_FILTERS[1][0] <= n <= PHASE_FILTERS[1][1]
    s = CurriculumSampler(_rows(), 30000, seed=42)
    for n in (s.next(8000)["n_tokens"] for _ in range(30)):     # phase 2
        assert PHASE_FILTERS[2][0] <= n <= PHASE_FILTERS[2][1]
    s = CurriculumSampler(_rows(), 30000, seed=42)
    for n in (s.next(20000)["n_tokens"] for _ in range(30)):    # phase 3
        assert PHASE_FILTERS[3][0] <= n <= PHASE_FILTERS[3][1]


def test_sampler_is_deterministic_for_a_seed():
    a = CurriculumSampler(_rows(), 30000, seed=7)
    b = CurriculumSampler(_rows(), 30000, seed=7)
    seq_a = [a.next(i)["n_tokens"] for i in range(50)]
    seq_b = [b.next(i)["n_tokens"] for i in range(50)]
    assert seq_a == seq_b


def test_resume_midstream_matches_uninterrupted():
    steps = list(range(40))
    base = CurriculumSampler(_rows(), 30000, seed=7)
    uninterrupted = [base.next(i)["n_tokens"] for i in steps]

    s = CurriculumSampler(_rows(), 30000, seed=7)
    out = []
    for i in steps:
        if i == 25:                       # simulate a crash + resume here
            saved = s.state()
            s = CurriculumSampler(_rows(), 30000, seed=7)
            s.load_state(saved)
        out.append(s.next(i)["n_tokens"])
    assert out == uninterrupted
