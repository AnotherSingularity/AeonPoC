"""Smoke: inference benchmark runs and reports a positive token rate (CPU)."""
import os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from aeon.config import AeonConfig
from aeon.model import AeonForCausalLM
from scripts.bench_inference import _tiny, bench


def _toks_per_sec(model, vocab, prompt_len, new_tokens, K):
    model.config.recursion_chunk_size = K
    model.reset_recursion_state(1)
    ids = torch.randint(0, vocab, (1, prompt_len))
    t0 = time.time()
    out = model.generate(ids, max_new_tokens=new_tokens, do_sample=False, num_beams=1)
    dt = time.time() - t0
    return (out.shape[1] - prompt_len) / max(dt, 1e-6), out.shape[1]


def test_benchmark_runs_and_is_positive():
    model, vocab = _tiny()
    for K in (0, 1):
        rate, total = _toks_per_sec(model, vocab, prompt_len=24, new_tokens=8, K=K)
        assert rate > 0
        assert total == 24 + 8


def test_bench_helper_runs():
    model, vocab = _tiny()
    # the module's bench() prints a table; just confirm it runs without error
    bench(model, vocab, "cpu", prompt_len=16, new_tokens=4, chunk_sizes=[0, 1])
