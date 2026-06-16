"""Integration: generate() across decode modes and batch sizes."""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from aeon.config import AeonConfig
from aeon.model import AeonForCausalLM


def _model():
    cfg = AeonConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                     num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                     max_position_embeddings=128, h_rec=16, tie_word_embeddings=False)
    cfg._attn_implementation = "eager"
    return cfg, AeonForCausalLM(cfg).eval()


def test_greedy():
    cfg, model = _model()
    model.reset_recursion_state(1)
    ids = torch.randint(0, cfg.vocab_size, (1, 5))
    out = model.generate(ids, max_new_tokens=8, do_sample=False, num_beams=1)
    assert out.shape == (1, 13)


def test_sampling_deterministic_with_seed():
    cfg, model = _model()
    ids = torch.randint(0, cfg.vocab_size, (1, 5))
    outs = []
    for _ in range(2):
        torch.manual_seed(123)
        model.reset_recursion_state(1)
        outs.append(model.generate(ids, max_new_tokens=8, do_sample=True,
                                    temperature=0.8, top_p=0.9, num_beams=1))
    assert torch.equal(outs[0], outs[1])   # same seed -> same draws


def test_batched_generation():
    cfg, model = _model()
    model.reset_recursion_state(2)
    ids = torch.randint(0, cfg.vocab_size, (2, 4))
    out = model.generate(ids, max_new_tokens=6, do_sample=False, num_beams=1)
    assert out.shape == (2, 10)


def test_generate_respects_chunk_default_then_decode():
    # default K=T: prefill is one batched chunk, decode steps are single-token.
    cfg, model = _model()
    assert model.config.recursion_chunk_size == 0
    model.reset_recursion_state(1)
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    out = model.generate(ids, max_new_tokens=4, do_sample=False)
    assert out.shape[1] == 10
    # recursion state advanced during decode
    r, _ = model.model.get_recursion_state()
    assert r is not None
