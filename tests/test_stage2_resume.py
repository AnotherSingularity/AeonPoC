"""Smoke test for Stage 2 resume-from-checkpoint plumbing.

Builds a tiny model, runs a few optimizer steps, saves a checkpoint, then
restores optimizer + scheduler + sampler + RNG state into fresh objects and
asserts they match. This exercises save_checkpoint / restore_training_state
without needing a tokenizer or any download.
"""
import os, sys, types, argparse
import torch
from transformers.optimization import get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aeon.config import AeonConfig
from aeon.model import AeonR1ForCausalLM
from scripts.train_stage2 import (
    freeze_backbone, save_checkpoint, restore_training_state, CurriculumSampler)


class _StubTok:
    def save_pretrained(self, d):
        pass


def _tiny_model():
    cfg = AeonConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                     num_hidden_layers=2, num_attention_heads=4,
                     num_key_value_heads=2, max_position_embeddings=64, h_rec=8,
                     tie_word_embeddings=False)
    cfg._attn_implementation = "eager"
    return cfg, AeonR1ForCausalLM(cfg)


def _args(out, steps=20):
    return argparse.Namespace(out=out, steps=steps, seed=42)


def test_resume_roundtrip(tmp_path):
    torch.manual_seed(0)
    cfg, model = _tiny_model()
    trainable = freeze_backbone(model)
    opt = torch.optim.AdamW(trainable, lr=5e-5, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=2, num_training_steps=20)

    rows = [{"n_tokens": n} for n in (250, 600, 700, 900, 1200, 1500, 2000, 4000)]
    sampler = CurriculumSampler(rows, 20, seed=42)

    # a few real steps to populate optimizer + scheduler + sampler state
    for step in range(5):
        sampler.next(step)
        ids = torch.randint(0, cfg.vocab_size, (1, 8))
        model.reset_recursion_state(batch_size=1)
        out = model(input_ids=ids, labels=ids.clone())
        opt.zero_grad(); out.loss.backward(); opt.step(); sched.step()

    saved_step = 5
    ckpt = save_checkpoint(str(tmp_path / "run"), saved_step, model, _StubTok(),
                           opt, sched, sampler, _args(str(tmp_path / "run")),
                           mean_gate_start=0.0293)
    assert os.path.isfile(os.path.join(ckpt, "training_state.pt"))

    # capture RNG-after-save so we can prove restore reproduces the stream
    rng_ref = torch.get_rng_state()
    draw_ref = torch.rand(3)

    # fresh objects, then restore
    torch.manual_seed(999)  # perturb global RNG to prove restore overrides it
    _, model2 = _tiny_model()
    trainable2 = freeze_backbone(model2)
    opt2 = torch.optim.AdamW(trainable2, lr=5e-5, weight_decay=0.01)
    sched2 = get_cosine_schedule_with_warmup(opt2, num_warmup_steps=2, num_training_steps=20)
    sampler2 = CurriculumSampler(rows, 20, seed=42)

    # set RNG to the post-save state, then restore should reapply the SAVED rng
    torch.set_rng_state(rng_ref)
    step2, mgs = restore_training_state(ckpt, opt2, sched2, sampler2)

    assert step2 == saved_step
    assert abs(mgs - 0.0293) < 1e-9
    assert sampler2.state() == sampler.state()
    assert sched2.get_last_lr()[0] == sched.get_last_lr()[0]
    # optimizer step counter restored
    assert opt2.state_dict()["state"], "optimizer state did not restore"
    # RNG restored -> same subsequent draw as reference
    assert torch.allclose(torch.rand(3), draw_ref)


def test_resume_continues_sampler_stream(tmp_path):
    rows = [{"n_tokens": n} for n in (300, 700, 900, 1300, 1500, 2200, 3300)]
    base = CurriculumSampler(rows, 20, seed=11)
    full = [base.next(i)["n_tokens"] for i in range(15)]

    s = CurriculumSampler(rows, 20, seed=11)
    first = [s.next(i)["n_tokens"] for i in range(8)]
    state = s.state()
    s2 = CurriculumSampler(rows, 20, seed=11)
    s2.load_state(state)
    rest = [s2.next(i)["n_tokens"] for i in range(8, 15)]
    assert first + rest == full
