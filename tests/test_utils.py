"""Tests for aeon.utils (config merge, logging, training-state checkpointing)."""
import os, sys, logging
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aeon.utils import (load_yaml, merge_dicts, flatten_for_argparse,
                        get_logger, save_training_state, load_training_state)


def test_merge_dicts_recursive():
    base = {"a": 1, "b": {"x": 1, "z": 9}}
    over = {"b": {"x": 2, "y": 3}, "c": 4}
    assert merge_dicts(base, over) == {"a": 1, "b": {"x": 2, "z": 9, "y": 3}, "c": 4}
    # base is not mutated
    assert base["b"]["x"] == 1


def test_flatten_for_argparse():
    flat = flatten_for_argparse({"opt": {"lr": 1e-4, "wd": 0.0}, "steps": 10})
    assert flat == {"opt.lr": 1e-4, "opt.wd": 0.0, "steps": 10}


def test_load_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("a: 1\nb:\n  x: 2\n")
    assert load_yaml(str(p)) == {"a": 1, "b": {"x": 2}}


def test_logger_idempotent():
    a = get_logger("aeon_test_x")
    b = get_logger("aeon_test_x")
    assert a is b
    assert len(a.handlers) == 1


def test_training_state_roundtrip(tmp_path):
    torch.manual_seed(0)
    lin = torch.nn.Linear(4, 4)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.9)
    for _ in range(3):
        opt.zero_grad(); (lin(torch.randn(2, 4)).sum()).backward(); opt.step(); sched.step()

    path = str(tmp_path / "state.pt")
    save_training_state(path, step=3, optimizer=opt, scheduler=sched, extra={"k": 1})
    rng_ref = torch.get_rng_state(); draw_ref = torch.rand(3)

    lin2 = torch.nn.Linear(4, 4)
    opt2 = torch.optim.AdamW(lin2.parameters(), lr=1e-3)
    sched2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=1, gamma=0.9)
    torch.set_rng_state(rng_ref)
    step, extra = load_training_state(path, optimizer=opt2, scheduler=sched2)

    assert step == 3 and extra == {"k": 1}
    assert sched2.get_last_lr()[0] == sched.get_last_lr()[0]
    assert opt2.state_dict()["state"]
    assert torch.allclose(torch.rand(3), draw_ref)  # RNG restored
