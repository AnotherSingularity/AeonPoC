"""Tests for scripts/train.py helpers and the examples."""
import os, sys, runpy
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train import apply_overrides, is_recursion_param, make_optimizer, set_trainable
from aeon.config import AeonConfig
from aeon.model import AeonForCausalLM
from aeon.utils import get_logger


def _tiny():
    cfg = AeonConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                     num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                     max_position_embeddings=32, h_rec=8, tie_word_embeddings=False)
    cfg._attn_implementation = "eager"
    return AeonForCausalLM(cfg)


def test_apply_overrides():
    cfg = {"train": {"steps": 10}}
    apply_overrides(cfg, ["train.steps=100", "train.lr=0.001", "model.init=./x"])
    assert cfg["train"]["steps"] == 100
    assert cfg["train"]["lr"] == 0.001
    assert cfg["model"]["init"] == "./x"


def test_is_recursion_param():
    assert is_recursion_param("model.layers.0.recursion_gate")
    assert is_recursion_param("model.recursion.A_h")
    assert is_recursion_param("model.layers.1.U.weight")
    assert not is_recursion_param("model.layers.0.transformer_layer.self_attn.q_proj.weight")


def test_freeze_backbone_then_optimizer():
    log = get_logger("aeon.test.train")
    model = _tiny()
    set_trainable(model, "backbone", log)
    # only recursion-path params require grad
    for n, p in model.named_parameters():
        assert p.requires_grad == is_recursion_param(n)
    opt = make_optimizer(model, {"freeze": "backbone", "lr": 1e-4})
    assert len(opt.param_groups) == 1


def test_two_lr_optimizer_groups():
    log = get_logger("aeon.test.train")
    model = _tiny()
    set_trainable(model, "two_lr", log)
    opt = make_optimizer(model, {"freeze": "two_lr", "lr": 1e-4, "backbone_lr": 1e-5})
    assert len(opt.param_groups) == 2
    lrs = sorted(g["lr"] for g in opt.param_groups)
    assert lrs == [1e-5, 1e-4]


def test_finetune_example_runs():
    # end-to-end smoke of examples/finetune.py (CPU, synthetic)
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "examples", "finetune.py")
    runpy.run_path(path, run_name="__main__")
