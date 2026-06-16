"""Aeon is registered with the transformers Auto* factories (Layer 4)."""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aeon
from aeon.config import AeonConfig
from aeon.model import AeonForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM


def _cfg():
    return AeonConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=32, h_rec=8)


def test_automodel_from_config_returns_aeon():
    model = AutoModelForCausalLM.from_config(_cfg())
    assert isinstance(model, AeonForCausalLM)


def test_registration_is_idempotent():
    aeon.register_for_auto_classes()
    aeon.register_for_auto_classes()  # must not raise


def test_autoconfig_roundtrip(tmp_path):
    cfg = _cfg()
    d = str(tmp_path / "cfg")
    cfg.save_pretrained(d)
    loaded = AutoConfig.from_pretrained(d)
    assert loaded.model_type == "aeon"
    assert loaded.h_rec == 8


def test_auto_map_emitted_on_save(tmp_path):
    # register_for_auto_class tags the config so save emits an auto_map (Hub
    # remote-code loading).
    model = AeonForCausalLM(_cfg())
    d = str(tmp_path / "m")
    model.save_pretrained(d)
    import json
    with open(os.path.join(d, "config.json")) as f:
        cfg = json.load(f)
    assert "auto_map" in cfg
    assert "AutoModelForCausalLM" in cfg["auto_map"]
