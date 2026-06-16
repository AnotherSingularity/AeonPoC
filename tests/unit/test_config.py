"""Unit tests for AeonConfig defaults and serialization."""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from aeon.config import AeonConfig


def test_defaults():
    c = AeonConfig()
    assert c.model_type == "aeon"
    assert c.h_rec == 256
    assert c.margin_h == 0.98 and c.margin_c == 0.95
    assert c.recursion_chunk_size == 0           # fully batched by default
    assert c.recursion_init_learnable is False


def test_custom_fields_roundtrip():
    c = AeonConfig(h_rec=128, margin_h=0.9, recursion_chunk_size=4,
                   vocab_size=100, hidden_size=32, num_attention_heads=4)
    d = c.to_dict()
    c2 = AeonConfig(**d)
    assert c2.h_rec == 128
    assert c2.margin_h == 0.9
    assert c2.recursion_chunk_size == 4
    assert c2.model_type == "aeon"


def test_save_load_roundtrip(tmp_path):
    c = AeonConfig(h_rec=64, recursion_chunk_size=2)
    d = str(tmp_path / "cfg")
    c.save_pretrained(d)
    c2 = AeonConfig.from_pretrained(d)
    assert c2.h_rec == 64 and c2.recursion_chunk_size == 2
