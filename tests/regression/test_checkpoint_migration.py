"""Regression: a v1-layout Aeon checkpoint migrates to v2 and the trained gates
survive (fabricated checkpoint — no download needed)."""
import os, sys, json, shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from aeon.config import AeonConfig
from aeon.model import AeonForCausalLM
from scripts.migrate_to_v2 import migrate, remap_key


def _tiny():
    cfg = AeonConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                     num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                     max_position_embeddings=32, h_rec=8, tie_word_embeddings=False)
    cfg._attn_implementation = "eager"
    return AeonForCausalLM(cfg)


def test_remap_key():
    assert remap_key("model.layers.0.qwen_block.self_attn.q_proj.weight") == \
        "model.layers.0.transformer_layer.self_attn.q_proj.weight"
    assert remap_key("model.layers.2.recursion_gate") == "model.layers.2.recursion_gate"


def test_v1_to_v2_migration_recovers_gates(tmp_path):
    base = str(tmp_path)
    v2, v1 = os.path.join(base, "v2"), os.path.join(base, "v1")
    m = _tiny()
    truth = {}
    with torch.no_grad():
        for i, b in enumerate(m.model.layers):
            val = 0.0293 + 0.001 * i
            b.recursion_gate.fill_(val); truth[i] = round(val, 4)
    m.save_pretrained(v2)

    # fabricate a v1-layout checkpoint: transformer_layer -> qwen_block, old config
    shutil.copytree(v2, v1)
    tensors, meta = {}, {}
    with safe_open(os.path.join(v2, "model.safetensors"), framework="pt") as f:
        meta = f.metadata() or {"format": "pt"}
        for k in f.keys():
            tensors[k.replace(".transformer_layer.", ".qwen_block.")] = f.get_tensor(k)
    save_file(tensors, os.path.join(v1, "model.safetensors"), metadata=meta)
    cfg = json.load(open(os.path.join(v1, "config.json")))
    cfg["model_type"] = "aeon_r1"; cfg["architectures"] = ["AeonR1ForCausalLM"]
    json.dump(cfg, open(os.path.join(v1, "config.json"), "w"))

    # migrate and reload
    n = migrate(v1, os.path.join(base, "v1_migrated"))
    assert n > 0
    mm = AeonForCausalLM.from_pretrained(os.path.join(base, "v1_migrated"),
                                         torch_dtype=torch.float32)
    got = [round(b.recursion_gate.item(), 4) for b in mm.model.layers]
    assert got == [truth[i] for i in range(len(got))]
    assert mm.config.model_type == "aeon"
