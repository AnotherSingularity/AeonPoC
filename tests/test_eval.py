"""Tests for aeon.eval (contains_key, perplexity, run_ablation)."""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aeon.config import AeonConfig
from aeon.model import AeonForCausalLM
from aeon.eval import contains_key, perplexity, run_ablation


class _Enc(dict):
    def to(self, device):
        return self
    @property
    def input_ids(self):
        return self["input_ids"]


class _FakeTok:
    eos_token_id = 0
    chat_template = None
    def __call__(self, text, return_tensors=None, truncation=False, max_length=64):
        ids = [(ord(ch) % 60) + 1 for ch in text][:max_length] or [1]
        return _Enc(input_ids=torch.tensor([ids]))
    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(i) % 60 + 65) for i in ids)
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return " ".join(m["content"] for m in messages)


def _tiny():
    cfg = AeonConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
                     num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                     max_position_embeddings=64, h_rec=8, tie_word_embeddings=False)
    cfg._attn_implementation = "eager"
    return AeonForCausalLM(cfg).eval()


def test_contains_key():
    assert contains_key("your friend Sarah", ["sarah"])
    assert not contains_key("no idea", ["sarah"])


def test_perplexity_is_finite():
    model, tok = _tiny(), _FakeTok()
    ppl = perplexity(model, tok, ["hello world", "another line"], device="cpu", max_len=32)
    assert ppl > 0 and ppl < float("inf")


def test_run_ablation_structure():
    model, tok = _tiny(), _FakeTok()
    probes = [
        {"id": "p1", "turns": ["I like 7.", "Say hi.", "What number?"], "answer_keys": ["7"]},
        {"id": "p2", "turns": ["Cats.", "Hello.", "What animal?"], "answer_keys": ["cat"]},
    ]
    res = run_ablation(model, tok, probes, device="cpu", seeds=(0,),
                       temperature=0.7, max_new_tokens=4)
    assert res["n_probes"] == 2
    assert 0 <= res["bar2_score"] <= 2
    assert len(res["per_probe"]) == 2
    assert set(res["per_probe"][0]) >= {"id", "on_hits", "off_hits", "on_correct"}
