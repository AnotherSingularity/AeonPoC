"""Tests for aeon.data (formatting, dataset, curriculum)."""
import os, sys, json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aeon.data import (format_alpaca, format_chat, JsonlTextDataset,
                       collate_causal, CurriculumSampler, phase_boundaries,
                       phase_for_step)


class _FakeTok:
    pad_token_id = 0
    def __call__(self, texts, padding=True, truncation=True, max_length=64, return_tensors=None):
        if isinstance(texts, str):
            texts = [texts]
        seqs = [[(ord(ch) % 50) + 1 for ch in t][:max_length] or [1] for t in texts]
        n = max(len(s) for s in seqs)
        ids = torch.zeros(len(seqs), n, dtype=torch.long)
        mask = torch.zeros(len(seqs), n, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s); mask[i, :len(s)] = 1
        class E:
            pass
        e = E(); e.input_ids = ids; e.attention_mask = mask
        return e


def test_format_alpaca():
    assert "### Input" in format_alpaca({"instruction": "x", "input": "y", "output": "z"})
    assert "### Input" not in format_alpaca({"instruction": "x", "output": "z"})


def test_format_chat_fallback():
    assert format_chat([{"role": "user", "content": "hi"}]) == "user: hi"


def test_jsonl_dataset(tmp_path):
    p = tmp_path / "d.jsonl"
    with open(p, "w") as f:
        f.write(json.dumps({"text": "a", "n_tokens": 3}) + "\n")
        f.write("\n")
        f.write(json.dumps({"text": "b", "n_tokens": 5}) + "\n")
        f.write(json.dumps({"nope": 1}) + "\n")   # no text field -> skipped
    ds = JsonlTextDataset(str(p))
    assert len(ds) == 2
    assert ds.texts() == ["a", "b"]


def test_collate_causal_masks_padding():
    ids, labels, mask = collate_causal(["abc", "abcdef"], _FakeTok(), max_len=16)
    assert ids.shape == labels.shape == mask.shape
    # padded label positions are -100
    assert (labels[mask == 0] == -100).all()


def test_curriculum_boundaries_and_filters():
    assert phase_boundaries(30000) == [7500, 17500, 30000]
    rows = [{"n_tokens": n} for n in (300, 700, 900, 1200, 1500, 3000)]
    s = CurriculumSampler(rows, 30000, seed=7)
    for n in (s.next(0)["n_tokens"] for _ in range(20)):
        assert n <= 768
    s = CurriculumSampler(rows, 30000, seed=7)
    for n in (s.next(20000)["n_tokens"] for _ in range(20)):    # phase 3: 1024-1536
        assert 1024 <= n <= 1536


def test_curriculum_resume_matches():
    rows = [{"n_tokens": n} for n in (300, 700, 900, 1300, 1500, 2200)]
    base = CurriculumSampler(rows, 20, seed=11)
    full = [base.next(i)["n_tokens"] for i in range(15)]
    s = CurriculumSampler(rows, 20, seed=11)
    first = [s.next(i)["n_tokens"] for i in range(8)]
    s2 = CurriculumSampler(rows, 20, seed=11)
    s2.load_state(s.state())
    rest = [s2.next(i)["n_tokens"] for i in range(8, 15)]
    assert first + rest == full
