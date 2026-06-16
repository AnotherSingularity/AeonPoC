"""aeon.data.dataset — a JSONL text dataset and a causal-LM collate fn."""
import json
import torch


class JsonlTextDataset:
    """Reads a .jsonl file with a text field per row. Keeps any precomputed
    metadata fields (e.g. n_tokens, n_turns) for the curriculum scheduler."""

    def __init__(self, path: str, text_field: str = "text"):
        self.rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if text_field in row:
                    self.rows.append(row)
        self.text_field = text_field

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]

    def texts(self):
        return [r[self.text_field] for r in self.rows]


def collate_causal(texts, tokenizer, max_len: int = 1024):
    """Tokenize a batch of strings into (input_ids, labels, attention_mask) for
    next-token cross-entropy. Padded positions get label -100."""
    enc = tokenizer(list(texts), padding=True, truncation=True,
                    max_length=max_len, return_tensors="pt")
    ids = enc.input_ids
    labels = ids.clone()
    labels[enc.attention_mask == 0] = -100
    return ids, labels, enc.attention_mask
