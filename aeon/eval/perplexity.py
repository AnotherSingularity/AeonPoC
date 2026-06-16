"""aeon.eval.perplexity — token-level perplexity over a set of texts."""
import math
import torch


@torch.no_grad()
def perplexity(model, tok, texts, device=None, max_len: int = 1024):
    """Mean per-token perplexity over `texts` (recursion state reset per text)."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for text in texts:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len)
        ids = enc.input_ids.to(device)
        if ids.shape[1] < 2:
            continue
        if hasattr(model, "reset_recursion_state"):
            model.reset_recursion_state(batch_size=1)
        logits = model(input_ids=ids).logits
        shift_logits = logits[:, :-1, :].float()
        shift_labels = ids[:, 1:]
        nll = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1), reduction="sum")
        total_nll += nll.item()
        total_tokens += shift_labels.numel()
    if total_tokens == 0:
        return float("nan")
    return math.exp(total_nll / total_tokens)
