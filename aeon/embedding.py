"""
aeon/embedding.py — Aeon's embedding, normalization, and rotary-position
utilities.

These are Aeon's own implementations of the standard transformer primitives.
They reproduce the arithmetic of the Qwen2-family reference (RMSNorm in fp32, the
default RoPE formulation) exactly, so that a model warm-started from
Qwen2-shaped pretrained weights is numerically identical to the reference at
gamma=0 (the byte-identity gate).
"""
import torch
import torch.nn as nn


class AeonEmbedding(nn.Embedding):
    """Token embedding table. A thin nn.Embedding so the component is named and
    owned by Aeon; behavior is the standard lookup."""
    pass


class AeonRMSNorm(nn.Module):
    """Root-mean-square layer norm, computed in fp32 then cast back."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def _head_dim(config) -> int:
    return getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads)


class AeonRotaryEmbedding(nn.Module):
    """Default rotary position embedding. Precomputes inv_freq and returns
    (cos, sin) for given position_ids, matching the reference RoPE."""

    def __init__(self, config):
        super().__init__()
        dim = _head_dim(config)
        base = getattr(config, "rope_theta", 10000.0)
        inv_freq = 1.0 / (base ** (
            torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0   # default (unscaled) RoPE

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        inv_freq = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1).to(x.device)
        pos = position_ids[:, None, :].float()
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq @ pos).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim: int = 1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
