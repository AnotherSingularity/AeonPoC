"""
aeon/attention.py — AeonAttention.

Grouped-query attention with rotary position embeddings and KV-cache support.
Reproduces the Qwen2-family attention arithmetic (biased q/k/v projections,
unbiased output projection, head_dim**-0.5 scaling, fp32 softmax in the eager
path) so a warm-started model is numerically identical to the reference.

Selectable implementation via config._attn_implementation: "eager" or "sdpa"
(a "flash_attention_2" request falls back to sdpa with a note — the flash path
is a drop-in for later and not required for correctness).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import apply_rotary_pos_emb, _head_dim


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, n_kv, T, D) -> (B, n_kv * n_rep, T, D)."""
    b, n_kv, t, d = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(b, n_kv, n_rep, t, d)
    return hidden_states.reshape(b, n_kv * n_rep, t, d)


class AeonAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = _head_dim(config)
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def _impl(self):
        impl = getattr(self.config, "_attn_implementation", "eager")
        return "sdpa" if impl in ("sdpa", "flash_attention_2") else "eager"

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_value=None, cache_position=None, use_cache=False, **kwargs):
        B, T, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            k, v = past_key_value.update(k, v, self.layer_idx,
                                         {"cache_position": cache_position})

        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)
        kv_len = k.shape[-2]

        # attention_mask, when present, is an additive (B, 1, T, kv_len) mask.
        mask = attention_mask[..., :kv_len] if attention_mask is not None else None

        if self._impl() == "sdpa":
            # No mask + a single query (the per-token loop) attends to all cached
            # keys, which is exactly causal. For a multi-token prefill with no
            # mask we ask SDPA to apply causal masking itself.
            is_causal = mask is None and T > 1
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                scale=self.scaling, is_causal=is_causal)
        else:  # eager
            attn = torch.matmul(q, k.transpose(2, 3)) * self.scaling
            if mask is not None:
                attn = attn + mask
            elif T > 1:
                # full-sequence prefill in eager mode needs an explicit causal mask
                offset = kv_len - T
                causal = torch.full((T, kv_len), float("-inf"),
                                    device=q.device, dtype=attn.dtype)
                causal = torch.triu(causal, diagonal=1 + offset)
                attn = attn + causal
            attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
            attn = F.dropout(attn, p=self.attention_dropout, training=self.training)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)
