"""
aeon/transformer.py — AeonDecoderLayer.

One pre-norm transformer block: RMSNorm -> attention -> residual -> RMSNorm ->
MLP -> residual. Submodule names (self_attn, mlp, input_layernorm,
post_attention_layernorm, and the projections within them) match the
Qwen2-family layout, so weights from a compatible pretrained checkpoint load by
name and the block is numerically identical to the reference.
"""
import torch.nn as nn

from .attention import AeonAttention
from .feedforward import AeonMLP
from .embedding import AeonRMSNorm


class AeonDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = AeonAttention(config, layer_idx)
        self.mlp = AeonMLP(config)
        self.input_layernorm = AeonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = AeonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_value=None, cache_position=None, use_cache=False, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return (hidden_states,)
