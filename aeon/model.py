"""
aeon/model.py — AeonR1ForCausalLM.

Subclasses Qwen2ForCausalLM. Overrides the forward pass to:
  - Replace Qwen2DecoderLayer with AeonBlock at each layer
  - Maintain a global recursion state (r, c) that persists across tokens
  - After each token's full block stack, run one Recursion step to
    update (r, c) from the aggregated per-block writes

Critical design point: r_t is read at the START of token t, BEFORE any
block runs. The state update happens AFTER the full L-block stack
finishes for token t. This means token t+1 reads the state that
incorporates token t's contribution.

Generation: state persists across tokens within a generation call AND
across calls (the model exposes get_recursion_state / set_recursion_state).
This is how the model carries continuity across chat turns.

WIRING NOTES (deviations from the handoff sketch, required for the
Stage 0 byte-identity gate to hold; see comments inline):

  (1) KV CACHE THREADING. The per-token loop runs the full block stack on
      one token at a time. For attention to be causal across the sequence
      (i.e. token t attends to tokens 0..t-1), the per-layer key/value of
      earlier tokens must be available when token t is processed. We thread
      a single Cache object through the whole loop so each layer accumulates
      its K/V left-to-right. Without this, every token would attend only to
      itself and the forward would NOT match vanilla Qwen2 -> Stage 0 fails.

  (2) LM HEAD RE-TIE. Qwen2-1.5B / R1-Distill tie lm_head to embed_tokens.
      Qwen2ForCausalLM ties them in __init__, but we then replace self.model
      with AeonModel, which builds a fresh embed_tokens. We must re-tie so
      lm_head follows the new embeddings; otherwise lm_head stays attached to
      the discarded random embeddings -> Stage 0 fails.
"""
import torch
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2Model
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import DynamicCache

from .config import AeonConfig
from .block import AeonBlock
from .recursion import RecursionChartB, audit_certificates


class AeonModel(Qwen2Model):
    """The bare transformer with Aeon blocks and the recursion cell."""

    config_class = AeonConfig

    def __init__(self, config: AeonConfig):
        super().__init__(config)
        # Replace each Qwen2DecoderLayer with an AeonBlock
        self.layers = nn.ModuleList([
            AeonBlock(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        # Global recursion cell
        # in_dim = H_rec because we feed it the aggregated write W_total (H_rec-wide)
        self.recursion = RecursionChartB(
            in_dim=config.h_rec,
            H=config.h_rec,
            margin_H=config.margin_h,
            margin_C=config.margin_c,
        )
        # Optional learned initial states
        if config.recursion_init_learnable:
            self.r_init = nn.Parameter(torch.zeros(config.h_rec))
            self.c_init = nn.Parameter(torch.zeros(config.h_rec))
        else:
            self.register_buffer("r_init", torch.zeros(config.h_rec))
            self.register_buffer("c_init", torch.zeros(config.h_rec))

        # Persistent state across calls (one per active batch).
        self._persistent_r = None    # (B, H_rec) or None
        self._persistent_c = None    # (B, H_rec) or None

        # Toggle: True means inject recursion. False means run as plain Qwen2
        # (used for Stage 0 verification and ablation experiments).
        self.recursion_enabled = True

    # ---- state management (chat persistence) ------------------------------
    @torch.no_grad()
    def reset_recursion_state(self, batch_size: int = 1):
        device = next(self.parameters()).device
        self._persistent_r = self.r_init.to(device).expand(batch_size, -1).clone()
        self._persistent_c = self.c_init.to(device).expand(batch_size, -1).clone()

    def get_recursion_state(self):
        return (None if self._persistent_r is None else self._persistent_r.clone(),
                None if self._persistent_c is None else self._persistent_c.clone())

    def set_recursion_state(self, r, c):
        self._persistent_r = r
        self._persistent_c = c

    # ---- forward --------------------------------------------------------
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        **kwargs,
    ):
        # Embedding
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        B, T, D = hidden_states.shape
        device = hidden_states.device

        # Initialize / fetch persistent recursion state
        if self._persistent_r is None or self._persistent_r.shape[0] != B:
            self.reset_recursion_state(batch_size=B)
        r = self._persistent_r.to(device)
        c = self._persistent_c.to(device)

        # KV cache that threads through the per-token loop so attention is
        # causal across the whole sequence (wiring note (1)). We always use a
        # Cache internally; we only return it to the caller when use_cache.
        if past_key_values is None:
            past_key_values = DynamicCache()
        past_seen = past_key_values.get_seq_length()

        if cache_position is None:
            cache_position = torch.arange(past_seen, past_seen + T, device=device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # RoPE position embeddings (Qwen2 computes them at model level since v4.43+)
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        # Additive key-padding mask, built once over the full KV length.
        # Causality itself needs no mask: in left-to-right processing the cache
        # only ever holds keys at positions <= the current query, so attending
        # to "all cached keys" IS causal attention. We only need to mask padded
        # keys. For unpadded inputs (the Stage 0 prompts) this is all zeros and
        # equals passing None.
        add_mask = None
        if attention_mask is not None and (attention_mask == 0).any():
            min_val = torch.finfo(hidden_states.dtype).min
            am = attention_mask.to(device)
            add_mask = torch.zeros(B, 1, 1, am.shape[-1],
                                   dtype=hidden_states.dtype, device=device)
            add_mask = add_mask.masked_fill(am[:, None, None, :] == 0, min_val)

        # Process tokens. The recursion state advances ONCE per token, after
        # the full L-block stack has produced all per-block writes for that
        # token.
        outputs_hidden = []
        n_layers = len(self.layers)
        for t in range(T):
            h_t = hidden_states[:, t:t + 1, :]                       # (B, 1, D)
            pe_t = (cos[:, t:t + 1], sin[:, t:t + 1])
            cp_t = cache_position[t:t + 1]
            pid_t = position_ids[:, t:t + 1]
            kv_len = past_seen + t + 1
            mask_t = add_mask[:, :, :, :kv_len] if add_mask is not None else None

            W_sum = torch.zeros(B, self.config.h_rec, device=device, dtype=h_t.dtype)

            for block in self.layers:
                block_out = block(
                    h_t,
                    r_t=r if self.recursion_enabled else torch.zeros_like(r),
                    attention_mask=mask_t,
                    position_ids=pid_t,
                    past_key_value=past_key_values,
                    use_cache=True,
                    cache_position=cp_t,
                    position_embeddings=pe_t,
                )
                h_t, w_l = block_out[0], block_out[1]
                W_sum = W_sum + w_l[:, 0, :]

            # Update global recursion state with the aggregated write
            if self.recursion_enabled:
                W_total = W_sum / n_layers                            # (B, H_rec)
                r, c = self.recursion.step(W_total, r, c)
            # else: leave r, c unchanged (each block was given zero r)

            outputs_hidden.append(h_t)

        hidden_states = torch.cat(outputs_hidden, dim=1)            # (B, T, D)
        hidden_states = self.norm(hidden_states)

        # Persist updated state for next call
        self._persistent_r = r.detach()
        self._persistent_c = c.detach()

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=None,
            attentions=None,
        )


class AeonR1ForCausalLM(Qwen2ForCausalLM):
    config_class = AeonConfig

    def __init__(self, config: AeonConfig):
        # Call parent so we set up lm_head etc.
        super().__init__(config)
        # Replace self.model with our AeonModel
        self.model = AeonModel(config)
        # Re-tie lm_head to the NEW embeddings (wiring note (2)).
        self.post_init()
        self.tie_weights()

    # convenience
    def reset_recursion_state(self, batch_size: int = 1):
        self.model.reset_recursion_state(batch_size)

    def disable_recursion(self):
        """Stage 0 / ablation: run as plain Qwen2."""
        self.model.recursion_enabled = False

    def enable_recursion(self):
        self.model.recursion_enabled = True

    def audit(self):
        return audit_certificates(self.model.recursion)
