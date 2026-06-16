"""
aeon/model.py — AeonModel and AeonForCausalLM.

The Aeon causal language model, built on Aeon's own transformer components
(aeon/transformer.py, attention.py, feedforward.py, embedding.py, cache.py). It
extends the transformers PreTrainedModel base for serialization/generation
infrastructure but contains none of another model's modeling code.

Every block reads from and writes to a single global recurrent state (the
contractive cell in aeon/recursion.py) that persists across tokens and across
calls, giving the model continuity a vanilla transformer does not have:

  - Each block applies a recurrent read (a gated residual shift) before
    attention and produces a recurrent write afterwards.
  - After a token's full block stack runs, one recurrent step advances the
    global state (r, c) from the aggregated per-block writes.

Critical design point: r_t is read at the START of token t, BEFORE any block
runs. The state update happens AFTER the full block stack finishes for token t,
so token t+1 reads the state that incorporates token t's contribution.

WIRING NOTE — KV CACHE THREADING. The per-token loop runs the full block stack
on one token at a time. For attention to be causal across the sequence (token t
attends to tokens 0..t-1), the per-layer key/value of earlier tokens must be
available when token t is processed, so we thread a single cache object through
the whole loop. (Layer 3 of the v2 cleanup replaces this loop with a batched
forward; the per-token loop is correct, not fast.)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .config import AeonConfig
from .block import AeonBlock
from .embedding import AeonEmbedding, AeonRMSNorm, AeonRotaryEmbedding
from .cache import AeonCache
from .recursion import RecursionChartB, audit_certificates


def _build_chunk_mask(q_abs, kv_len, attention_mask, dtype, device):
    """Additive (B, 1, K, kv_len) attention mask for a chunk of K query tokens.

    Encodes causality (query at absolute position p attends to key positions
    <= p) and key padding. Returns None when no masking is needed (a single
    query with no padding is trivially causal — the K=1 / per-token path).
    """
    K = q_abs.shape[0]
    has_pad = attention_mask is not None and (attention_mask == 0).any()
    if K == 1 and not has_pad:
        return None
    min_val = torch.finfo(dtype).min
    key_pos = torch.arange(kv_len, device=device)
    allowed = key_pos[None, :] <= q_abs[:, None]          # (K, kv_len) bool
    mask = torch.zeros(K, kv_len, dtype=dtype, device=device)
    mask = mask.masked_fill(~allowed, min_val)
    B = attention_mask.shape[0] if attention_mask is not None else 1
    mask = mask[None, None].expand(B, 1, K, kv_len).clone()
    if has_pad and attention_mask.shape[-1] >= kv_len:
        pad = attention_mask[:, :kv_len] == 0                # (B, kv_len)
        mask = mask.masked_fill(pad[:, None, None, :], min_val)
    return mask


class AeonPreTrainedModel(PreTrainedModel):
    config_class = AeonConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["AeonBlock"]
    _supports_sdpa = True
    _supports_cache_class = True
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class AeonModel(AeonPreTrainedModel):
    """The bare transformer with Aeon blocks and the recurrent cell."""

    def __init__(self, config: AeonConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = AeonEmbedding(config.vocab_size, config.hidden_size,
                                          self.padding_idx)
        self.layers = nn.ModuleList([
            AeonBlock(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = AeonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = AeonRotaryEmbedding(config)

        # Global recurrent cell. in_dim = H_rec: it consumes the aggregated,
        # H_rec-wide per-block write W_total.
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
        self._persistent_r = None
        self._persistent_c = None
        # True means inject recursion. False runs the bare transformer (gate test
        # / ablation).
        self.recursion_enabled = True
        self.gradient_checkpointing = False

        # Preserve the canonical recurrent-cell init (and the small per-block
        # U/D_proj init) from the generic _init_weights pass that post_init runs.
        self._protect_custom_init()
        self.post_init()

    def _protect_custom_init(self):
        for mod in self.recursion.modules():
            mod._is_hf_initialized = True
        for blk in self.layers:
            blk.U._is_hf_initialized = True
            blk.D_proj._is_hf_initialized = True

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

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
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        B, T, D = hidden_states.shape
        device = hidden_states.device

        # Initialize / fetch persistent recurrent state
        if self._persistent_r is None or self._persistent_r.shape[0] != B:
            self.reset_recursion_state(batch_size=B)
        r = self._persistent_r.to(device)
        c = self._persistent_c.to(device)

        # KV cache threaded through the per-token loop (see wiring note).
        if past_key_values is None:
            past_key_values = AeonCache()
        past_seen = past_key_values.get_seq_length()

        if cache_position is None:
            cache_position = torch.arange(past_seen, past_seen + T, device=device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        cos, sin = self.rotary_emb(hidden_states, position_ids)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Chunked-batch forward (Layer 3). K = chunk size:
        #   K = T  -> fully batched (default); the recurrent read for every token
        #             in the chunk is the chunk-start state (broadcast).
        #   K = 1  -> per-token, identical to the v1 loop.
        # Within a chunk, attention is batched over the chunk's tokens (causal,
        # attending to the threaded cache). The recurrent state advances once per
        # token *after* the chunk's full block stack, via a small sequential scan.
        chunk = self.config.recursion_chunk_size
        K = T if (not chunk or chunk <= 0 or chunk >= T) else chunk

        outputs_hidden = []
        n_layers = len(self.layers)
        pos = 0
        while pos < T:
            end = min(pos + K, T)
            kc = end - pos                                   # tokens in this chunk
            h_chunk = hidden_states[:, pos:end, :]           # (B, kc, D)
            pe_chunk = (cos[:, pos:end], sin[:, pos:end])
            cp_chunk = cache_position[pos:end]
            pid_chunk = position_ids[:, pos:end]
            kv_len = past_seen + end
            mask_chunk = _build_chunk_mask(cp_chunk, kv_len, attention_mask,
                                           hidden_states.dtype, device)

            r_chunk = r if self.recursion_enabled else torch.zeros_like(r)
            W_sum = torch.zeros(B, kc, self.config.h_rec, device=device, dtype=h_chunk.dtype)

            for block in self.layers:
                block_out = block(
                    h_chunk,
                    r_t=r_chunk,                              # constant across the chunk
                    attention_mask=mask_chunk,
                    position_ids=pid_chunk,
                    past_key_value=past_key_values,
                    use_cache=True,
                    cache_position=cp_chunk,
                    position_embeddings=pe_chunk,
                )
                h_chunk, w_l = block_out[0], block_out[1]     # w_l (B, kc, H_rec)
                W_sum = W_sum + w_l

            outputs_hidden.append(h_chunk)

            if self.recursion_enabled:
                W_total = W_sum / n_layers                    # (B, kc, H_rec)
                for j in range(kc):
                    r, c = self.recursion.step(W_total[:, j, :], r, c)

            pos = end

        hidden_states = torch.cat(outputs_hidden, dim=1)
        hidden_states = self.norm(hidden_states)

        self._persistent_r = r.detach()
        self._persistent_c = c.detach()

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=None,
            attentions=None,
        )


class AeonForCausalLM(AeonPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: AeonConfig):
        super().__init__(config)
        self.model = AeonModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    # ---- embedding / decoder accessors (used by tie_weights, resize, generate)
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    # ---- forward --------------------------------------------------------
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        num_logits_to_keep=0,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = (slice(-num_logits_to_keep, None)
                         if num_logits_to_keep else slice(None))
        logits = self.lm_head(hidden_states[:, slice_indices, :]).float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1).to(shift_logits.device),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      attention_mask=None, inputs_embeds=None,
                                      cache_position=None, use_cache=True, **kwargs):
        past_len = 0
        if past_key_values is not None:
            past_len = past_key_values.get_seq_length()
            if input_ids.shape[1] > past_len:
                input_ids = input_ids[:, past_len:]
            else:
                input_ids = input_ids[:, -1:]
        if cache_position is None:
            cache_position = torch.arange(past_len, past_len + input_ids.shape[1],
                                          device=input_ids.device)
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
        }

    # ---- convenience ----------------------------------------------------
    def reset_recursion_state(self, batch_size: int = 1):
        self.model.reset_recursion_state(batch_size)

    def disable_recursion(self):
        """Ablation / warm-start gate: run with the recurrent path inert."""
        self.model.recursion_enabled = False

    def enable_recursion(self):
        self.model.recursion_enabled = True

    def audit(self):
        return audit_certificates(self.model.recursion)
