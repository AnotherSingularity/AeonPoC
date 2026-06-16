"""
aeon/block.py — AeonBlock.

Wraps a Qwen2DecoderLayer with recursion read/write:
  read:  shift the residual stream by gamma_l * U_l @ r_t  BEFORE attention
  write: produce w_l = D_l @ x_post  AFTER attention+MLP

The AeonModel collects all w_l from all blocks for one token, averages
them, and runs one Recursion step to update (r, c) before the next token.

At init: gamma_l = 0. Stage 0 verifier requires byte-identical output
to vanilla Qwen2 with these gates at zero.
"""
import torch
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer


class AeonBlock(nn.Module):
    """
    A drop-in replacement for Qwen2DecoderLayer that wraps the underlying
    Qwen2DecoderLayer with recursion read/write paths.

    The underlying Qwen2DecoderLayer is unchanged in structure; we add
    a residual shift before it and produce a recursion write after it.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.D = config.hidden_size
        self.H_rec = config.h_rec

        # The underlying Qwen2 block — loads R1's weights into here
        self.qwen_block = Qwen2DecoderLayer(config, layer_idx)

        # Recursion read: r_t (B, H_rec) -> shift (B, D)
        self.U = nn.Linear(self.H_rec, self.D, bias=False)
        nn.init.normal_(self.U.weight, std=config.recursion_output_std)

        # Recursion write: x_post (B, T, D) -> proposal (B, T, H_rec)
        self.D_proj = nn.Linear(self.D, self.H_rec, bias=False)
        nn.init.normal_(self.D_proj.weight, std=config.recursion_input_std)

        # Per-block gate gamma_l, exactly zero at init.
        # Named `recursion_gate`, not `gamma`: transformers' save/load applies a
        # legacy key rewrite (any key containing "gamma" -> "weight", "beta" ->
        # "bias"), which silently drops a parameter literally named "gamma" on
        # reload. `recursion_gate` matches no shim, so trained values round-trip.
        self.recursion_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        hidden_states: torch.Tensor,     # (B, T, D)
        r_t: torch.Tensor,               # (B, H_rec)   broadcast across T
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        """
        Returns: (hidden_states_out, w_l, *qwen_outputs)
            w_l : (B, T, H_rec) per-token recursion writes from this block
        """
        # 1. Read: shift the residual using r_t (broadcast across T)
        # r_t: (B, H_rec) -> m: (B, D) -> (B, 1, D) -> broadcast add to (B, T, D)
        m = self.U(r_t)                                # (B, D)
        shift = (self.recursion_gate * m).unsqueeze(1)  # (B, 1, D)
        hidden_states = hidden_states + shift          # (B, T, D)

        # 2. Standard Qwen2 block
        qwen_out = self.qwen_block(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        # Qwen2DecoderLayer returns a tuple; the first element is hidden_states
        if isinstance(qwen_out, tuple):
            hidden_states_out = qwen_out[0]
            rest = qwen_out[1:]
        else:
            hidden_states_out = qwen_out
            rest = ()

        # 3. Write: per-token recursion proposal from this block
        w_l = self.D_proj(hidden_states_out)           # (B, T, H_rec)

        return (hidden_states_out, w_l) + rest
