"""
aeon/feedforward.py — AeonMLP, the SwiGLU feed-forward block.

Matches the gated-SiLU shape of the Qwen2-family reference (gate/up/down
projections, no bias) so warm-started weights are reproduced exactly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AeonMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
