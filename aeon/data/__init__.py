"""aeon.data — dataset utilities: formatting, a JSONL dataset, and the length
curriculum scheduler."""
from .formatting import format_alpaca, format_chat
from .dataset import JsonlTextDataset, collate_causal
from .curriculum import CurriculumSampler, phase_for_step, phase_boundaries

__all__ = [
    "format_alpaca",
    "format_chat",
    "JsonlTextDataset",
    "collate_causal",
    "CurriculumSampler",
    "phase_for_step",
    "phase_boundaries",
]
