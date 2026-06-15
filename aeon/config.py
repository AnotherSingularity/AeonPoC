"""aeon/config.py — AeonConfig subclasses Qwen2Config."""
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config


class AeonConfig(Qwen2Config):
    model_type = "aeon_r1"

    def __init__(
        self,
        h_rec: int = 256,
        margin_h: float = 0.98,
        margin_c: float = 0.95,
        recursion_init_learnable: bool = False,
        recursion_input_std: float = 0.01,
        recursion_output_std: float = 0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.h_rec = h_rec
        self.margin_h = margin_h
        self.margin_c = margin_c
        self.recursion_init_learnable = recursion_init_learnable
        self.recursion_input_std = recursion_input_std
        self.recursion_output_std = recursion_output_std
