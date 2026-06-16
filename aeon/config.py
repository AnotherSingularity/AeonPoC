"""aeon/config.py — AeonConfig subclasses Qwen2Config."""
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config


class AeonConfig(Qwen2Config):
    model_type = "aeon"

    def __init__(
        self,
        h_rec: int = 256,
        margin_h: float = 0.98,
        margin_c: float = 0.95,
        recursion_init_learnable: bool = False,
        recursion_input_std: float = 0.01,
        recursion_output_std: float = 0.01,
        recursion_chunk_size: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.h_rec = h_rec
        self.margin_h = margin_h
        self.margin_c = margin_c
        self.recursion_init_learnable = recursion_init_learnable
        self.recursion_input_std = recursion_input_std
        self.recursion_output_std = recursion_output_std
        # Chunk size K for the batched forward (Layer 3):
        #   0 (default) -> K = T, fully batched (one batched attention pass).
        #   1           -> per-token, identical semantics to the v1 loop.
        #   k           -> chunks of k tokens: batched attention within a chunk
        #                  with the recurrent read held fixed at the chunk-start
        #                  state, state advanced once per chunk.
        self.recursion_chunk_size = recursion_chunk_size
