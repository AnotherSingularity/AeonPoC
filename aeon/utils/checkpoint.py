"""aeon.utils.checkpoint — resumable training-state save/load.

The model weights/tokenizer are saved with the usual save_pretrained; this
module handles the *training* state (optimizer, scheduler, step, RNG) so a run
can resume bit-for-bit after an interruption.
"""
import os
import random
import numpy as np
import torch


def save_training_state(path: str, *, step, optimizer, scheduler=None, extra=None):
    """Write optimizer/scheduler/step/RNG state to `path` (a .pt file)."""
    state = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "extra": extra or {},
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(state, path)


def load_training_state(path: str, *, optimizer, scheduler=None, restore_rng=True):
    """Restore optimizer/scheduler/RNG from `path`. Returns (step, extra)."""
    # weights_only=False: our own trusted training-state pickle (optimizer/RNG).
    state = torch.load(path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if restore_rng:
        torch.set_rng_state(state["torch_rng"])
        if state.get("cuda_rng") is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(state["cuda_rng"])
            except Exception:
                pass
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
    return state["step"], state.get("extra", {})
