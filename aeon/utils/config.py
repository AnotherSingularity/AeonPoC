"""aeon.utils.config — YAML config loading and shallow merge."""
import copy


def load_yaml(path: str) -> dict:
    """Load a YAML file into a dict. Returns {} for an empty file."""
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f)
    return data or {}


def merge_dicts(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base` (override wins)."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_dicts(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def flatten_for_argparse(cfg: dict, prefix: str = "") -> dict:
    """Flatten a nested config to dotted keys, for logging/overrides."""
    flat = {}
    for k, v in cfg.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(flatten_for_argparse(v, key + "."))
        else:
            flat[key] = v
    return flat
