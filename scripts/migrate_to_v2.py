"""
scripts/migrate_to_v2.py — migrate a v1 Aeon checkpoint to the v2 layout.

Layer 2 renamed the wrapped transformer layer attribute `qwen_block` ->
`transformer_layer`, which changes the state-dict key prefix from
`model.layers.N.qwen_block.*` to `model.layers.N.transformer_layer.*`. A
checkpoint saved by v1 code (e.g. the June 15 Stage 1 checkpoint) must be passed
through this script once before it will load into v2 `AeonForCausalLM`.

This also normalizes config.json (`model_type` -> "aeon", architectures ->
["AeonForCausalLM"]). It does NOT touch the recurrent gate keys — run
fix_gate_keys.py first if the checkpoint predates the `recursion_gate` rename.

Usage:
    python scripts/migrate_to_v2.py --ckpt ./aeon_stage1_fixed          # -> ..._v2
    python scripts/migrate_to_v2.py --ckpt ./aeon_stage1_fixed --inplace
"""
import os, sys, json, shutil, argparse
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OLD = ".qwen_block."
NEW = ".transformer_layer."


def remap_key(key: str) -> str:
    return key.replace(OLD, NEW)


def _load_shard(path):
    tensors = {}
    with safe_open(path, framework="pt") as f:
        meta = f.metadata() or {}
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors, meta


def migrate(ckpt, out):
    index_path = os.path.join(ckpt, "model.safetensors.index.json")
    single_path = os.path.join(ckpt, "model.safetensors")
    if os.path.isfile(single_path):
        shards, index = ["model.safetensors"], None
    elif os.path.isfile(index_path):
        with open(index_path) as f:
            index = json.load(f)
        shards = sorted(set(index["weight_map"].values()))
    else:
        raise FileNotFoundError(f"no model.safetensors[.index.json] in {ckpt}")

    if os.path.abspath(out) != os.path.abspath(ckpt):
        os.makedirs(out, exist_ok=True)
        for name in os.listdir(ckpt):
            src = os.path.join(ckpt, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(out, name))

    n_renamed = 0
    for shard in shards:
        sd, meta = _load_shard(os.path.join(ckpt, shard))
        new_sd = {}
        for k, v in sd.items():
            nk = remap_key(k)
            n_renamed += (nk != k)
            new_sd[nk] = v
        meta.setdefault("format", "pt")
        save_file(new_sd, os.path.join(out, shard), metadata=meta)

    if index is not None:
        index["weight_map"] = {remap_key(k): v for k, v in index["weight_map"].items()}
        with open(os.path.join(out, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2)

    # normalize config.json
    cfg_path = os.path.join(out, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        cfg["model_type"] = "aeon"
        cfg["architectures"] = ["AeonForCausalLM"]
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

    return n_renamed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--inplace", action="store_true")
    args, _ = ap.parse_known_args()
    out = args.ckpt if args.inplace else (args.out or args.ckpt.rstrip("/") + "_v2")

    print(f"migrating {args.ckpt} -> {out} ...")
    n = migrate(args.ckpt, out)
    print(f"remapped {n} keys ({OLD.strip('.')} -> {NEW.strip('.')})")

    print("verifying by reloading ...")
    from aeon.model import AeonForCausalLM
    model = AeonForCausalLM.from_pretrained(out, torch_dtype=torch.float32)
    gates = [blk.recursion_gate.item() for blk in model.model.layers]
    mean_abs = sum(abs(g) for g in gates) / len(gates)
    print(f"loaded OK. mean|recursion_gate| = {mean_abs:.4f} across {len(gates)} layers")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
