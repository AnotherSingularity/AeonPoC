"""
scripts/fix_gate_keys.py — recover trained per-block gates from a pre-rename
checkpoint.

Background. The per-block gate used to be a parameter named `gamma`.
transformers' `save_pretrained` applies a legacy key rewrite ("gamma" -> "weight",
"beta" -> "bias", from old LayerNorm naming), so a checkpoint saved by that code
stores each gate under `model.layers.N.weight` instead of the parameter's real
name. `from_pretrained` cannot map it back, so the trained gate loads as zero.

This script renames those stray keys to the current parameter name
(`recursion_gate`) so the gate round-trips. The target is `recursion_gate`, NOT
`gamma`, on purpose: the shim also fires on load, so a `.gamma` key would be
rewritten back to `.weight` and still fail to match.

Usage:
    python scripts/fix_gate_keys.py --ckpt ./aeon_stage1            # -> ./aeon_stage1_fixed
    python scripts/fix_gate_keys.py --ckpt ./aeon_stage1 --inplace  # overwrite in place
"""
import os, re, sys, json, shutil, argparse
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STRAY_LAYER_WEIGHT = re.compile(r"^model\.layers\.\d+\.weight$")


def _load_shard(path):
    """Return (state_dict, metadata) for one safetensors file."""
    tensors = {}
    with safe_open(path, framework="pt") as f:
        meta = f.metadata() or {}
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors, meta


def remap_state_dict(sd, target_suffix="recursion_gate"):
    """Rename stray `model.layers.N.weight` keys to `...<target_suffix>`.
    Returns (new_sd, renamed) where renamed maps old_key -> new_key."""
    new_sd, renamed = {}, {}
    for k, v in sd.items():
        if STRAY_LAYER_WEIGHT.match(k):
            nk = k[: -len("weight")] + target_suffix
            new_sd[nk] = v
            renamed[k] = nk
        else:
            new_sd[k] = v
    return new_sd, renamed


def _stats(vals):
    vals = [abs(v) for v in vals]
    if not vals:
        return "none"
    return (f"n={len(vals)}  mean|.|={sum(vals)/len(vals):.4f}  "
            f"min|.|={min(vals):.4f}  max|.|={max(vals):.4f}")


def fix_checkpoint(ckpt, out, target_suffix="recursion_gate"):
    """Remap stray gate keys across single-file or sharded safetensors."""
    index_path = os.path.join(ckpt, "model.safetensors.index.json")
    single_path = os.path.join(ckpt, "model.safetensors")

    if os.path.isfile(single_path):
        shard_files = ["model.safetensors"]
        index = None
    elif os.path.isfile(index_path):
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
    else:
        raise FileNotFoundError(
            f"no model.safetensors or .index.json in {ckpt}")

    # Prepare output dir (copy everything, then overwrite the safetensors we touch).
    if os.path.abspath(out) != os.path.abspath(ckpt):
        os.makedirs(out, exist_ok=True)
        for name in os.listdir(ckpt):
            src = os.path.join(ckpt, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(out, name))

    before_vals, all_renamed = [], {}
    for shard in shard_files:
        sd, meta = _load_shard(os.path.join(ckpt, shard))
        before_vals += [v.float().flatten()[0].item()
                        for k, v in sd.items() if STRAY_LAYER_WEIGHT.match(k)]
        new_sd, renamed = remap_state_dict(sd, target_suffix)
        all_renamed.update(renamed)
        if "format" not in meta:
            meta["format"] = "pt"
        save_file(new_sd, os.path.join(out, shard), metadata=meta)

    # Update the index weight_map if sharded.
    if index is not None and all_renamed:
        index["weight_map"] = {all_renamed.get(k, k): v
                               for k, v in index["weight_map"].items()}
        with open(os.path.join(out, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2)

    return before_vals, all_renamed


def verify(out):
    """Reload the corrected checkpoint and read the gate values back."""
    from aeon.model import AeonR1ForCausalLM
    model = AeonR1ForCausalLM.from_pretrained(out, torch_dtype=torch.float32)
    return [blk.recursion_gate.item() for blk in model.model.layers]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint directory to fix")
    ap.add_argument("--out", default=None,
                    help="output dir (default: <ckpt>_fixed). Ignored with --inplace.")
    ap.add_argument("--inplace", action="store_true",
                    help="overwrite the safetensors in --ckpt directly")
    ap.add_argument("--target-name", default="recursion_gate",
                    help="parameter name to remap stray .weight keys to")
    args, _ = ap.parse_known_args()

    out = args.ckpt if args.inplace else (args.out or args.ckpt.rstrip("/") + "_fixed")

    print(f"reading {args.ckpt} ...")
    before_vals, renamed = fix_checkpoint(args.ckpt, out, args.target_name)

    if not renamed:
        print("no stray `model.layers.N.weight` keys found — nothing to remap.")
        print("(checkpoint is either already fixed or was not affected.)")
        return

    print(f"remapped {len(renamed)} keys: .weight -> .{args.target_name}")
    print(f"  BEFORE (values found under stray .weight keys): {_stats(before_vals)}")
    print(f"wrote {out}")

    print("verifying by reloading ...")
    after_vals = verify(out)
    nonzero = sum(1 for v in after_vals if abs(v) > 1e-12)
    print(f"  AFTER  (recursion_gate values on reload):       {_stats(after_vals)}")
    print(f"  {nonzero}/{len(after_vals)} gates loaded nonzero")
    if nonzero == 0:
        print("FAILED: gates still zero after reload.")
        sys.exit(1)
    print("OK: trained gates recovered.")


if __name__ == "__main__":
    main()
