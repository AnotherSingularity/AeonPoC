"""
scripts/audit_checkpoint.py — read the three-chart audit on any checkpoint.

Usage:
    python scripts/audit_checkpoint.py --ckpt ./aeon_stage1
"""
import os, sys, argparse, json
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aeon.model import AeonR1ForCausalLM
from aeon.recursion import audit_certificates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    args, _ = ap.parse_known_args()
    model = AeonR1ForCausalLM.from_pretrained(args.ckpt, torch_dtype=torch.float32)
    a = audit_certificates(model.model.recursion)
    print(json.dumps(a, indent=2))


if __name__ == "__main__":
    main()
