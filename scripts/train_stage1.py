"""
scripts/train_stage1.py — Stage 1: freeze R1 backbone, train recursion path only.

Trains: recursion.* (Cayley-D params), each block's U, D_proj, gamma.
Frozen: every Qwen2 weight loaded from R1.

Objective: next-token cross-entropy on instruction-following data.
The recursion learns to contribute without breaking R1's behavior.
By the end, gamma values should be small but nonzero.

Data: pass --data pointing to a .jsonl file with a 'text' field per row.
A helper, scripts/prepare_alpaca.py, writes such a file from tatsu-lab/alpaca.

Usage:
    python scripts/train_stage1.py --data <path> --out ./aeon_stage1/
"""
import os, sys, argparse, json
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aeon.model import AeonR1ForCausalLM
from aeon.recursion import audit_certificates


def freeze_r1_parts(model: AeonR1ForCausalLM):
    """Freeze everything except recursion + per-block U/D/gamma."""
    for name, p in model.named_parameters():
        keep = (
            "recursion." in name
            or name.endswith(".U.weight")
            or name.endswith(".D_proj.weight")
            or name.endswith(".recursion_gate")
            or "r_init" in name
            or "c_init" in name
        )
        p.requires_grad_(bool(keep))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable: {trainable:,} / {total:,} "
          f"({100*trainable/total:.2f}%)")


def collate(batch, tok, max_len=1024):
    # batch: list of strings (preformatted prompts+responses).
    enc = tok(batch, padding=True, truncation=True,
              max_length=max_len, return_tensors="pt")
    ids = enc.input_ids
    labels = ids.clone()
    labels[enc.attention_mask == 0] = -100
    return ids, labels, enc.attention_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="./aeon_init")
    ap.add_argument("--out", default="./aeon_stage1")
    ap.add_argument("--data", required=True,
                    help="path to a .jsonl file with field 'text' per row")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--audit_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=500)
    args, _ = ap.parse_known_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"loading from {args.init} ...")
    model = AeonR1ForCausalLM.from_pretrained(
        args.init, torch_dtype=torch.bfloat16).to(device)
    tok = AutoTokenizer.from_pretrained(args.init)
    freeze_r1_parts(model)
    model.train()

    # Dataset
    with open(args.data) as f:
        rows = [json.loads(line)["text"] for line in f if line.strip()]
    print(f"{len(rows)} training rows")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    step = 0
    for epoch in range(10**6):
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i:i + args.batch_size]
            ids, labels, mask = collate(batch, tok, args.seq_len)
            ids = ids.to(device); labels = labels.to(device); mask = mask.to(device)

            model.reset_recursion_state(batch_size=ids.shape[0])
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = out.loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            opt.step()

            if step % 10 == 0:
                gates = [model.model.layers[l].recursion_gate.item()
                         for l in range(len(model.model.layers))]
                print(f"step {step:6d}  loss {loss.item():.4f}  "
                      f"mean|g|={sum(abs(g) for g in gates)/len(gates):.4f}")
            if step % args.audit_every == 0:
                a = audit_certificates(model.model.recursion)
                print(f"  audit: sigma(Wh)={a['chart_A_sigma_Wh']:.4f}  "
                      f"sigma(Wc)={a['chart_A_sigma_Wc']:.4f}  "
                      f"holds={a['chart_A_holds']}")
                assert a['chart_A_holds'], "Certificate violated during training!"
            if step > 0 and step % args.save_every == 0:
                os.makedirs(args.out, exist_ok=True)
                model.save_pretrained(args.out)
                tok.save_pretrained(args.out)
                print(f"  saved checkpoint to {args.out}")
            step += 1
            if step >= args.steps:
                model.save_pretrained(args.out)
                tok.save_pretrained(args.out)
                print(f"done. final checkpoint at {args.out}")
                return


if __name__ == "__main__":
    main()
