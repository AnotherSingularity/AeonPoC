"""
scripts/train_stage2.py — Stage 2: full co-training.

Same objective and data format as Stage 1, but the R1 backbone is unfrozen
and the learning rate is lowered. Optional — Stage 1's output is already a
working chat model. By the end of Stage 2 the recursion path should be
load-bearing: ablating it (model.disable_recursion()) should degrade quality.

Usage:
    python scripts/train_stage2.py --init ./aeon_stage1 --data <path> --out ./aeon_stage2
"""
import os, sys, argparse, json
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aeon.model import AeonR1ForCausalLM
from aeon.recursion import audit_certificates
from scripts.train_stage1 import collate  # reuse the collate fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="./aeon_stage1")
    ap.add_argument("--out", default="./aeon_stage2")
    ap.add_argument("--data", required=True,
                    help="path to a .jsonl file with field 'text' per row")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--audit_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=500)
    args, _ = ap.parse_known_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"loading from {args.init} ...")
    model = AeonR1ForCausalLM.from_pretrained(
        args.init, torch_dtype=torch.bfloat16).to(device)
    tok = AutoTokenizer.from_pretrained(args.init)
    # Stage 2: everything trains. No freeze call.
    for p in model.parameters():
        p.requires_grad_(True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable: {trainable:,} (full co-training)")
    model.train()

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
                gammas = [model.model.layers[l].gamma.item()
                          for l in range(len(model.model.layers))]
                print(f"step {step:6d}  loss {loss.item():.4f}  "
                      f"mean|g|={sum(abs(g) for g in gammas)/len(gammas):.4f}")
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
