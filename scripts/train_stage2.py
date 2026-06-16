"""
scripts/train_stage2.py — Stage 2: full co-training (UNFROZEN backbone).

Loads a Stage 1 checkpoint and continues training with the backbone unfrozen so
attention/MLP and the recursion path adapt to each other. By the end, ablating
the recursion path (`disable_recursion()`) should degrade quality — that is the
proof the recursion has become load-bearing.

STATUS: scaffold. The pieces below run, but the recipe is NOT finalized. See the
TODOs before launching a real Stage 2 — Stage 1's data/curriculum/LR will not be
right for co-training.

  TODO(dataset): alpaca (~52k short instruction pairs) is too small and too
      short to co-train a 1.5B backbone without overfitting / catastrophic
      forgetting. Move to a larger, longer-context instruction/chat mix
      (e.g. OpenAssistant + a FLAN slice + some long-form), and hold out a
      val set to watch for backbone drift.

  TODO(curriculum): the recursion only earns its keep when the sequence is long
      enough to need cross-token memory. Use a length curriculum — start near
      the Stage 1 seq_len and grow it — so the global state is actually
      exercised rather than bypassed by local attention.

  TODO(lr): the recursion params were warmed up at lr=1e-4; the backbone is
      pretrained and must move much more slowly (~1e-5 or lower) to avoid
      wrecking R1's behavior. This script uses two param groups
      (--recursion_lr, --backbone_lr) for exactly this. Tune both, and consider
      a warmup + cosine schedule on the backbone group only.

  NOTE(gate serialization): resolved. The gate param is now `recursion_gate`
      (no HF shim collision), so Stage 2 checkpoints round-trip cleanly. A
      pre-rename Stage 1 checkpoint must be passed through
      scripts/fix_gate_keys.py once before it will load here.

Usage (once the TODOs are settled):
    python scripts/train_stage2.py --init ./aeon_stage1 --data <path> --out ./aeon_stage2
"""
import os, sys, argparse, json
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aeon.model import AeonR1ForCausalLM
from aeon.recursion import audit_certificates
from scripts.train_stage1 import collate  # reuse the collate fn


def is_recursion_param(name: str) -> bool:
    """The params Stage 1 trained: recursion cell + per-block U / D_proj / recursion_gate."""
    return (
        "recursion." in name
        or name.endswith(".U.weight")
        or name.endswith(".D_proj.weight")
        or name.endswith(".recursion_gate")
        or "r_init" in name
        or "c_init" in name
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="./aeon_stage1",
                    help="Stage 1 checkpoint to continue from")
    ap.add_argument("--out", default="./aeon_stage2")
    ap.add_argument("--data", required=True,
                    help="path to a .jsonl file with field 'text' per row")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--recursion_lr", type=float, default=1e-4,
                    help="LR for the recursion path (continues Stage 1)")
    ap.add_argument("--backbone_lr", type=float, default=1e-5,
                    help="LR for the unfrozen R1 backbone (much lower)")
    ap.add_argument("--audit_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=500)
    args, _ = ap.parse_known_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"loading from {args.init} ...")
    model = AeonR1ForCausalLM.from_pretrained(
        args.init, torch_dtype=torch.bfloat16).to(device)
    tok = AutoTokenizer.from_pretrained(args.init)

    # Stage 2: everything trains, but in two LR groups.
    for p in model.parameters():
        p.requires_grad_(True)
    recursion_params, backbone_params = [], []
    for name, p in model.named_parameters():
        (recursion_params if is_recursion_param(name) else backbone_params).append(p)
    n_rec = sum(p.numel() for p in recursion_params)
    n_bb = sum(p.numel() for p in backbone_params)
    print(f"recursion group: {n_rec:,} params @ lr={args.recursion_lr}")
    print(f"backbone  group: {n_bb:,} params @ lr={args.backbone_lr}")
    model.train()

    with open(args.data) as f:
        rows = [json.loads(line)["text"] for line in f if line.strip()]
    print(f"{len(rows)} training rows")

    opt = torch.optim.AdamW([
        {"params": recursion_params, "lr": args.recursion_lr},
        {"params": backbone_params, "lr": args.backbone_lr},
    ])

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
