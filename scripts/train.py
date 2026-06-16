"""
scripts/train.py — unified, config-driven trainer (Layer 4).

Replaces the per-stage scripts as the primary entry point; the old
scripts/train_stage1.py / train_stage2.py remain as historical references.
Driven by a YAML config (see configs/). Uses aeon.data, aeon.utils, aeon.eval.

Usage:
    python scripts/train.py --config configs/stage1.yaml
    python scripts/train.py --config configs/stage1.yaml --set train.steps=100
"""
import os, sys, json, argparse
import torch
from transformers import AutoTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aeon.config import AeonConfig
from aeon.model import AeonForCausalLM
from aeon.recursion import audit_certificates
from aeon.utils import load_yaml, merge_dicts, get_logger, save_training_state, load_training_state
from aeon.data import JsonlTextDataset, collate_causal, CurriculumSampler
from aeon.eval import perplexity

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def is_recursion_param(name: str) -> bool:
    return ("recursion." in name or name.endswith(".U.weight")
            or name.endswith(".D_proj.weight") or name.endswith(".recursion_gate")
            or "r_init" in name or "c_init" in name)


def apply_overrides(cfg, pairs):
    for p in pairs or []:
        key, _, val = p.partition("=")
        try:
            val = json.loads(val)
        except Exception:
            pass
        node = cfg
        parts = key.split(".")
        for k in parts[:-1]:
            node = node.setdefault(k, {})
        node[parts[-1]] = val
    return cfg


def build_model(mc, device):
    dtype = DTYPES[mc.get("dtype", "bfloat16")]
    if mc.get("init"):
        model = AeonForCausalLM.from_pretrained(mc["init"], torch_dtype=dtype)
        tok = AutoTokenizer.from_pretrained(mc["init"])
    else:                                            # from-scratch
        model = AeonForCausalLM(AeonConfig(**mc["config"])).to(dtype)
        tok = AutoTokenizer.from_pretrained(mc["tokenizer"]) if mc.get("tokenizer") else None
    model.config.recursion_chunk_size = mc.get("recursion_chunk_size", 1)
    return model.to(device), tok


def set_trainable(model, mode, log):
    if mode == "backbone":     # freeze backbone, train recursion path
        for n, p in model.named_parameters():
            p.requires_grad_(is_recursion_param(n))
    else:                      # "none"/"two_lr": everything trains
        for p in model.parameters():
            p.requires_grad_(True)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    log.info(f"trainable {n_tr:,}/{n_all:,} ({100*n_tr/n_all:.2f}%), freeze={mode}")


def make_optimizer(model, tc):
    if tc.get("freeze") == "two_lr":
        rec = [p for n, p in model.named_parameters() if p.requires_grad and is_recursion_param(n)]
        bb = [p for n, p in model.named_parameters() if p.requires_grad and not is_recursion_param(n)]
        return torch.optim.AdamW(
            [{"params": rec, "lr": tc["lr"]},
             {"params": bb, "lr": tc.get("backbone_lr", tc["lr"] * 0.1)}],
            weight_decay=tc.get("weight_decay", 0.0))
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                             lr=tc["lr"], weight_decay=tc.get("weight_decay", 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. train.steps=100")
    ap.add_argument("--resume", default=None, help="a checkpoint dir to resume from")
    args, _ = ap.parse_known_args()

    cfg = apply_overrides(load_yaml(args.config), args.set)
    mc, tc, dc, ec, lc = (cfg.get("model", {}), cfg.get("train", {}),
                          cfg.get("data", {}), cfg.get("eval", {}), cfg.get("log", {}))
    out = lc.get("out", "./aeon_out")
    os.makedirs(out, exist_ok=True)
    log = get_logger("aeon.train", logfile=os.path.join(out, "train.log"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = tc.get("seed", 42)
    torch.manual_seed(seed)

    init_dir = args.resume or None
    if init_dir:
        mc = dict(mc, init=init_dir)
    model, tok = build_model(mc, device)
    set_trainable(model, tc.get("freeze", "backbone"), log)
    model.train()

    steps = tc.get("steps", 1000)
    opt = make_optimizer(model, tc)
    sched = get_cosine_schedule_with_warmup(opt, tc.get("warmup_steps", 0), steps)

    ds = JsonlTextDataset(dc["path"], dc.get("text_field", "text"))
    rows = ds.rows
    # ensure n_tokens present for curriculum
    if dc.get("curriculum"):
        for r in rows:
            r.setdefault("n_tokens", len(tok(r["text"], add_special_tokens=False).input_ids))
    # holdout for perplexity eval
    import random as _r
    order = list(range(len(rows))); _r.Random(seed).shuffle(order)
    n_hold = min(ec.get("holdout_rows", 0), len(rows) // 10) if ec.get("every", 0) else 0
    holdout = [rows[i]["text"] for i in order[:n_hold]]
    train_rows = [rows[i] for i in order[n_hold:]]
    sampler = CurriculumSampler(train_rows, steps, seed=seed) if dc.get("curriculum") else None
    log.info(f"{len(train_rows)} train rows, {len(holdout)} holdout, curriculum={bool(sampler)}")

    start_step = 0
    if args.resume and os.path.isfile(os.path.join(args.resume, "training_state.pt")):
        start_step, extra = load_training_state(
            os.path.join(args.resume, "training_state.pt"), optimizer=opt, scheduler=sched)
        if sampler is not None and extra.get("sampler"):
            sampler.load_state(extra["sampler"])
        log.info(f"resumed at step {start_step}")

    bs = tc.get("batch_size", 1)
    seq_len = tc.get("seq_len", 512)
    grad_clip = tc.get("grad_clip", 1.0)
    flat = train_rows  # simple-loader pool
    step = start_step
    while step < steps:
        if sampler is not None:
            batch = [sampler.next(step) for _ in range(bs)]
            texts = [b["text"] for b in batch]
            max_len = min(seq_len, sampler.budget(step))
        else:
            i = (step * bs) % max(1, len(flat))
            texts = [flat[(i + j) % len(flat)]["text"] for j in range(bs)]
            max_len = seq_len
        ids, labels, mask = collate_causal(texts, tok, max_len)
        ids, labels, mask = ids.to(device), labels.to(device), mask.to(device)
        model.reset_recursion_state(batch_size=ids.shape[0])
        loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
        opt.step(); sched.step()
        step += 1

        if step % lc.get("every", 10) == 0:
            gates = [b.recursion_gate.item() for b in model.model.layers]
            log.info(f"step {step}/{steps} loss {loss.item():.4f} "
                     f"mean|g|={sum(abs(g) for g in gates)/len(gates):.4f}")
        if step % lc.get("audit_every", 50) == 0:
            a = audit_certificates(model.model.recursion)
            log.info(f"audit sigma(Wh)={a['chart_A_sigma_Wh']:.4f} holds={a['chart_A_holds']}")
            assert a["chart_A_holds"], "certificate violated"
        if ec.get("every", 0) and step % ec["every"] == 0 and holdout:
            model.eval()
            log.info(f"eval step {step}: perplexity={perplexity(model, tok, holdout, device, seq_len):.3f}")
            model.train()
        if step % lc.get("checkpoint_every", steps) == 0 or step == steps:
            d = os.path.join(out, f"step_{step}")
            model.save_pretrained(d);
            if tok is not None: tok.save_pretrained(d)
            extra = {"sampler": sampler.state()} if sampler is not None else {}
            save_training_state(os.path.join(d, "training_state.pt"),
                                step=step, optimizer=opt, scheduler=sched, extra=extra)
            log.info(f"checkpoint {d}")

    log.info("done.")


if __name__ == "__main__":
    main()
