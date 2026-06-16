"""
examples/finetune.py — minimal fine-tuning loop.

Shows the training shape: freeze the backbone, train the recursion path at K=1,
step the optimizer, check the contraction certificate. Runs on CPU with a tiny
synthetic model and synthetic token data (no download). For real runs use
scripts/train.py with a config in configs/.

    python examples/finetune.py
"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aeon import AeonConfig, AeonForCausalLM, audit_certificates


def is_recursion_param(name):
    return ("recursion." in name or name.endswith(".U.weight")
            or name.endswith(".D_proj.weight") or name.endswith(".recursion_gate"))


def main():
    torch.manual_seed(0)
    cfg = AeonConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
                     num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                     max_position_embeddings=128, h_rec=16, tie_word_embeddings=False)
    cfg.recursion_chunk_size = 1            # K=1 so the recursion is in the gradient path
    model = AeonForCausalLM(cfg).train()

    # freeze the backbone, train only the recursion path
    for n, p in model.named_parameters():
        p.requires_grad_(is_recursion_param(n))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    for step in range(20):
        ids = torch.randint(0, cfg.vocab_size, (1, 16))
        model.reset_recursion_state(batch_size=1)
        loss = model(input_ids=ids, labels=ids.clone()).loss
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 5 == 0:
            a = audit_certificates(model.model.recursion)
            mean_g = sum(abs(b.recursion_gate.item()) for b in model.model.layers) / cfg.num_hidden_layers
            print(f"step {step:2d}  loss {loss.item():.3f}  mean|gate| {mean_g:.4f}  "
                  f"cert_holds {a['chart_A_holds']}")
    print("done — the certificate held throughout.")


if __name__ == "__main__":
    main()
