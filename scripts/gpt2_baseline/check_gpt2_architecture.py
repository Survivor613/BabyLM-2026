from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Instantiate the BabyLM 2026 GPT2 Strict-Small architecture.")
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints/gpt2-strict-small"))
    parser.add_argument("--forward", action="store_true", help="Run one random forward pass.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = GPT2Config.from_pretrained(args.model_dir)
    model = GPT2LMHeadModel(config)
    n_params = sum(p.numel() for p in model.parameters())

    print("model=GPT2LMHeadModel")
    print(f"vocab_size={config.vocab_size}")
    print(f"n_positions={config.n_positions}")
    print(f"n_ctx={config.n_ctx}")
    print(f"n_embd={config.n_embd}")
    print(f"n_layer={config.n_layer}")
    print(f"n_head={config.n_head}")
    print(f"activation_function={config.activation_function}")
    print(f"params={n_params:,}")

    if not args.forward:
        return

    model.to(args.device)
    model.eval()
    input_ids = torch.randint(0, config.vocab_size, (args.batch_size, args.seq_len), device=args.device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    print(f"logits_shape={tuple(outputs.logits.shape)}")


if __name__ == "__main__":
    main()
