from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gpt_bert import ModelConfig, GPTBERTForCausalLM, GPTBERTForMaskedLM


def load_config(config_path: Path) -> ModelConfig:
    config = ModelConfig(config_file=config_path)
    if not hasattr(config, "num_layers"):
        config.num_layers = config.num_hidden_layers
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Instantiate the original GPT-BERT 100M architecture.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "gpt_bert_100m.json")
    parser.add_argument("--mode", choices=["causal", "masked"], default="causal")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--forward", action="store_true", help="Also run a forward pass. This needs enough memory.")
    args = parser.parse_args()

    config = load_config(args.config)
    model_cls = GPTBERTForCausalLM if args.mode == "causal" else GPTBERTForMaskedLM
    model = model_cls(config)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model={model_cls.__name__}")
    print(f"vocab_size={config.vocab_size}")
    print(f"hidden_size={config.hidden_size}")
    print(f"intermediate_size={config.intermediate_size}")
    print(f"num_hidden_layers={config.num_hidden_layers}")
    print(f"num_attention_heads={config.num_attention_heads}")
    print(f"max_position_embeddings={config.max_position_embeddings}")
    print(f"position_bucket_size={config.position_bucket_size}")
    print(f"params={n_params:,}")

    if not args.forward:
        return

    device = torch.device(args.device)
    model.to(device)
    model.eval()
    input_ids = torch.randint(5, config.vocab_size, (args.batch_size, args.seq_len), device=device)
    attention_mask = torch.ones(args.batch_size, args.seq_len, device=device, dtype=torch.long)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    print(f"logits_shape={tuple(outputs.logits.shape)}")


if __name__ == "__main__":
    main()
