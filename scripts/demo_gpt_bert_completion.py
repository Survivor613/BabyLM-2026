from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gpt_bert import ModelConfig, GPTBERTForCausalLM


def load_model(model_dir: Path, config_path: Path, device: str, checkpoint_pt: Path | None = None) -> tuple[AutoTokenizer, GPTBERTForCausalLM]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    config = ModelConfig(config_file=config_path)
    if not hasattr(config, "num_layers"):
        config.num_layers = config.num_hidden_layers

    model = GPTBERTForCausalLM(config)
    weights_path = checkpoint_pt if checkpoint_pt is not None else model_dir / "pytorch_model.bin"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing checkpoint file: {weights_path}")

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"missing_keys={missing}")
    if unexpected:
        print(f"unexpected_keys={unexpected}")

    model.to(device)
    model.eval()
    return tokenizer, model


def print_topk(tokenizer, logits: torch.Tensor, k: int) -> None:
    probs = torch.softmax(logits.float(), dim=-1)
    values, indices = torch.topk(probs, k)

    print("\nTop next-token predictions:")
    for rank, (token_id, prob) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
        token = tokenizer.decode([token_id])
        raw_token = tokenizer.convert_ids_to_tokens(token_id)
        print(f"{rank:>2}. id={token_id:<5} prob={prob:.4f} token={token!r} raw={raw_token!r}")


@torch.no_grad()
def generate(tokenizer, model, text: str, device: str, max_new_tokens: int, top_k: int, temperature: float) -> list[int]:
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    generated = input_ids

    for _ in range(max_new_tokens):
        attention_mask = torch.ones_like(generated)
        outputs = model(input_ids=generated, attention_mask=attention_mask, return_dict=True)
        next_logits = outputs.logits[:, -1, :] / max(temperature, 1e-6)

        if top_k > 0:
            top_values, top_indices = torch.topk(next_logits, top_k, dim=-1)
            next_probs = torch.softmax(top_values, dim=-1)
            sampled = torch.multinomial(next_probs, num_samples=1)
            next_token = top_indices.gather(-1, sampled)
        else:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=-1)

    return generated[0].tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Show GPT-BERT checkpoint next-token and completion behavior.")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "checkpoints" / "babyLM-gpt-bert-mixed")
    parser.add_argument("--checkpoint-pt", type=Path, default=None, help="Training checkpoint .pt saved by train_gpt_bert_10m.py.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "gpt_bert_100m.json")
    parser.add_argument("--text", default="I love NLP very much")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding instead of top-k sampling.")
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_dir, args.config, args.device, args.checkpoint_pt)

    inputs = tokenizer(args.text, return_tensors="pt").to(args.device)
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)

    print(f"prompt={args.text!r}")
    print(f"input_ids={inputs['input_ids'].tolist()[0]}")
    print_topk(tokenizer, outputs.logits[0, -1], args.top_k)

    decode_top_k = 0 if args.greedy else args.top_k
    generated_ids = generate(
        tokenizer=tokenizer,
        model=model,
        text=args.text,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        top_k=decode_top_k,
        temperature=args.temperature,
    )
    print("\nGenerated:")
    print(tokenizer.decode(generated_ids))
    print(f"generated_ids={generated_ids}")


if __name__ == "__main__":
    main()
