from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gpt_bert import GPTBERTForCausalLM, ModelConfig


def load_gpt_bert(model_dir: Path, config_path: Path, checkpoint_pt: Path, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    config = ModelConfig(config_file=config_path)
    if not hasattr(config, "num_layers"):
        config.num_layers = config.num_hidden_layers
    model = GPTBERTForCausalLM(config)
    state_dict = torch.load(checkpoint_pt, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return tokenizer, model


def load_gpt2(model_dir: Path, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir).to(device).eval()
    return tokenizer, model


def print_topk(tokenizer, logits: torch.Tensor, k: int) -> None:
    probs = torch.softmax(logits.float(), dim=-1)
    values, indices = torch.topk(probs, k)
    for rank, (token_id, prob) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
        token = tokenizer.decode([token_id])
        raw = tokenizer.convert_ids_to_tokens(token_id)
        print(f"  {rank:>2}. id={token_id:<5} prob={prob:.4f} token={token!r} raw={raw!r}")


@torch.no_grad()
def generate(tokenizer, model, text: str, device: str, max_new_tokens: int, top_k: int, greedy: bool, temperature: float):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    generated = inputs["input_ids"]
    for _ in range(max_new_tokens):
        attention_mask = torch.ones_like(generated)
        outputs = model(input_ids=generated, attention_mask=attention_mask, return_dict=True)
        next_logits = outputs.logits[:, -1, :] / max(temperature, 1e-6)
        if greedy:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
            top_values, top_indices = torch.topk(next_logits, top_k, dim=-1)
            next_probs = torch.softmax(top_values, dim=-1)
            sampled = torch.multinomial(next_probs, num_samples=1)
            next_token = top_indices.gather(-1, sampled)
        generated = torch.cat([generated, next_token], dim=-1)
    return generated[0].tolist()


def run_one(name: str, tokenizer, model, text: str, device: str, max_new_tokens: int, top_k: int, greedy: bool, temperature: float):
    print(f"\n===== {name} =====")
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
    print(f"prompt={text!r}")
    print(f"input_ids={inputs['input_ids'].tolist()[0]}")
    print("Top next-token predictions:")
    print_topk(tokenizer, outputs.logits[0, -1], top_k)
    generated = generate(tokenizer, model, text, device, max_new_tokens, top_k, greedy, temperature)
    print("Generated:")
    print(tokenizer.decode(generated))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GPT-BERT against GPT2 baselines on the same prompt.")
    parser.add_argument("--text", default="The little boy went to the")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true")

    parser.add_argument("--gpt-bert-dir", type=Path, default=ROOT / "checkpoints" / "babyLM-gpt-bert-mixed")
    parser.add_argument("--gpt-bert-config", type=Path, default=ROOT / "configs" / "gpt_bert_100m.json")
    parser.add_argument("--gpt-bert-pt", type=Path, required=True)

    parser.add_argument("--gpt2-strict-dir", type=Path, default=ROOT / "checkpoints" / "gpt2-strict")
    parser.add_argument("--gpt2-small-dir", type=Path, default=ROOT / "checkpoints" / "gpt2-strict-small")
    args = parser.parse_args()

    gbt_tok, gbt_model = load_gpt_bert(args.gpt_bert_dir, args.gpt_bert_config, args.gpt_bert_pt, args.device)
    gpt2_tok, gpt2_model = load_gpt2(args.gpt2_strict_dir, args.device)
    gpt2s_tok, gpt2s_model = load_gpt2(args.gpt2_small_dir, args.device)

    run_one("GPT-BERT (10M epoch9)", gbt_tok, gbt_model, args.text, args.device, args.max_new_tokens, args.top_k, args.greedy, args.temperature)
    run_one("GPT2 Strict", gpt2_tok, gpt2_model, args.text, args.device, args.max_new_tokens, args.top_k, args.greedy, args.temperature)
    run_one("GPT2 Strict-Small", gpt2s_tok, gpt2s_model, args.text, args.device, args.max_new_tokens, args.top_k, args.greedy, args.temperature)


if __name__ == "__main__":
    main()
