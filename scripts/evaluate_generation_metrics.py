from __future__ import annotations

import argparse
import math
import random
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
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


def score_prompt(tokenizer, model, text: str, device: str):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, return_dict=True)
        logits = out.logits[:, :-1, :]
        labels = inputs["input_ids"][:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="mean")
    return float(loss.item())


@torch.no_grad()
def generate_text(tokenizer, model, text: str, device: str, max_new_tokens: int, top_k: int, greedy: bool, temperature: float) -> str:
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
    return tokenizer.decode(generated[0], skip_special_tokens=False)


def ngram_counts(tokens: list[int], n: int) -> Counter[tuple[int, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def distinct_n(texts: list[str], tokenizer, n: int) -> float:
    all_ngrams = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        all_ngrams.extend(list(ngram_counts(ids, n).keys()))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def repetition_rate(text: str, tokenizer, n: int = 4) -> float:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) < n * 2:
        return 0.0
    grams = [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - (len(set(grams)) / len(grams))


def length_stats(text: str, tokenizer) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def evaluate_model(name: str, tokenizer, model, prompts: list[str], device: str, max_new_tokens: int, top_k: int, greedy: bool, temperature: float):
    prompt_losses = []
    generations = []
    lengths = []
    reps = []
    for prompt in prompts:
        prompt_losses.append(score_prompt(tokenizer, model, prompt, device))
        gen = generate_text(tokenizer, model, prompt, device, max_new_tokens, top_k, greedy, temperature)
        generations.append(gen)
        lengths.append(length_stats(gen, tokenizer))
        reps.append(repetition_rate(gen, tokenizer))

    avg_nll = sum(prompt_losses) / len(prompt_losses)
    ppl = math.exp(min(avg_nll, 20))
    out = {
        "name": name,
        "avg_prompt_nll": avg_nll,
        "prompt_ppl": ppl,
        "avg_generated_len": sum(lengths) / len(lengths),
        "avg_repetition_rate_4gram": sum(reps) / len(reps),
        "distinct1": distinct_n(generations, tokenizer, 1),
        "distinct2": distinct_n(generations, tokenizer, 2),
        "distinct3": distinct_n(generations, tokenizer, 3),
    }
    return out, generations


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generation quality with quantitative metrics.")
    parser.add_argument("--prompts-file", type=Path, default=None, help="One prompt per line. If omitted, use built-in prompts.")
    parser.add_argument("--max-prompts", type=int, default=10)
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

    prompts = [
        "The little boy went to the",
        "The teacher asked the child",
        "Once upon a time",
        "In the morning",
        "The woman opened the",
        "The cat sat on the",
        "After dinner, the family",
        "The children were playing",
        "My friend said that",
        "When the door opened,",
    ]
    if args.prompts_file is not None:
        prompts = [line.strip() for line in args.prompts_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    prompts = prompts[: args.max_prompts]

    gbt_tok, gbt_model = load_gpt_bert(args.gpt_bert_dir, args.gpt_bert_config, args.gpt_bert_pt, args.device)
    gpt2_tok, gpt2_model = load_gpt2(args.gpt2_strict_dir, args.device)
    gpt2s_tok, gpt2s_model = load_gpt2(args.gpt2_small_dir, args.device)

    results = []
    for name, tok, model in [
        ("GPT-BERT", gbt_tok, gbt_model),
        ("GPT2 Strict", gpt2_tok, gpt2_model),
        ("GPT2 Strict-Small", gpt2s_tok, gpt2s_model),
    ]:
        result, generations = evaluate_model(name, tok, model, prompts, args.device, args.max_new_tokens, args.top_k, args.greedy, args.temperature)
        results.append((result, generations))

    print("\n=== Generation Metrics ===")
    for result, _ in results:
        print(f"\n[{result['name']}]")
        for k, v in result.items():
            if k == "name":
                continue
            print(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")

    print("\nLower avg_prompt_nll / prompt_ppl / repetition rate is better. Higher distinct-n is better.")

    print("\n=== Sample generations ===")
    for (result, generations) in results:
        print(f"\n[{result['name']}]")
        for prompt, gen in zip(prompts[:3], generations[:3]):
            print(f"PROMPT: {prompt!r}")
            print(f"GEN: {gen}")


if __name__ == "__main__":
    main()
