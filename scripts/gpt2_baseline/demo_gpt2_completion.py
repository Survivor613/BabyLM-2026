from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def print_topk(tokenizer, logits: torch.Tensor, k: int) -> None:
    probs = torch.softmax(logits.float(), dim=-1)
    values, indices = torch.topk(probs, k)
    print("\nTop next-token predictions:")
    for rank, (token_id, prob) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
        token = tokenizer.decode([token_id])
        raw_token = tokenizer.convert_ids_to_tokens(token_id)
        print(f"{rank:>2}. id={token_id:<5} prob={prob:.4f} token={token!r} raw={raw_token!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Show BabyLM GPT2 Strict-Small completion behavior.")
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints/gpt2-strict-small"))
    parser.add_argument("--text", default="I love NLP very much")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir).to(args.device)
    model.eval()

    inputs = tokenizer(args.text, return_tensors="pt").to(args.device)
    with torch.no_grad():
        outputs = model(**inputs)

    print(f"prompt={args.text!r}")
    print(f"input_ids={inputs['input_ids'].tolist()[0]}")
    print(f"logits_shape={tuple(outputs.logits.shape)}")
    print_topk(tokenizer, outputs.logits[0, -1], args.top_k)

    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if args.greedy:
        generate_kwargs["do_sample"] = False
    else:
        generate_kwargs["do_sample"] = True
        generate_kwargs["top_k"] = args.top_k

    with torch.no_grad():
        generated = model.generate(**inputs, **generate_kwargs)

    print("\nGenerated:")
    print(tokenizer.decode(generated[0], skip_special_tokens=False))
    print(f"generated_ids={generated[0].tolist()}")


if __name__ == "__main__":
    main()
