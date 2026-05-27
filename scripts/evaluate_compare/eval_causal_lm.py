from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.gpt_bert import ModelConfig, GPTBERTForCausalLM


@dataclass
class EvalResult:
    name: str
    nll: float
    tokens: int
    words: int
    bytes_: int
    docs: int

    @property
    def nll_per_token(self) -> float:
        return self.nll / max(1, self.tokens)

    @property
    def ppl_per_token(self) -> float:
        return math.exp(min(self.nll_per_token, 20))

    @property
    def nll_per_word(self) -> float:
        return self.nll / max(1, self.words)

    @property
    def bits_per_byte(self) -> float:
        return self.nll / max(1, self.bytes_) / math.log(2)


def read_texts(path: Path, limit: int | None) -> list[str]:
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if isinstance(item, str):
                    text = item
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content") or item.get("document") or ""
                else:
                    text = str(item)
            except json.JSONDecodeError:
                text = line

            text = text.strip()
            if text:
                texts.append(text)
            if limit is not None and len(texts) >= limit:
                break
    return texts


def load_gpt_bert(model_dir: Path, config_path: Path, device: torch.device, checkpoint_pt: Path | None = None):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    config = ModelConfig(config_file=config_path)
    if not hasattr(config, "num_layers"):
        config.num_layers = config.num_hidden_layers

    model = GPTBERTForCausalLM(config)
    weights_path = checkpoint_pt if checkpoint_pt is not None else model_dir / "pytorch_model.bin"
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return tokenizer, model, config.max_position_embeddings


def load_gpt2(model_dir: Path, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    max_len = getattr(model.config, "n_positions", 1024)
    return tokenizer, model, max_len


def score_text(tokenizer, model, text: str, device: torch.device, max_len: int, stride: int) -> tuple[float, int]:
    ids = tokenizer(text, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
    if ids.numel() < 2:
        return 0.0, 0

    total_nll = 0.0
    total_tokens = 0
    start = 0

    while start < ids.numel() - 1:
        end = min(start + max_len, ids.numel())
        chunk = ids[start:end]
        if chunk.numel() < 2:
            break

        input_ids = chunk[:-1].unsqueeze(0).to(device)
        labels = chunk[1:].to(device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            logits = outputs.logits[0]
            loss_sum = F.cross_entropy(logits, labels, reduction="sum")

        total_nll += float(loss_sum.item())
        total_tokens += labels.numel()

        if end == ids.numel():
            break
        start += stride

    return total_nll, total_tokens


def evaluate(name: str, tokenizer, model, texts: list[str], device: torch.device, max_len: int, stride: int) -> EvalResult:
    total_nll = 0.0
    total_tokens = 0
    total_words = 0
    total_bytes = 0

    for i, text in enumerate(texts, start=1):
        nll, tokens = score_text(tokenizer, model, text, device, max_len=max_len, stride=stride)
        total_nll += nll
        total_tokens += tokens
        total_words += len(text.split())
        total_bytes += len(text.encode("utf-8"))
        if i == 1:
            preview_ids = tokenizer(text, add_special_tokens=True)["input_ids"][:64]
            print(f"\n{name} first_doc_ids={preview_ids}")
            print(f"{name} first_doc_decoded={tokenizer.decode(preview_ids)}")

    return EvalResult(name, total_nll, total_tokens, total_words, total_bytes, len(texts))


def print_result(result: EvalResult) -> None:
    print(f"\n[{result.name}]")
    print(f"docs={result.docs}")
    print(f"tokens={result.tokens}")
    print(f"words={result.words}")
    print(f"bytes={result.bytes_}")
    print(f"nll_total={result.nll:.4f}")
    print(f"nll_per_token={result.nll_per_token:.4f}")
    print(f"ppl_per_token={result.ppl_per_token:.2f}")
    print(f"nll_per_word={result.nll_per_word:.4f}")
    print(f"bits_per_byte={result.bits_per_byte:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GPT-BERT and GPT2 checkpoints on held-out raw text.")
    parser.add_argument("--data", type=Path, required=True, help="Plain text or JSONL file. One document per line.")
    parser.add_argument("--limit", type=int, default=1000, help="Maximum number of documents to evaluate.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpt-bert-dir", type=Path, default=ROOT / "checkpoints" / "babyLM-gpt-bert-mixed")
    parser.add_argument("--gpt-bert-pt", type=Path, default=None, help="Training checkpoint .pt saved by train_gpt_bert_10m.py.")
    parser.add_argument("--gpt-bert-config", type=Path, default=ROOT / "configs" / "gpt_bert_100m.json")
    parser.add_argument("--gpt2-dir", type=Path, default=ROOT / "checkpoints" / "gpt2-strict-small")
    parser.add_argument("--models", choices=["both", "gpt-bert", "gpt2"], default="both")
    parser.add_argument("--stride", type=int, default=256)
    args = parser.parse_args()

    texts = read_texts(args.data, args.limit)
    if not texts:
        raise ValueError(f"No texts found in {args.data}")

    device = torch.device(args.device)
    print(f"data={args.data}")
    print(f"docs={len(texts)}")
    print(f"device={device}")
    print(f"first_text={texts[0][:300]!r}")

    results: list[EvalResult] = []

    if args.models in {"both", "gpt-bert"}:
        tokenizer, model, max_len = load_gpt_bert(args.gpt_bert_dir, args.gpt_bert_config, device, args.gpt_bert_pt)
        results.append(evaluate("gpt-bert", tokenizer, model, texts, device, max_len=min(max_len, 512), stride=args.stride))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.models in {"both", "gpt2"}:
        tokenizer, model, max_len = load_gpt2(args.gpt2_dir, device)
        results.append(evaluate("gpt2", tokenizer, model, texts, device, max_len=min(max_len, 1024), stride=args.stride))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for result in results:
        print_result(result)

    if len(results) == 2:
        a, b = results
        print("\n[Comparison]")
        print("Lower is better for nll_per_token, nll_per_word, bits_per_byte.")
        print(f"best_by_token={'gpt-bert' if results[0].nll_per_token < results[1].nll_per_token else 'gpt2'}")
        print(f"best_by_word={'gpt-bert' if results[0].nll_per_word < results[1].nll_per_word else 'gpt2'}")
        print(f"best_by_byte={'gpt-bert' if results[0].bits_per_byte < results[1].bits_per_byte else 'gpt2'}")


if __name__ == "__main__":
    main()
