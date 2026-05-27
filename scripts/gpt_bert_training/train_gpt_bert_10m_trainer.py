from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "src" / "gpt_bert").is_dir():
            return path
    raise RuntimeError(f"Could not find repo root containing src/gpt_bert from {start}")


ROOT = find_repo_root(Path(__file__).resolve().parent)
sys.path.insert(0, str(ROOT))

from src.gpt_bert import GPTBERTForCausalLM, ModelConfig


GPT_MODE = 0
BERT_MODE = 1


class Lamb(torch.optim.Optimizer):
    """Official GPT-BERT LAMB optimizer implementation."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    def step(self, closure=None):
        loss = True
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Lamb does not support sparse gradients.")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                m_t = exp_avg / bias_correction1
                v_t = exp_avg_sq / bias_correction2
                torch.sqrt_(v_t)
                update = m_t / (v_t + group["eps"])

                ratio = 1.0
                if group["weight_decay"] > 0:
                    update.add_(p.data, alpha=group["weight_decay"])
                    g_norm = torch.norm(update.flatten())
                    w_norm = torch.norm(p.data.flatten())
                    if w_norm > 0.0 and g_norm > 0.0:
                        ratio = w_norm / g_norm

                p.data.add_(update, alpha=-group["lr"] * ratio)

        return loss


def cosine_schedule_with_warmup_cooldown(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_cooldown_steps: int,
    num_training_steps: int,
    min_factor: float,
):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if current_step >= num_training_steps - num_cooldown_steps:
            return min_factor * float(num_training_steps - current_step) / float(max(1, num_cooldown_steps))

        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_factor, min_factor + (1 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def seed_everything(seed_value: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed() -> tuple[torch.device, int, int, bool]:
    if not is_distributed():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device, 0, 1, True

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank), rank, world_size, rank == 0


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    return tensor


class SpanMaskingStrategy:
    def __init__(self, n_special_tokens: int, random_p: float, keep_p: float, vocab_size: int, mask_token_id: int):
        self.n_special_tokens = n_special_tokens
        self.random_p = random_p
        self.keep_p = keep_p
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.max_span_length = 3

    def __call__(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        length = tokens.size(0)
        span_lengths = torch.randint(1, self.max_span_length + 1, size=(length,), dtype=torch.int)
        cumsum = torch.cumsum(span_lengths, dim=0)

        indices = torch.zeros(cumsum[-1].item(), dtype=torch.int)
        indices[cumsum - span_lengths] = torch.arange(length, dtype=torch.int)
        indices = torch.cummax(indices, dim=0)[0][:length]

        span_random_1, span_random_2 = torch.rand([(indices[-1].item() + 1) * 2]).chunk(2)
        mask_ratios = span_random_1[indices]
        mask_ratios[tokens < self.n_special_tokens] = float("inf")

        replacement_p = span_random_2[indices]
        random_mask = replacement_p < self.random_p
        replacement_tokens = tokens.clone()
        replacement_tokens[random_mask] = torch.randint(
            low=self.n_special_tokens,
            high=self.vocab_size,
            size=[random_mask.sum().item()],
            dtype=torch.long,
        )
        replacement_tokens[replacement_p > (self.random_p + self.keep_p)] = self.mask_token_id
        return mask_ratios, replacement_tokens


class OfficialStyleTextDataset(Dataset):
    """Read train.txt files, tokenize into document segments, and pack with official masks."""

    def __init__(
        self,
        data_dir: Path,
        tokenizer,
        seq_len: int,
        seed: int,
        hybrid_numerator: int,
        hybrid_denominator: int,
        n_special_tokens: int,
        mask_p_start: float,
        mask_p_end: float,
        mask_random_p: float,
        mask_keep_p: float,
        max_steps: int,
        max_lines_per_file: int | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.seed = seed
        self.hybrid_numerator = hybrid_numerator
        self.hybrid_denominator = hybrid_denominator
        self.n_special_tokens = n_special_tokens
        self.mask_p_start = mask_p_start
        self.mask_p_end = mask_p_end
        self.max_steps = max_steps
        self.global_step = 0

        self.cls_id = self._required_token_id("bos_token_id", "<s>")
        self.pad_id = self._required_token_id("pad_token_id", "<pad>")
        self.mask_id = self._required_token_id("mask_token_id", "<mask>")
        self.vocab_size = len(tokenizer)
        self.masking_strategy = SpanMaskingStrategy(n_special_tokens, mask_random_p, mask_keep_p, self.vocab_size, self.mask_id)

        self.segments = self._load_segments(max_lines_per_file)
        self.examples: list[dict[str, torch.Tensor]] = []
        self.set_epoch(0)

    def _required_token_id(self, attr: str, token: str) -> int:
        value = getattr(self.tokenizer, attr, None)
        if value is not None:
            return int(value)
        converted = self.tokenizer.convert_tokens_to_ids(token)
        if converted is None or converted == self.tokenizer.unk_token_id:
            raise ValueError(f"Tokenizer must define {attr} or token {token!r}.")
        return int(converted)

    def _load_segments(self, max_lines_per_file: int | None) -> list[torch.Tensor]:
        files = sorted(self.data_dir.glob("*.train.txt"))
        if not files:
            raise FileNotFoundError(f"No *.train.txt files found in {self.data_dir}")

        max_segment_tokens = self.seq_len - 2
        if max_segment_tokens < 2:
            raise ValueError("seq_len must be at least 4 for official-style segment packing.")

        segments: list[torch.Tensor] = []
        for path in files:
            n_lines = 0
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    ids = self.tokenizer(line, add_special_tokens=False)["input_ids"]
                    for offset in range(0, len(ids), max_segment_tokens):
                        piece = ids[offset : offset + max_segment_tokens]
                        if len(piece) > 1:
                            segments.append(torch.tensor(piece, dtype=torch.long))
                    n_lines += 1
                    if max_lines_per_file is not None and n_lines >= max_lines_per_file:
                        break

        if not segments:
            raise ValueError(f"No tokenized segments were built from {self.data_dir}")
        return segments

    def set_global_step(self, global_step: int) -> None:
        self.global_step = global_step

    def set_epoch(self, epoch: int) -> None:
        rng = random.Random(self.seed + epoch)
        indices = list(range(len(self.segments)))
        rng.shuffle(indices)

        n_bert = len(indices) * self.hybrid_numerator // self.hybrid_denominator
        bert_indices = indices[:n_bert]
        gpt_indices = indices[n_bert:]

        examples = []
        examples.extend(self._pack_mode(bert_indices, BERT_MODE))
        examples.extend(self._pack_mode(gpt_indices, GPT_MODE))
        rng.shuffle(examples)
        self.examples = examples

    def _pack_mode(self, segment_indices: list[int], mode: int) -> list[dict[str, torch.Tensor]]:
        examples: list[dict[str, torch.Tensor]] = []
        current: list[torch.Tensor] = []
        current_len = 0

        for idx in segment_indices:
            segment = self.segments[idx]
            needed = segment.numel() + 1
            if current and current_len + needed > self.seq_len + 1:
                examples.append(self._build_example(current, mode))
                current = []
                current_len = 0
            current.append(segment)
            current_len += needed

        if current:
            examples.append(self._build_example(current, mode))
        return examples

    def _build_example(self, segments: list[torch.Tensor], mode: int) -> dict[str, torch.Tensor]:
        input_parts: list[torch.Tensor] = []
        target_parts: list[torch.Tensor] = []
        block_sizes: list[int] = []
        mask_p_values: list[float] = []

        for segment in segments:
            if mode == BERT_MODE:
                masked_input, target, real_mask_p = self._apply_mask(segment)
                segment_input = masked_input
                segment_target = target
                mask_p_values.append(float(real_mask_p))
            else:
                segment_input = segment.long()
                segment_target = segment.long()

            input_parts.extend([torch.tensor([self.cls_id], dtype=torch.long), segment_input])
            target_parts.extend([torch.tensor([-100], dtype=torch.long), segment_target])
            block_sizes.append(segment.numel() + 1)

        full_input = torch.cat(input_parts)
        full_target = torch.cat(target_parts)
        full_mask = torch.block_diag(*[torch.ones(size, size, dtype=torch.bool) for size in block_sizes])

        padding_length = self.seq_len - full_input.size(0) + 1
        if padding_length > 0:
            full_input = torch.cat([full_input, torch.full((padding_length,), self.pad_id, dtype=torch.long)])
            full_target = torch.cat([full_target, torch.full((padding_length,), -100, dtype=torch.long)])
            full_mask = torch.block_diag(full_mask, torch.zeros(padding_length, padding_length, dtype=torch.bool))

        if mode == GPT_MODE:
            full_mask = full_mask.tril()
        attention_mask = ~full_mask

        return {
            "input_ids": full_input[:-1].contiguous(),
            "labels": full_target[1:].contiguous(),
            "attention_mask": attention_mask[:-1, :-1].contiguous(),
            "mode": torch.tensor(mode, dtype=torch.long),
            "mask_p": torch.tensor(sum(mask_p_values) / max(1, len(mask_p_values)), dtype=torch.float),
        }

    def _apply_mask(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_ratios, replacement_ids = self.masking_strategy(input_ids)
        progress = min(1.0, self.global_step / max(1, self.max_steps))
        target_mask_p = self.mask_p_start + (self.mask_p_end - self.mask_p_start) * progress
        n_to_mask = max(1, int(mask_ratios.size(0) * target_mask_p + torch.rand(1).item()))
        threshold = torch.topk(mask_ratios, n_to_mask, largest=False).values.max().item()

        mask = mask_ratios <= threshold
        labels = torch.where(mask, input_ids, -100)
        corrupted = torch.where(mask, replacement_ids, input_ids)
        real_mask_p = mask.sum() / mask_ratios.numel()
        return corrupted, labels, real_mask_p

    def mode_counts(self) -> tuple[int, int]:
        bert = sum(1 for item in self.examples if int(item["mode"]) == BERT_MODE)
        return len(self.examples) - bert, bert

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.examples[idx]


class HybridRatioBatchSampler:
    def __init__(
        self,
        dataset: OfficialStyleTextDataset,
        batch_size: int,
        hybrid_numerator: int,
        hybrid_denominator: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if batch_size % hybrid_denominator != 0:
            raise ValueError(
                "For exact official-style hybrid batches, batch_size must be divisible by "
                f"hybrid_denominator. Got batch_size={batch_size}, hybrid_denominator={hybrid_denominator}."
            )
        self.dataset = dataset
        self.batch_size = batch_size
        self.bert_per_batch = batch_size * hybrid_numerator // hybrid_denominator
        self.gpt_per_batch = batch_size - self.bert_per_batch
        if self.bert_per_batch <= 0 or self.gpt_per_batch <= 0:
            raise ValueError("hybrid ratio must leave at least one BERT and one GPT row per batch.")
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _build_batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + 10_000 + self.epoch)
        bert_indices = [idx for idx, item in enumerate(self.dataset.examples) if int(item["mode"]) == BERT_MODE]
        gpt_indices = [idx for idx, item in enumerate(self.dataset.examples) if int(item["mode"]) == GPT_MODE]
        rng.shuffle(bert_indices)
        rng.shuffle(gpt_indices)

        n_batches = min(len(bert_indices) // self.bert_per_batch, len(gpt_indices) // self.gpt_per_batch)
        batches: list[list[int]] = []
        for batch_idx in range(n_batches):
            b_start = batch_idx * self.bert_per_batch
            g_start = batch_idx * self.gpt_per_batch
            batch = (
                bert_indices[b_start : b_start + self.bert_per_batch]
                + gpt_indices[g_start : g_start + self.gpt_per_batch]
            )
            rng.shuffle(batch)
            batches.append(batch)

        rng.shuffle(batches)
        if self.world_size > 1:
            n_even = (len(batches) // self.world_size) * self.world_size
            batches = batches[:n_even]
        return batches[self.rank :: self.world_size]

    def __iter__(self):
        yield from self._build_batches()

    def __len__(self) -> int:
        gpt_count, bert_count = self.dataset.mode_counts()
        total_batches = min(bert_count // self.bert_per_batch, gpt_count // self.gpt_per_batch)
        if self.world_size > 1:
            total_batches = (total_batches // self.world_size) * self.world_size
        return total_batches // self.world_size


def collate_fn(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in features]),
        "labels": torch.stack([x["labels"] for x in features]),
        "attention_mask": torch.stack([x["attention_mask"] for x in features]),
        "modes": torch.stack([x["mode"] for x in features]),
        "mask_p": torch.stack([x["mask_p"] for x in features]),
    }


def build_model(config_path: Path, device: torch.device) -> GPTBERTForCausalLM:
    config = ModelConfig(config_file=config_path)
    if not hasattr(config, "num_layers"):
        config.num_layers = config.num_hidden_layers
    return GPTBERTForCausalLM(config).to(device)


def build_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    no_decay = ["bias", "layer_norm"]
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    grouped = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    if args.optimizer == "lamb":
        return Lamb(grouped, lr=args.learning_rate, betas=(args.optimizer_beta1, args.optimizer_beta2), eps=args.optimizer_eps)
    if args.optimizer in {"adam", "adamw"}:
        return torch.optim.AdamW(
            grouped,
            lr=args.learning_rate,
            betas=(args.optimizer_beta1, args.optimizer_beta2),
            eps=args.optimizer_eps,
        )
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def compute_loss(model, batch: dict[str, torch.Tensor], args: argparse.Namespace):
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        causal_attention_mask=False,
        return_dict=True,
    )
    logits = outputs.logits
    labels = batch["labels"]
    modes = batch["modes"]
    bert_rows = modes.eq(BERT_MODE)
    gpt_rows = modes.eq(GPT_MODE)

    if bert_rows.any():
        bert_logits = logits[bert_rows]
        bert_labels = labels[bert_rows]
        bert_flat_logits = bert_logits.reshape(-1, bert_logits.size(-1))
        bert_gold = bert_labels.reshape(-1)
        bert_selected_logits = bert_flat_logits[bert_gold != -100]
        bert_gold = bert_gold[bert_gold != -100]
        bert_loss = F.cross_entropy(bert_selected_logits, bert_gold)
        bert_z_loss = torch.logsumexp(bert_selected_logits, dim=-1).pow(2).mean()
        with torch.no_grad():
            bert_acc = (bert_selected_logits.argmax(-1) == bert_gold).float().mean()
    else:
        bert_loss = logits.new_zeros([])
        bert_z_loss = logits.new_zeros([])
        bert_acc = logits.new_zeros([])

    if gpt_rows.any():
        gpt_logits = logits[gpt_rows]
        gpt_labels = labels[gpt_rows]
        gpt_flat_logits = gpt_logits.reshape(-1, gpt_logits.size(-1))
        gpt_gold = gpt_labels.reshape(-1)
        gpt_selected_logits = gpt_flat_logits[gpt_gold != -100]
        gpt_gold = gpt_gold[gpt_gold != -100]
        gpt_loss = F.cross_entropy(gpt_selected_logits, gpt_gold)
        gpt_z_loss = torch.logsumexp(gpt_selected_logits, dim=-1).pow(2).mean()
        with torch.no_grad():
            gpt_acc = (gpt_selected_logits.argmax(-1) == gpt_gold).float().mean()
    else:
        gpt_loss = logits.new_zeros([])
        gpt_z_loss = logits.new_zeros([])
        gpt_acc = logits.new_zeros([])

    ratio = args.hybrid_numerator / args.hybrid_denominator
    loss = ratio * bert_loss + (1.0 - ratio) * gpt_loss
    z_loss = ratio * bert_z_loss + (1.0 - ratio) * gpt_z_loss
    accuracy = ratio * bert_acc + (1.0 - ratio) * gpt_acc
    return loss, z_loss, accuracy, bert_loss.detach(), gpt_loss.detach()


def checkpoint_stem(args: argparse.Namespace, global_step: int, epoch: int) -> str:
    return f"{args.name}_step{global_step:06d}_epoch{epoch:02d}"


def save_checkpoint(
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    tokenizer,
    global_step: int,
    epoch: int,
    args: argparse.Namespace,
    main_process: bool,
) -> None:
    if not main_process:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    stem = checkpoint_stem(args, global_step, epoch)
    output_path = args.output_dir / f"{stem}.bin"
    ema_output_path = args.output_dir / f"{stem}_ema.bin"
    state_output_path = args.output_dir / f"{stem}_state_dict.bin"
    torch.save(model_to_save.state_dict(), output_path)
    torch.save(ema_model.state_dict(), ema_output_path)
    torch.save(
        {
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
            "args": namespace_to_yaml_dict(args),
        },
        state_output_path,
    )
    if args.save_hf_eval_dir:
        save_hf_eval_checkpoint(model_to_save, tokenizer, args, stem)
    (args.output_dir / "latest_checkpoint.txt").write_text(str(output_path), encoding="utf-8")
    (args.output_dir / "latest_ema_checkpoint.txt").write_text(str(ema_output_path), encoding="utf-8")
    (args.output_dir / "latest_state_checkpoint.txt").write_text(str(state_output_path), encoding="utf-8")
    print(f"saved_checkpoint={output_path} global_step={global_step} epoch={epoch}", flush=True)


def save_hf_eval_checkpoint(model_to_save: nn.Module, tokenizer, args: argparse.Namespace, stem: str) -> None:
    out = args.output_dir / f"{stem}-hf"
    out.mkdir(parents=True, exist_ok=True)
    model_to_save.save_pretrained(out, safe_serialization=False)
    state_dict = model_to_save.state_dict()
    if "lm_head.nonlinearity.5.weight" not in state_dict:
        state_dict["lm_head.nonlinearity.5.weight"] = model_to_save.lm_head.nonlinearity[-1].weight.detach().clone()
    torch.save(state_dict, out / "pytorch_model.bin")
    for stale in [out / "model.safetensors", out / "model.safetensors.index.json"]:
        if stale.exists():
            stale.unlink()
    tokenizer.save_pretrained(out)
    normalize_tokenizer_files(out)

    for path in [ROOT / "src" / "gpt_bert" / "configuration_gpt_bert.py", ROOT / "src" / "gpt_bert" / "modeling_gpt_bert.py", ROOT / "src" / "gpt_bert" / "__init__.py"]:
        path = Path(path)
        if path.exists():
            shutil.copy2(path, out / path.name)
    (args.output_dir / "latest_hf_checkpoint.txt").write_text(str(out), encoding="utf-8")


def normalize_tokenizer_files(out: Path) -> None:
    tokenizer_config_path = out / "tokenizer_config.json"
    tokenizer_config: dict[str, Any] = {}
    if tokenizer_config_path.exists():
        tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    tokenizer_config.update(
        {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "sep_token": "</s>",
            "pad_token": "<pad>",
            "cls_token": "<s>",
            "mask_token": "<mask>",
        }
    )
    tokenizer_config_path.write_text(json.dumps(tokenizer_config, indent=2), encoding="utf-8")

    special_tokens = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "sep_token": "</s>",
        "pad_token": "<pad>",
        "cls_token": "<s>",
        "mask_token": "<mask>",
    }
    (out / "special_tokens_map.json").write_text(json.dumps(special_tokens, indent=2), encoding="utf-8")


DEFAULT_TRAIN_CONFIG: dict[str, Any] = {
    "data_dir": ROOT / "data" / "babylm26_eng_clean",
    "source_model_dir": ROOT / "tokenizers" / "gpt_bert_tokenizer.json",
    "config": ROOT / "configs" / "gpt_bert_100m.json",
    "output_dir": ROOT / "checkpoints" / "gpt-bert-babylm26-10m-official",
    "name": "gpt_bert_10m_official",
    "resume_from_checkpoint": None,
    "epochs": 10,
    "seq_len": 128,
    "batch_size": 16,
    "global_batch_size": 32768,
    "batch_reduction": 4,
    "max_steps": 31_250 // 4,
    "hybrid_numerator": 15,
    "hybrid_denominator": 16,
    "learning_rate": 1.41e-2,
    "optimizer": "lamb",
    "weight_decay": 0.1,
    "optimizer_eps": 1e-8,
    "optimizer_beta1": 0.9,
    "optimizer_beta2": 0.98,
    "warmup_proportion": 0.016,
    "cooldown_proportion": 0.016,
    "lr_min_factor": 0.1,
    "ema_decay": 0.999,
    "max_gradient": 2.0,
    "z_loss_weight": 1e-4,
    "mask_p_start": 0.3,
    "mask_p_end": 0.15,
    "mask_random_p": 0.1,
    "mask_keep_p": 0.1,
    "n_special_tokens": 16,
    "mixed_precision": True,
    "save_every": 1000,
    "save_hf_eval_dir": True,
    "log_every": 100,
    "num_workers": 0,
    "seed": 42,
    "max_lines_per_file": None,
}

PATH_KEYS = {
    "data_dir",
    "source_model_dir",
    "config",
    "output_dir",
    "resume_from_checkpoint",
}


def resolve_path(value: Any) -> Path | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


def as_hf_local_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def load_tokenizer(path: Path) -> PreTrainedTokenizerFast:
    if path.is_file():
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=as_hf_local_path(path),
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            sep_token="</s>",
            pad_token="<pad>",
            cls_token="<s>",
            mask_token="<mask>",
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(as_hf_local_path(path), trust_remote_code=True)
    return tokenizer


def load_train_config(path: Path | None) -> argparse.Namespace:
    config = dict(DEFAULT_TRAIN_CONFIG)
    if path is not None:
        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Training config must be a YAML mapping: {path}")
        unknown = sorted(set(loaded) - set(DEFAULT_TRAIN_CONFIG))
        if unknown:
            raise ValueError(f"Unknown training config keys in {path}: {unknown}")
        config.update(loaded)

    for key in PATH_KEYS:
        config[key] = resolve_path(config[key])
    args = argparse.Namespace(**config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_path = str(args.output_dir / f"{args.name}.bin")
    return args


def namespace_to_yaml_dict(args: argparse.Namespace) -> dict[str, Any]:
    output = vars(args).copy()
    for key in PATH_KEYS:
        if output.get(key) is not None:
            output[key] = str(output[key])
    return output


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs > 10:
        raise ValueError("This 10M dataset may be traversed at most 10 times. Set epochs <= 10.")
    if not (0 < args.hybrid_numerator < args.hybrid_denominator):
        raise ValueError("hybrid_numerator must satisfy 0 < numerator < denominator.")
    if args.batch_size % args.hybrid_denominator != 0:
        raise ValueError("batch_size must be divisible by hybrid_denominator to preserve the exact hybrid ratio.")
    if args.max_lines_per_file is not None:
        print("warning: max_lines_per_file is set, so this is not a full 10M traversal.", flush=True)


def train(args: argparse.Namespace) -> None:
    validate_args(args)
    device, rank, world_size, main_process = setup_distributed()
    seed_everything(args.seed + rank)

    tokenizer = load_tokenizer(args.source_model_dir)

    estimated_max_steps = args.max_steps if args.max_steps is not None else 1
    dataset = OfficialStyleTextDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        seed=args.seed,
        hybrid_numerator=args.hybrid_numerator,
        hybrid_denominator=args.hybrid_denominator,
        n_special_tokens=args.n_special_tokens,
        mask_p_start=args.mask_p_start,
        mask_p_end=args.mask_p_end,
        mask_random_p=args.mask_random_p,
        mask_keep_p=args.mask_keep_p,
        max_steps=estimated_max_steps,
        max_lines_per_file=args.max_lines_per_file,
    )

    batch_sampler = HybridRatioBatchSampler(
        dataset=dataset,
        batch_size=args.batch_size,
        hybrid_numerator=args.hybrid_numerator,
        hybrid_denominator=args.hybrid_denominator,
        seed=args.seed,
        rank=rank,
        world_size=world_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    steps_per_epoch = len(dataloader)
    if steps_per_epoch == 0:
        raise ValueError("No training batches can be built. Check data size, batch_size, and hybrid ratio.")
    max_steps = args.max_steps if args.max_steps is not None else steps_per_epoch * args.epochs
    dataset.max_steps = max_steps

    model = build_model(args.config, device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[device.index], output_device=device.index)
    ema_model = copy.deepcopy(model.module if hasattr(model, "module") else model)
    for param in ema_model.parameters():
        param.requires_grad = False

    optimizer = build_optimizer(model, args)
    scheduler = cosine_schedule_with_warmup_cooldown(
        optimizer,
        int(max_steps * args.warmup_proportion),
        int(max_steps * args.cooldown_proportion),
        max_steps,
        args.lr_min_factor,
    )

    global_step = 0
    start_epoch = 1
    if args.resume_from_checkpoint is not None:
        state = torch.load(args.resume_from_checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])
        ema_model.load_state_dict(state["ema_model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        global_step = int(state["global_step"])
        start_epoch = int(state["epoch"]) + 1

    if main_process:
        resolved = args.output_dir / "resolved_train_config.yaml"
        resolved.write_text(yaml.safe_dump(namespace_to_yaml_dict(args), sort_keys=False), encoding="utf-8")
        metadata = {
            "data_dir": str(args.data_dir),
            "segments": len(dataset.segments),
            "epochs": args.epochs,
            "max_dataset_traversals": 10,
            "traversal_guard": "validate_args rejects epochs > 10 before tokenization/training; each epoch repartitions the train.txt-derived segments once, with no random filler segment reuse.",
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "per_batch_bert": batch_sampler.bert_per_batch,
            "per_batch_gpt": batch_sampler.gpt_per_batch,
            "hybrid_ratio": f"{args.hybrid_numerator}/{args.hybrid_denominator}",
            "world_size": world_size,
            "steps_per_epoch": steps_per_epoch,
            "max_steps": max_steps,
            "attention_mask": "official block-diagonal bool mask where True means masked/disallowed; causal examples additionally use lower-triangular visibility",
            "checkpoint_format": "official .bin files plus optional HuggingFace eval directory when save_hf_eval_dir=true",
        }
        (args.output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        n_params = sum(p.numel() for p in (model.module if hasattr(model, "module") else model).parameters() if p.requires_grad)
        print(f"device={device} world_size={world_size} params={n_params:,}")
        print(f"segments={len(dataset.segments)} steps_per_epoch={steps_per_epoch} max_steps={max_steps}")
        print(f"hybrid_ratio={args.hybrid_numerator}/{args.hybrid_denominator} batch_bert={batch_sampler.bert_per_batch} batch_gpt={batch_sampler.gpt_per_batch}")
        print("epoch_guard=epochs<=10 enforced before training; each epoch consumes each train.txt segment once as a primary segment")

    scaler = torch.amp.GradScaler("cuda", enabled=args.mixed_precision and device.type == "cuda")
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            dataset.set_epoch(epoch)
            batch_sampler.set_epoch(epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)

            for local_step, batch in enumerate(dataloader, start=1):
                dataset.set_global_step(global_step)
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                with torch.amp.autocast("cuda", enabled=args.mixed_precision and device.type == "cuda", dtype=torch.bfloat16):
                    loss, z_loss, accuracy, bert_loss, gpt_loss = compute_loss(model, batch, args)
                    train_loss = loss + args.z_loss_weight * z_loss

                scaler.scale(train_loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    model_for_ema = model.module if hasattr(model, "module") else model
                    for param_q, param_k in zip(model_for_ema.parameters(), ema_model.parameters()):
                        param_k.data.mul_(args.ema_decay).add_((1.0 - args.ema_decay) * param_q.detach().data)

                metrics = torch.stack([
                    loss.detach(),
                    z_loss.detach(),
                    accuracy.detach(),
                    bert_loss.detach(),
                    gpt_loss.detach(),
                    grad_norm.detach() if isinstance(grad_norm, torch.Tensor) else torch.tensor(float(grad_norm), device=device),
                    batch["mask_p"].mean().detach(),
                ])
                metrics = reduce_mean(metrics)

                global_step += 1
                if main_process and global_step % args.log_every == 0:
                    print(
                        f"epoch={epoch} global_step={global_step} "
                        f"loss={metrics[0].item():.4f} z_loss={metrics[1].item():.4f} "
                        f"acc={metrics[2].item() * 100:.2f} bert_loss={metrics[3].item():.4f} "
                        f"gpt_loss={metrics[4].item():.4f} grad_norm={metrics[5].item():.4f} "
                        f"mask_p={metrics[6].item():.4f} lr={optimizer.param_groups[0]['lr']:.6g}",
                        flush=True,
                    )

                if args.save_every > 0 and global_step % args.save_every == 0:
                    save_checkpoint(model, ema_model, optimizer, scheduler, tokenizer, global_step, epoch, args, main_process)

                if global_step >= max_steps:
                    save_checkpoint(model, ema_model, optimizer, scheduler, tokenizer, global_step, epoch, args, main_process)
                    return

            save_checkpoint(model, ema_model, optimizer, scheduler, tokenizer, global_step, epoch, args, main_process)
    finally:
        cleanup_distributed()


def main() -> None:
    parser = argparse.ArgumentParser(description="Official-style GPT-BERT 10M training from train.txt with .bin checkpoints.")
    parser.add_argument("--train-config", type=Path, default=ROOT / "configs" / "gpt_bert_trainer_8gpu.yaml")
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    cli_args = parser.parse_args()
    args = load_train_config(cli_args.train_config)
    if cli_args.resume_from_checkpoint is not None:
        args.resume_from_checkpoint = resolve_path(cli_args.resume_from_checkpoint)
    train(args)


if __name__ == "__main__":
    main()
