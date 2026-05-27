from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.gpt_bert import ModelConfig, GPTBERTForCausalLM


GPT_MODE = 0
BERT_MODE = 1


@dataclass
class BatchStats:
    loss: float
    gpt_loss: float
    bert_loss: float
    gpt_rows: int
    bert_rows: int
    lr: float


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
    device = torch.device("cuda", local_rank)
    return device, rank, world_size, rank == 0


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PackedGPTBERTDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        tokenizer,
        seq_len: int,
        seed: int,
        max_lines_per_file: int | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.seed = seed
        self.examples = self._load_and_pack(max_lines_per_file)
        self.modes = [GPT_MODE] * len(self.examples)
        self.set_epoch(0)

    def _load_and_pack(self, max_lines_per_file: int | None) -> list[torch.Tensor]:
        files = sorted(self.data_dir.glob("*.train.txt"))
        if not files:
            raise FileNotFoundError(f"No *.train.txt files found in {self.data_dir}")

        eos_id = self.tokenizer.eos_token_id
        token_buffer: list[int] = []
        examples: list[torch.Tensor] = []
        chunk_len = self.seq_len + 1

        for path in files:
            n_lines = 0
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    ids = self.tokenizer(line, add_special_tokens=True)["input_ids"]
                    if eos_id is not None and ids[-1] != eos_id:
                        ids.append(eos_id)
                    token_buffer.extend(ids)

                    while len(token_buffer) >= chunk_len:
                        examples.append(torch.tensor(token_buffer[:chunk_len], dtype=torch.long))
                        del token_buffer[:chunk_len]

                    n_lines += 1
                    if max_lines_per_file is not None and n_lines >= max_lines_per_file:
                        break

        if len(token_buffer) >= 2:
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = eos_id if eos_id is not None else 0
            padded = token_buffer + [pad_id] * (chunk_len - len(token_buffer))
            examples.append(torch.tensor(padded[:chunk_len], dtype=torch.long))

        if not examples:
            raise ValueError(f"No tokenized examples were built from {self.data_dir}")
        return examples

    def set_epoch(self, epoch: int) -> None:
        rng = random.Random(self.seed + epoch)
        indices = list(range(len(self.examples)))
        rng.shuffle(indices)
        half = len(indices) // 2
        modes = [GPT_MODE] * len(self.examples)
        for idx in indices[:half]:
            modes[idx] = BERT_MODE
        self.modes = modes

    def mode_counts(self) -> tuple[int, int]:
        bert = sum(1 for m in self.modes if m == BERT_MODE)
        return len(self.modes) - bert, bert

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.examples[idx], self.modes[idx]


class BalancedModeBatchSampler:
    """Build batches with exactly half GPT rows and half BERT-like rows."""

    def __init__(self, dataset: PackedGPTBERTDataset, batch_size: int, seed: int, rank: int = 0, world_size: int = 1):
        if batch_size % 2 != 0:
            raise ValueError("Balanced GPT/BERT batches require an even --batch-size.")
        self.dataset = dataset
        self.batch_size = batch_size
        self.half_batch = batch_size // 2
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _build_batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + 10_000 + self.epoch)
        gpt_indices = [idx for idx, mode in enumerate(self.dataset.modes) if mode == GPT_MODE]
        bert_indices = [idx for idx, mode in enumerate(self.dataset.modes) if mode == BERT_MODE]
        rng.shuffle(gpt_indices)
        rng.shuffle(bert_indices)

        n_batches = min(len(gpt_indices), len(bert_indices)) // self.half_batch
        batches: list[list[int]] = []
        for batch_idx in range(n_batches):
            start = batch_idx * self.half_batch
            batch = gpt_indices[start : start + self.half_batch] + bert_indices[start : start + self.half_batch]
            rng.shuffle(batch)
            batches.append(batch)

        rng.shuffle(batches)
        # DDP requires every rank to execute the same number of forwards.
        # Drop tail batches that cannot be evenly sharded across ranks.
        if self.world_size > 1:
            n_even = (len(batches) // self.world_size) * self.world_size
            batches = batches[:n_even]
        return batches[self.rank :: self.world_size]

    def __iter__(self):
        yield from self._build_batches()

    def __len__(self) -> int:
        gpt_count, bert_count = self.dataset.mode_counts()
        total_batches = min(gpt_count, bert_count) // self.half_batch
        if self.world_size > 1:
            total_batches = (total_batches // self.world_size) * self.world_size
        return total_batches // self.world_size


def collate_examples(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    sequences, modes = zip(*batch)
    return torch.stack(sequences), torch.tensor(modes, dtype=torch.long)


def mask_bert_inputs(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    tokenizer,
    mask_prob: float,
    random_prob: float,
    keep_prob: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError("GPT-BERT tokenizer must define a mask_token_id for BERT-like training.")

    special_ids = {x for x in [tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id, mask_id] if x is not None}
    can_mask = torch.ones_like(input_ids, dtype=torch.bool)
    for token_id in special_ids:
        can_mask &= input_ids.ne(token_id)

    rand = torch.rand(input_ids.shape, device=input_ids.device)
    mask_positions = (rand < mask_prob) & can_mask

    # Ensure every BERT-like row has at least one masked token when possible.
    for row in range(mask_positions.size(0)):
        if not mask_positions[row].any() and can_mask[row].any():
            candidates = torch.nonzero(can_mask[row], as_tuple=False).flatten()
            chosen = candidates[torch.randint(candidates.numel(), (1,), device=input_ids.device)]
            mask_positions[row, chosen] = True

    corrupted = input_ids.clone()
    replacement_rand = torch.rand(input_ids.shape, device=input_ids.device)
    random_positions = mask_positions & (replacement_rand < random_prob)
    keep_positions = mask_positions & (replacement_rand >= random_prob) & (replacement_rand < random_prob + keep_prob)
    mask_token_positions = mask_positions & ~random_positions & ~keep_positions

    if random_positions.any():
        vocab_size = len(tokenizer)
        random_tokens = torch.randint(0, vocab_size, input_ids.shape, device=input_ids.device)
        corrupted = torch.where(random_positions, random_tokens, corrupted)
    corrupted = torch.where(mask_token_positions, torch.full_like(corrupted, mask_id), corrupted)

    continuation_mask = torch.cumsum(mask_positions.long(), dim=1).bool()
    bert_labels = labels.masked_fill(~continuation_mask, -100)
    return corrupted, bert_labels


def build_model(args: argparse.Namespace, device: torch.device) -> GPTBERTForCausalLM:
    config = ModelConfig(config_file=args.config)
    if not hasattr(config, "num_layers"):
        config.num_layers = config.num_hidden_layers
    model = GPTBERTForCausalLM(config)

    if args.init_checkpoint_dir is not None:
        ckpt = args.init_checkpoint_dir / "pytorch_model.bin"
        state_dict = torch.load(ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"loaded_init_checkpoint={ckpt}")
        print(f"missing_keys={len(missing)} unexpected_keys={len(unexpected)}")

    return model.to(device)


def compute_losses(
    model,
    sequences: torch.Tensor,
    modes: torch.Tensor,
    tokenizer,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    input_ids = sequences[:, :-1].contiguous()
    labels = sequences[:, 1:].contiguous()

    pad_id = tokenizer.pad_token_id
    if pad_id is not None:
        labels = labels.masked_fill(labels.eq(pad_id), -100)
        attention_mask = input_ids.ne(pad_id).long()
    else:
        attention_mask = torch.ones_like(input_ids)

    bert_rows = modes.eq(BERT_MODE)
    gpt_rows = modes.eq(GPT_MODE)

    train_inputs = input_ids.clone()
    train_labels = labels.clone()
    if bert_rows.any():
        corrupted, bert_labels = mask_bert_inputs(
            input_ids=input_ids[bert_rows],
            labels=labels[bert_rows],
            tokenizer=tokenizer,
            mask_prob=args.mask_prob,
            random_prob=args.mask_random_prob,
            keep_prob=args.mask_keep_prob,
        )
        train_inputs[bert_rows] = corrupted
        train_labels[bert_rows] = bert_labels

    outputs = model(
        input_ids=train_inputs,
        attention_mask=attention_mask,
        causal_attention_mask=gpt_rows,
        return_dict=True,
    )
    logits = outputs.logits

    total_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), train_labels.reshape(-1), ignore_index=-100)

    if gpt_rows.any():
        gpt_loss = F.cross_entropy(logits[gpt_rows].reshape(-1, logits.size(-1)), labels[gpt_rows].reshape(-1), ignore_index=-100)
    else:
        gpt_loss = torch.zeros([], device=logits.device)

    if bert_rows.any():
        bert_loss = F.cross_entropy(logits[bert_rows].reshape(-1, logits.size(-1)), train_labels[bert_rows].reshape(-1), ignore_index=-100)
    else:
        bert_loss = torch.zeros([], device=logits.device)

    combined = gpt_loss + args.lambda_bert * bert_loss if (gpt_rows.any() and bert_rows.any()) else total_loss
    return combined, gpt_loss.detach(), bert_loss.detach(), int(gpt_rows.sum().item()), int(bert_rows.sum().item())


def save_checkpoint(model, tokenizer, args: argparse.Namespace, epoch: int, global_step: int, main_process: bool) -> None:
    if not main_process:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    ckpt_path = args.output_dir / f"checkpoint_epoch{epoch:02d}_step{global_step}.pt"
    torch.save(
        {
            "model_state_dict": model_to_save.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "args": vars(args),
        },
        ckpt_path,
    )
    tokenizer.save_pretrained(args.output_dir / "tokenizer")
    print(f"saved_checkpoint={ckpt_path}", flush=True)


def train(args: argparse.Namespace) -> None:
    if args.epochs > 10:
        raise ValueError("This 10M dataset may be traversed at most 10 times. Set --epochs <= 10.")

    device, rank, world_size, main_process = setup_distributed()
    seed_everything(args.seed + rank)

    tokenizer_source = args.tokenizer_dir if args.tokenizer_dir is not None else args.init_checkpoint_dir
    if tokenizer_source is None:
        tokenizer_source = ROOT / "checkpoints" / "babyLM-gpt-bert-mixed"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)

    dataset = PackedGPTBERTDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        seed=args.seed,
        max_lines_per_file=args.max_lines_per_file,
    )

    batch_sampler = BalancedModeBatchSampler(
        dataset=dataset,
        batch_size=args.batch_size,
        seed=args.seed,
        rank=rank,
        world_size=world_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_examples,
    )

    model = build_model(args, device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[device.index], output_device=device.index)

    no_decay = ["bias", "layer_norm"]
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay_params if any(nd in name for nd in no_decay) else decay_params).append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )

    steps_per_epoch = math.ceil(len(dataloader) / args.grad_accum_steps)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.amp.GradScaler("cuda", enabled=args.mixed_precision and device.type == "cuda")
    global_step = 0

    if main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "data_dir": str(args.data_dir),
            "examples": len(dataset),
            "epochs": args.epochs,
            "max_total_traversals": 10,
            "world_size": world_size,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "train_rule": "each epoch randomly assigns exactly half packed examples to GPT and half to BERT-like masked continuation; each batch is balanced with batch_size/2 GPT rows and batch_size/2 BERT rows; GPT rows use causal attention and BERT rows use bidirectional attention",
        }
        (args.output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        n_params = sum(p.numel() for p in (model.module if hasattr(model, "module") else model).parameters())
        print(f"device={device} world_size={world_size}")
        print(f"data_dir={args.data_dir}")
        print(f"examples={len(dataset)} seq_len={args.seq_len} params={n_params:,}")
        print(f"batch_size={args.batch_size} per_batch_gpt={args.batch_size // 2} per_batch_bert={args.batch_size // 2}")
        print(f"epochs={args.epochs} total_steps={total_steps} warmup_steps={warmup_steps}")

    for epoch in range(1, args.epochs + 1):
        dataset.set_epoch(epoch)
        batch_sampler.set_epoch(epoch)
        gpt_count, bert_count = dataset.mode_counts()
        if main_process:
            print(f"epoch={epoch} assignment_gpt={gpt_count} assignment_bert={bert_count}")

        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = []

        for step, (sequences, modes) in enumerate(dataloader, start=1):
            sequences = sequences.to(device, non_blocking=True)
            modes = modes.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=args.mixed_precision and device.type == "cuda", dtype=torch.bfloat16):
                loss, gpt_loss, bert_loss, gpt_rows, bert_rows = compute_losses(model, sequences, modes, tokenizer, args)
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()

            if step % args.grad_accum_steps != 0:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            stat = BatchStats(
                loss=float(loss.item() * args.grad_accum_steps),
                gpt_loss=float(gpt_loss.item()),
                bert_loss=float(bert_loss.item()),
                gpt_rows=gpt_rows,
                bert_rows=bert_rows,
                lr=optimizer.param_groups[0]["lr"],
            )
            running.append(stat)

            if main_process and global_step % args.log_every == 0:
                window = running[-args.log_every :]
                avg_loss = sum(x.loss for x in window) / len(window)
                avg_gpt = sum(x.gpt_loss for x in window) / len(window)
                avg_bert = sum(x.bert_loss for x in window) / len(window)
                print(
                    f"epoch={epoch} global_step={global_step} "
                    f"loss={avg_loss:.4f} gpt_loss={avg_gpt:.4f} bert_loss={avg_bert:.4f} "
                    f"last_batch_gpt_rows={stat.gpt_rows} last_batch_bert_rows={stat.bert_rows} lr={stat.lr:.6g}",
                    flush=True,
                )

            if args.save_every > 0 and global_step % args.save_every == 0:
                save_checkpoint(model, tokenizer, args, epoch, global_step, main_process)

        save_checkpoint(model, tokenizer, args, epoch, global_step, main_process)

    cleanup_distributed()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GPT-BERT on BabyLM26 ENG 10M cleaned data.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "babylm26_eng_clean")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "gpt_bert_100m.json")
    parser.add_argument("--tokenizer-dir", type=Path, default=ROOT / "checkpoints" / "babyLM-gpt-bert-mixed")
    parser.add_argument("--init-checkpoint-dir", type=Path, default=None, help="Optional warm-start checkpoint directory containing pytorch_model.bin.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints" / "gpt-bert-babylm26-10m")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lambda-bert", type=float, default=1.0)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--mask-random-prob", type=float, default=0.1)
    parser.add_argument("--mask-keep-prob", type=float, default=0.1)
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-lines-per-file", type=int, default=None, help="Debug only. Do not use for full training.")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
