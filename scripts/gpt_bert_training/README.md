# GPT-BERT 10M Training On BabyLM26 ENG Clean Data

本目录用于训练 GPT-BERT baseline，数据来自今年 ENG 清洗数据：

```text
data/babylm26_eng_clean
```

这 6 个文件合计约 7.48M words，属于 10M / Strict-Small 级别。训练最多允许遍历 10 遍，脚本会强制：

```text
--epochs <= 10
```

## 训练目标分配

每个 epoch 开始时，脚本会随机打乱 packed examples，并将其中一半分配给：

```text
GPT causal next-token prediction
```

另一半分配给：

```text
BERT-like masked continuation
```

具体输出会打印：

```text
epoch=1 assignment_gpt=... assignment_bert=...
```

每个 epoch 都会重新随机分配，因此同一个样本在不同 epoch 可能进入不同训练分支。

batch 内部也强制均衡：如果 `--batch-size 16`，每个 batch 固定包含：

```text
8 GPT examples + 8 BERT-like examples
```

## 默认训练方式

默认从随机初始化的 GPT-BERT 架构开始训练，不加载去年 checkpoint。

```bash
torchrun --nproc_per_node=1 scripts/gpt_bert_training/train_gpt_bert_10m.py \
  --data-dir data/babylm26_eng_clean \
  --epochs 10 \
  --seq-len 128 \
  --batch-size 16 \
  --grad-accum-steps 8 \
  --lr 3e-4
```

Windows 单 GPU 也可以直接运行：

```bash
python scripts/gpt_bert_training/train_gpt_bert_10m.py --epochs 10 --batch-size 16 --grad-accum-steps 8
```

## 多 GPU

例如 4 张 GPU：

```bash
torchrun --nproc_per_node=4 scripts/gpt_bert_training/train_gpt_bert_10m.py \
  --epochs 10 \
  --batch-size 16 \
  --grad-accum-steps 8
```

全局 batch tokens 约为：

```text
n_gpu * batch_size * grad_accum_steps * seq_len
```

默认 4 GPU 时：

```text
4 * 16 * 8 * 128 = 65,536 tokens/update
```

每张 GPU 的每个 micro-batch 内仍然保持：

```text
8 GPT examples + 8 BERT-like examples
```

## 可选：从去年 GPT-BERT checkpoint warm-start

默认不建议用于公平 10M baseline，但如果只是调试或继续训练，可以加：

```bash
python scripts/gpt_bert_training/train_gpt_bert_10m.py \
  --init-checkpoint-dir checkpoints/babyLM-gpt-bert-mixed \
  --epochs 1
```

## 输出

默认输出到：

```text
checkpoints/gpt-bert-babylm26-10m
```

包含：

```text
training_metadata.json
checkpoint_epochXX_stepYYYY.pt
tokenizer/
```

## 重要约束

- 不要设置 `--epochs` 大于 10。
- 不要把 `data/babylm26_eng_clean` 重复复制成更大的训练集。
- 如果要 debug，可用 `--max-lines-per-file`，但正式训练不能用。
- 当前脚本是 GPT-BERT baseline；后续 LTD 会另写入口，避免污染 baseline。
