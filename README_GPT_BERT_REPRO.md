# GPT-BERT 架构复现

本目录已经按 HuggingFace baseline `BabyLM-community/babylm-baseline-100m-gpt-bert-mixed` 写好本地 GPT-BERT 架构复现入口。

## 文件说明

- `src/gpt_bert/configuration_gpt_bert.py`：GPT-BERT 配置类，来自 HuggingFace baseline。
- `src/gpt_bert/modeling_gpt_bert.py`：GPT-BERT 模型结构，包含 shared decoder-only backbone、causal LM head、masked LM head。
- `configs/gpt_bert_100m.json`：100M baseline 配置，12 层、hidden size 768、12 heads、vocab 16384。
- `tokenizers/gpt_bert_tokenizer.json`：baseline tokenizer。
- `scripts/check_gpt_bert_architecture.py`：本地随机输入检查，用于验证架构 forward/backward。
- `scripts/gpt_bert_training/`：GPT-BERT 训练脚本与训练说明。
- `reference/`：保留官方 HuggingFace 与 `ltgoslo/gpt-bert` 参考代码，便于后续扩展训练。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 1. 检查原始 100M 架构

只实例化原始 GPT-BERT 100M 架构并打印参数规模：

```bash
python scripts/check_gpt_bert_architecture.py --mode causal
```

检查 masked LM 包装类：

```bash
python scripts/check_gpt_bert_architecture.py --mode masked
```

如果机器内存/显存足够，可以额外做一次 forward：

```bash
python scripts/check_gpt_bert_architecture.py --mode causal --forward --device cuda --batch-size 1 --seq-len 128
```

本仓库不包含 tiny 架构；`configs/gpt_bert_100m.json` 与 HuggingFace baseline 的 100M GPT-BERT 配置保持一致。

## 2. 架构要点

GPT-BERT 使用同一套 backbone 支持两类接口：

- `GPTBERTForCausalLM`：causal attention，用于 GPT 风格 next-token prediction。
- `GPTBERTForMaskedLM`：非 causal attention，用于 masked inference。

核心结构包括：

- word embedding + relative position embedding；
- 多层 Attention + GeGLU FeedForward；
- DWA（Dense Weighted Average）跨层融合；
- tied embedding LM head。

## 3. 后续训练说明

官方 `ltgoslo/gpt-bert` 仓库的训练脚本依赖 SLURM + 多 GPU + DDP，且 README 写明单 GPU 训练脚本尚未完成。本仓库的训练入口统一放在 `scripts/gpt_bert_training/`。

当前训练脚本已经覆盖：

- GPT causal dataset；
- BERT-like masked continuation dataset；
- 50/50 mixed batch；
- AdamW + warmup/cosine schedule；
- HuggingFace Trainer checkpoint。
