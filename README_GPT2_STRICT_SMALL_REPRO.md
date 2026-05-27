# BabyLM 2026 Baseline GPT2 Strict-Small 复现

该部分与 GPT-BERT/LTD 项目分开，目录和脚本均单独放置。

## 来源

HuggingFace 模型仓库：

```text
https://huggingface.co/BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict-Small
```

## 本地目录

```text
checkpoints/gpt2-strict-small
scripts/gpt2_baseline
```

## 架构配置

该模型是标准 Transformers `GPT2LMHeadModel`，不是 GPT-BERT 自定义架构。

关键配置：

```text
model_type=gpt2
vocab_size=16384
n_positions=1024
n_ctx=1024
n_embd=768
n_layer=12
n_head=12
activation_function=gelu_new
```

## 下载 checkpoint

见：

```text
scripts/gpt2_baseline/download_gpt2_strict_small.md
```

推荐命令：

```bash
huggingface-cli download BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict-Small --local-dir checkpoints/gpt2-strict-small
```

## 检查架构

```bash
python scripts/gpt2_baseline/check_gpt2_architecture.py --model-dir checkpoints/gpt2-strict-small
```

如果要做一次随机 forward：

```bash
python scripts/gpt2_baseline/check_gpt2_architecture.py --model-dir checkpoints/gpt2-strict-small --forward
```

## 查看 checkpoint 生成效果

```bash
python scripts/gpt2_baseline/demo_gpt2_completion.py --model-dir checkpoints/gpt2-strict-small --text "I love NLP very much"
```

贪心生成：

```bash
python scripts/gpt2_baseline/demo_gpt2_completion.py --model-dir checkpoints/gpt2-strict-small --text "Once upon a time" --greedy
```

## 与 GPT-BERT 的隔离

GPT-BERT 相关文件仍在：

```text
src/gpt_bert
configs/gpt_bert_100m.json
checkpoints/babyLM-gpt-bert-mixed
scripts/demo_gpt_bert_completion.py
```

GPT2 baseline 相关文件在：

```text
checkpoints/gpt2-strict-small
scripts/gpt2_baseline
README_GPT2_STRICT_SMALL_REPRO.md
```
