# BabyLM 2026 GPT2 Strict-Small 下载说明

这个 baseline 与 GPT-BERT 项目分开保存。

模型仓库：

```text
BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict-Small
```

本地建议目录：

```text
checkpoints/gpt2-strict-small
```

## 登录 HuggingFace

```bash
huggingface-cli login
```

## Windows CMD 下载

```bat
set HF_HUB_DISABLE_XET=1
huggingface-cli download BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict-Small --local-dir checkpoints\gpt2-strict-small
```

如果国内网络卡住，可以用镜像：

```bat
set HF_ENDPOINT=https://hf-mirror.com
set HF_HUB_DISABLE_XET=1
huggingface-cli download BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict-Small --local-dir checkpoints\gpt2-strict-small
```

## PowerShell 下载

```powershell
$env:HF_HUB_DISABLE_XET="1"
huggingface-cli download BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict-Small --local-dir checkpoints\gpt2-strict-small
```

镜像：

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
$env:HF_HUB_DISABLE_XET="1"
huggingface-cli download BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict-Small --local-dir checkpoints\gpt2-strict-small
```

## 应有文件

```text
checkpoints/gpt2-strict-small/config.json
checkpoints/gpt2-strict-small/generation_config.json
checkpoints/gpt2-strict-small/model.safetensors
checkpoints/gpt2-strict-small/tokenizer.json
checkpoints/gpt2-strict-small/tokenizer_config.json
checkpoints/gpt2-strict-small/special_tokens_map.json
```
