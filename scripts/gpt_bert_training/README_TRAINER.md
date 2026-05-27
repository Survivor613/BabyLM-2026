# Official-Style GPT-BERT 10M Training

Script:

```text
scripts/gpt_bert_training/train_gpt_bert_10m_trainer.py
```

Despite the historical filename, this is no longer a HuggingFace `Trainer` script. It uses an official-style custom loop:

- train.txt tokenization into segments
- span masking with a 0.30 -> 0.15 mask schedule
- block-diagonal official attention masks
- configurable hybrid BERT/GPT ratio from YAML
- LAMB, warmup/cooldown cosine schedule, z-loss, EMA
- official `.bin` checkpoint files

## Single GPU

```bash
python scripts/gpt_bert_training/train_gpt_bert_10m_trainer.py \
  --train-config configs/gpt_bert_trainer_1gpu.yaml
```

## Eight GPUs

```bash
torchrun --nproc_per_node=8 scripts/gpt_bert_training/train_gpt_bert_10m_trainer.py \
  --train-config configs/gpt_bert_trainer_8gpu.yaml
```

## Epoch Guard

The script rejects `epochs > 10` before tokenization or training starts. Each epoch repartitions the train.txt-derived segments once, without random filler segment reuse, so the primary dataset traversal count is bounded by `epochs`.

## Checkpoints

For `name: gpt_bert_10m_official_1gpu`, checkpoints are:

```text
checkpoints/gpt-bert-babylm26-10m-official/gpt_bert_10m_official_1gpu.bin
checkpoints/gpt-bert-babylm26-10m-official/gpt_bert_10m_official_1gpu_ema.bin
checkpoints/gpt-bert-babylm26-10m-official/gpt_bert_10m_official_1gpu_state_dict.bin
```
