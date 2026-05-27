# GPT-BERT Experiment Log

## Run Configs

| Run name | Train config | Output dir | Ratio | Tokenizer | Model | Params | LR | Epochs | Max steps | Status |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `gpt_bert_10m_official_33m` | `configs/gpt_bert_trainer_8gpu.yaml` | `/home/babylm26_g2/data/BabyLM/checkpoints/gpt-bert/gpt-bert-10m-official_33m_parameter` | 15/16 BERT, 1/16 GPT | `tokenizers/hf_10m_gpt_bert_mixed` | `configs/gpt_bert_small_16k.json` | 33,046,084 | 0.0141 | 10 | 3940 | trained, eval e10 done |
| `gpt_bert_10m_ratio_3to1_33m` | `configs/gpt_bert_trainer_8gpu_ratio_3to1.yaml` | `/home/babylm26_g2/data/BabyLM/checkpoints/gpt-bert/gpt-bert-10m-ratio_3to1_33m_parameter` | 3/4 BERT, 1/4 GPT | `tokenizers/hf_10m_gpt_bert_mixed` | `configs/gpt_bert_small_16k.json` | 33,046,084 | 0.0141 | 10 | TBD | planned |
| `gpt_bert_10m_ratio_1to1_33m` | `configs/gpt_bert_trainer_8gpu_ratio_1to1.yaml` | `/home/babylm26_g2/data/BabyLM/checkpoints/gpt-bert/gpt-bert-10m-ratio_1to1_33m_parameter` | 1/2 BERT, 1/2 GPT | `tokenizers/hf_10m_gpt_bert_mixed` | `configs/gpt_bert_small_16k.json` | 33,046,084 | 0.0141 | 10 | TBD | planned |
| `gpt_bert_10m_ratio_1to3_33m` | `configs/gpt_bert_trainer_8gpu_ratio_1to3.yaml` | `/home/babylm26_g2/data/BabyLM/checkpoints/gpt-bert/gpt-bert-10m-ratio_1to3_33m_parameter` | 1/4 BERT, 3/4 GPT | `tokenizers/hf_10m_gpt_bert_mixed` | `configs/gpt_bert_small_16k.json` | 33,046,084 | 0.0141 | 10 | TBD | planned |

## Training Curve: `gpt_bert_10m_official_33m`

| Epoch | Step | Checkpoint stem | Loss | Acc | BERT loss | GPT loss | Grad norm | Mask p | LR |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 398 | `gpt_bert_10m_official_33m_step000398_epoch01` | 4.6940 | 27.49 | 4.6508 | 5.3412 | 0.4032 | 0.3142 | 0.0139834 |
| 2 | 796 | `gpt_bert_10m_official_33m_step000796_epoch02` | 4.4203 | 30.63 | 4.3889 | 4.8913 | 0.4078 | 0.2966 | 0.0132733 |
| 3 | 1000 | `gpt_bert_10m_official_33m_step001000_epoch03` | 4.4313 | 30.02 | 4.4017 | 4.8757 | 0.8632 | 0.2824 | 0.0123573 |
| 3 | 1194 | `gpt_bert_10m_official_33m_step001194_epoch03` | 4.2427 | 30.60 | 4.2195 | 4.5904 | 0.5070 | 0.2805 | 0.0119886 |
| 4 | 1591 | `gpt_bert_10m_official_33m_step001591_epoch04` | 4.4213 | 29.69 | 4.4014 | 4.7211 | 2.3284 | 0.2669 | 0.0102631 |
| 5 | 1989 | `gpt_bert_10m_official_33m_step001989_epoch05` | 4.5495 | 27.60 | 4.5204 | 4.9857 | 3.0380 | 0.2539 | 0.00827627 |
| 6 | 2000 | `gpt_bert_10m_official_33m_step002000_epoch06` | 4.5754 | 27.59 | 4.5407 | 5.0964 | 2.6827 | 0.2390 | 0.00776271 |
| 6 | 2385 | `gpt_bert_10m_official_33m_step002385_epoch06` | 4.5090 | 28.14 | 4.4730 | 5.0493 | 3.8596 | 0.2389 | 0.0062352 |
| 7 | 2779 | `gpt_bert_10m_official_33m_step002779_epoch07` | 4.3610 | 29.79 | 4.3214 | 4.9558 | 5.6427 | 0.2243 | 0.0043524 |
| 8 | 3000 | `gpt_bert_10m_official_33m_step003000_epoch08` | 4.3516 | 31.13 | 4.3351 | 4.5995 | 5.3210 | 0.2100 | 0.00316334 |
| 8 | 3176 | `gpt_bert_10m_official_33m_step003176_epoch08` | 4.2188 | 31.80 | 4.1849 | 4.7264 | 1.6958 | 0.2121 | 0.00282395 |
| 9 | 3571 | `gpt_bert_10m_official_33m_step003571_epoch09` | 4.0865 | 32.11 | 4.0494 | 4.6432 | 4.0058 | 0.2007 | 0.00180903 |
| 10 | 3940 | `gpt_bert_10m_official_33m_step003940_epoch10` | 4.0039 | 33.42 | 3.9749 | 4.4391 | 1.6879 | 0.1864 | 0.000895238 |

Best logged point before checkpoint save:

| Epoch | Step | Loss | Acc | BERT loss | GPT loss | Mask p | LR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 3700 | 3.9611 | 34.37 | 3.9293 | 4.4384 | 0.1846 | 0.00152961 |

## Zero-Shot Eval: `gpt_bert_10m_official_33m`

Checkpoint: `gpt_bert_10m_official_33m_step003940_epoch10-hf`  
Revision: `e10`; backend: `causal`

| Task | This run | GPT-2 Strict baseline | Diff vs Strict | GPT-2 Strict-Small baseline | Diff vs Strict-Small |
| --- | ---: | ---: | ---: | ---: | ---: |
| `zero_shot/blimp/blimp_filtered` | 59.93 | 74.53 | -14.60 | 65.08 | -5.15 |
| `zero_shot/blimp/supplement_filtered` | 54.42 | 65.00 | -10.58 | 57.25 | -2.83 |
| `zero_shot/comps/comps` | 50.69 | 55.85 | -5.16 | 51.81 | -1.12 |
| `zero_shot/entity_tracking/entity_tracking` | 33.01 | 23.58 | +9.43 | 21.07 | +11.94 |

## Finetune Eval: `gpt_bert_10m_official_33m`

Checkpoint: `gpt_bert_10m_official_8gpu_step003940_epoch10-hf`  
Script: `scripts/eval_finetuning.sh`; LR `3e-5`; batch size `32`; boolq/multirc batch size `16`; max epochs `10`; seed `42`

| Task | Metric | This run | GPT-2 Strict baseline | Diff vs Strict | GPT-2 Strict-Small baseline | Diff vs Strict-Small | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `boolq` | accuracy | 64.28 | 67.46 | -3.18 | 65.87 | -1.59 | completed from full finetune script |
| `mnli` | accuracy | TBD | 59.94 | TBD | 49.80 | TBD | pending |
| `mrpc` | f1 | TBD | 84.35 | TBD | 83.49 | TBD | pending |
| `multirc` | accuracy | TBD | 63.90 | TBD | 64.52 | TBD | pending |
| `qqp` | f1 | TBD | 70.73 | TBD | 60.86 | TBD | pending |
| `rte` | accuracy | TBD | 56.83 | TBD | 60.43 | TBD | pending |

Legacy note: the first training run originally used checkpoint stems beginning with `gpt_bert_10m_official_8gpu`; after renaming, future runs should use the YAML `name` value `gpt_bert_10m_official_33m`.
