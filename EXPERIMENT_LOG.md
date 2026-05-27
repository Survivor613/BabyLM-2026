# GPT-BERT Experiment Log

## Run Configs

| Run name | Train config | Output dir | Ratio | Tokenizer | Model | Params | LR | Epochs | Max steps | Status |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `gpt_bert_10m_official_33m` | `configs/gpt_bert_trainer_8gpu.yaml` | `/home/babylm26_g2/data/BabyLM/checkpoints/gpt-bert/gpt-bert-10m-official_33m_parameter` | 15/16 BERT, 1/16 GPT | `tokenizers/hf_10m_gpt_bert_mixed` | `configs/gpt_bert_small_16k.json` | 33,046,084 | 0.0141 | 10 | 3940 | trained, eval e10 done |
| `gpt_bert_10m_ratio_3to1_33m` | `configs/gpt_bert_trainer_8gpu_ratio_3to1.yaml` | `/home/babylm26_g2/data/BabyLM/checkpoints/gpt-bert/gpt-bert-10m-ratio_3to1_33m_parameter` | 3/4 BERT, 1/4 GPT | `tokenizers/hf_10m_gpt_bert_mixed` | `configs/gpt_bert_small_16k.json` | 33,046,084 | 0.0141 | 10 | TBD | planned |
| `gpt_bert_10m_ratio_1to1_33m` | `configs/gpt_bert_trainer_8gpu_ratio_1to1.yaml` | `/home/babylm26_g2/data/BabyLM/checkpoints/gpt-bert/gpt-bert-10m-ratio_1to1_33m_parameter` | 1/2 BERT, 1/2 GPT | `tokenizers/hf_10m_gpt_bert_mixed` | `configs/gpt_bert_small_16k.json` | 33,046,084 | 0.0141 | 10 | 3970 | trained, eval e10_ratio_1to1 done |
| `gpt_bert_10m_ratio_1to1_33m_muon` | `configs/gpt_bert_trainer_8gpu_ratio_1to1_muon.yaml` | `/home/babylm26_g2/data/BabyLM/checkpoints/gpt-bert/gpt-bert-10m-ratio_1to1_33m_muon` | 1/2 BERT, 1/2 GPT | `tokenizers/hf_10m_gpt_bert_mixed` | `configs/gpt_bert_small_16k.json` | 33,046,084 | Muon 0.02 / AdamW 3e-4 | 10 | TBD | planned |
| `gpt_bert_10m_ratio_1to3_33m` | `configs/gpt_bert_trainer_8gpu_ratio_1to3.yaml` | `/home/babylm26_g2/data/BabyLM/checkpoints/gpt-bert/gpt-bert-10m-ratio_1to3_33m_parameter` | 1/4 BERT, 3/4 GPT | `tokenizers/hf_10m_gpt_bert_mixed` | `configs/gpt_bert_small_16k.json` | 33,046,084 | 0.0141 | 10 | 3970 | trained, eval e10_ratio_1to3 done |

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

## Training Curve: `gpt_bert_10m_ratio_1to1_33m`

| Epoch | Step | Checkpoint stem | Loss | Acc | BERT loss | GPT loss | Grad norm | Mask p | LR |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 398 | `gpt_bert_10m_ratio_1to1_33m_step000398_epoch01` | 4.6045 | 26.34 | 4.6284 | 4.5807 | 0.3622 | 0.1652 | 0.0139851 |
| 2 | 796 | `gpt_bert_10m_ratio_1to1_33m_step000796_epoch02` | 4.3739 | 27.66 | 4.3027 | 4.4451 | 0.3657 | 0.1583 | 0.0132857 |
| 3 | 1000 | `gpt_bert_10m_ratio_1to1_33m_step001000_epoch03` | 4.3890 | 27.14 | 4.3412 | 4.4368 | 0.3625 | 0.1550 | 0.0123827 |
| 3 | 1193 | `gpt_bert_10m_ratio_1to1_33m_step001193_epoch03` | 4.3106 | 28.75 | 4.2580 | 4.3633 | 0.3561 | 0.1512 | 0.0120191 |
| 4 | 1590 | `gpt_bert_10m_ratio_1to1_33m_step001590_epoch04` | 4.3851 | 27.41 | 4.3552 | 4.4149 | 0.5649 | 0.1449 | 0.0103151 |
| 5 | 1988 | `gpt_bert_10m_ratio_1to1_33m_step001988_epoch05` | 4.5575 | 26.08 | 4.5662 | 4.5488 | 6.3015 | 0.1375 | 0.00834851 |
| 6 | 2000 | `gpt_bert_10m_ratio_1to1_33m_step002000_epoch06` | 4.4326 | 26.98 | 4.3555 | 4.5097 | 2.7036 | 0.1284 | 0.00783918 |
| 6 | 2384 | `gpt_bert_10m_ratio_1to1_33m_step002384_epoch06` | 4.4715 | 26.41 | 4.3492 | 4.5938 | 9.3857 | 0.1293 | 0.00632109 |
| 7 | 2780 | `gpt_bert_10m_ratio_1to1_33m_step002780_epoch07` | 4.3267 | 27.60 | 4.1959 | 4.4575 | 7.9384 | 0.1231 | 0.00444073 |
| 8 | 3000 | `gpt_bert_10m_ratio_1to1_33m_step003000_epoch08` | 4.2682 | 28.91 | 4.1340 | 4.4023 | 2.9005 | 0.1138 | 0.00324412 |
| 8 | 3178 | `gpt_bert_10m_ratio_1to1_33m_step003178_epoch08` | 4.1586 | 29.36 | 4.0449 | 4.2723 | 2.4382 | 0.1144 | 0.00290028 |
| 9 | 3575 | `gpt_bert_10m_ratio_1to1_33m_step003575_epoch09` | 4.0410 | 30.24 | 3.8298 | 4.2522 | 1.0560 | 0.1074 | 0.00185775 |
| 10 | 3970 | `gpt_bert_10m_ratio_1to1_33m_step003970_epoch10` | 3.9711 | 31.50 | 3.8108 | 4.1315 | 2.8856 | 0.0991 | 0.00142005 |

## Zero-Shot Eval: `gpt_bert_10m_ratio_1to1_33m`

Checkpoint: `gpt_bert_10m_ratio_1to1_33m_step003970_epoch10-hf`  
Revision: `e10_ratio_1to1`; backend: `causal`

| Task | This run | 15/16 run | Diff vs 15/16 | GPT-2 Strict baseline | Diff vs Strict | GPT-2 Strict-Small baseline | Diff vs Strict-Small |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `zero_shot/blimp/blimp_filtered` | 64.47 | 59.93 | +4.54 | 74.53 | -10.06 | 65.08 | -0.61 |
| `zero_shot/blimp/supplement_filtered` | 56.00 | 54.42 | +1.58 | 65.00 | -9.00 | 57.25 | -1.25 |
| `zero_shot/comps/comps` | 51.34 | 50.69 | +0.65 | 55.85 | -4.51 | 51.81 | -0.47 |
| `zero_shot/entity_tracking/entity_tracking` | 37.38 | 33.01 | +4.37 | 23.58 | +13.80 | 21.07 | +16.31 |

## Training Curve: `gpt_bert_10m_ratio_1to3_33m`

| Epoch | Step | Checkpoint stem | Loss | Acc | BERT loss | GPT loss | Grad norm | Mask p | LR |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 396 | `gpt_bert_10m_ratio_1to3_33m_step000396_epoch01` | 4.5180 | 24.80 | 4.8126 | 4.4198 | 0.3533 | 0.0837 | 0.0139851 |
| 2 | 794 | `gpt_bert_10m_ratio_1to3_33m_step000794_epoch02` | 4.3327 | 27.01 | 4.2824 | 4.3495 | 0.2985 | 0.0808 | 0.0132857 |
| 3 | 1000 | `gpt_bert_10m_ratio_1to3_33m_step001000_epoch03` | 4.2122 | 28.52 | 4.4346 | 4.1381 | 0.3025 | 0.0754 | 0.0123827 |
| 3 | 1192 | `gpt_bert_10m_ratio_1to3_33m_step001192_epoch03` | 4.2094 | 28.35 | 4.4946 | 4.1143 | 0.2903 | 0.0737 | 0.0120191 |
| 4 | 1589 | `gpt_bert_10m_ratio_1to3_33m_step001589_epoch04` | 4.2957 | 27.23 | 4.5277 | 4.2183 | 0.3851 | 0.0712 | 0.0103151 |
| 5 | 1987 | `gpt_bert_10m_ratio_1to3_33m_step001987_epoch05` | 4.2595 | 27.46 | 4.3666 | 4.2237 | 0.5371 | 0.0696 | 0.00834851 |
| 6 | 2000 | `gpt_bert_10m_ratio_1to3_33m_step002000_epoch06` | 4.1542 | 27.94 | 4.1365 | 4.1601 | 0.7291 | 0.0656 | 0.00783918 |
| 6 | 2382 | `gpt_bert_10m_ratio_1to3_33m_step002382_epoch06` | 4.1380 | 27.97 | 4.1327 | 4.1398 | 1.1616 | 0.0635 | 0.00632109 |
| 7 | 2779 | `gpt_bert_10m_ratio_1to3_33m_step002779_epoch07` | 4.0141 | 29.35 | 4.1684 | 3.9626 | 1.0372 | 0.0600 | 0.00444073 |
| 8 | 3000 | `gpt_bert_10m_ratio_1to3_33m_step003000_epoch08` | 3.9357 | 30.18 | 4.0182 | 3.9082 | 0.6465 | 0.0574 | 0.00324412 |
| 8 | 3176 | `gpt_bert_10m_ratio_1to3_33m_step003176_epoch08` | 3.9467 | 29.98 | 3.9459 | 3.9469 | 0.4738 | 0.0580 | 0.00290028 |
| 9 | 3574 | `gpt_bert_10m_ratio_1to3_33m_step003574_epoch09` | 3.8266 | 31.20 | 3.8128 | 3.8311 | 0.3436 | 0.0522 | 0.00185775 |
| 10 | 3970 | `gpt_bert_10m_ratio_1to3_33m_step003970_epoch10` | 3.7925 | 32.37 | 3.6510 | 3.8397 | 0.3668 | 0.0502 | 0.00142005 |

Best logged point before checkpoint save:

| Epoch | Step | Loss | Acc | BERT loss | GPT loss | Mask p | LR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 3700 | 3.7071 | 32.69 | 3.3984 | 3.8099 | 0.0505 | 0.00155895 |

## Zero-Shot Eval: `gpt_bert_10m_ratio_1to3_33m`

Checkpoint: `gpt_bert_10m_ratio_1to3_33m_step003970_epoch10-hf`  
Revision: `e10_ratio_1to3`; backend: `causal`

| Task | This run | 1:1 run | Diff vs 1:1 | 15/16 run | Diff vs 15/16 | GPT-2 Strict baseline | Diff vs Strict | GPT-2 Strict-Small baseline | Diff vs Strict-Small |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `zero_shot/blimp/blimp_filtered` | 68.13 | 64.47 | +3.66 | 59.93 | +8.20 | 74.53 | -6.40 | 65.08 | +3.05 |
| `zero_shot/blimp/supplement_filtered` | 53.85 | 56.00 | -2.15 | 54.42 | -0.57 | 65.00 | -11.15 | 57.25 | -3.40 |
| `zero_shot/comps/comps` | 52.51 | 51.34 | +1.17 | 50.69 | +1.82 | 55.85 | -3.34 | 51.81 | +0.70 |
| `zero_shot/entity_tracking/entity_tracking` | 31.28 | 37.38 | -6.10 | 33.01 | -1.73 | 23.58 | +7.70 | 21.07 | +10.21 |

Legacy note: the first training run originally used checkpoint stems beginning with `gpt_bert_10m_official_8gpu`; after renaming, future runs should use the YAML `name` value `gpt_bert_10m_official_33m`.
