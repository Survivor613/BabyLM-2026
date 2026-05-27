from __future__ import annotations

from pathlib import Path

from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

from .configuration_gpt_bert import ModelConfig
from .modeling_gpt_bert import GPTBERTForCausalLM, GPTBERTForMaskedLM


HF_REPO_ID = "BabyLM-community/babylm-baseline-100m-gpt-bert-mixed"


def load_config(config_path: str | Path = "configs/gpt_bert_100m.json") -> ModelConfig:
    return ModelConfig(config_file=Path(config_path))


def build_causal_model(config_path: str | Path = "configs/gpt_bert_100m.json") -> GPTBERTForCausalLM:
    return GPTBERTForCausalLM(load_config(config_path))


def build_masked_model(config_path: str | Path = "configs/gpt_bert_100m.json") -> GPTBERTForMaskedLM:
    return GPTBERTForMaskedLM(load_config(config_path))


def load_hf_causal_model(repo_id: str = HF_REPO_ID):
    return AutoModelForCausalLM.from_pretrained(repo_id, trust_remote_code=True)


def load_hf_masked_model(repo_id: str = HF_REPO_ID):
    return AutoModelForMaskedLM.from_pretrained(repo_id, trust_remote_code=True)


def load_hf_tokenizer(repo_id: str = HF_REPO_ID):
    return AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
