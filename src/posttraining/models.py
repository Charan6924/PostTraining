"""Shared Qwen3 model/tokenizer loading and chat-template handling."""
from typing import Any, Mapping

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    set_seed,
)

def load_model_and_tokenizer(
    config: Mapping[str, Any], *, training: bool
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping of model settings")
    if type(training) is not bool:
        raise TypeError("training must be a boolean")

    required = (
        "model_name_or_path", "revision", "seed", "torch_dtype",
        "enable_thinking", "finetuning", "gradient_checkpointing",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing required model config values: {', '.join(missing)}")

    for key in ("model_name_or_path", "revision", "torch_dtype", "finetuning"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be an integer between 0 and 2**32 - 1")
    for key in ("enable_thinking", "gradient_checkpointing"):
        if type(config[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
    dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if config["torch_dtype"] not in dtypes:
        raise ValueError("torch_dtype must be bfloat16, float16, or float32")
    if config["finetuning"] != "full":
        raise ValueError("finetuning must be 'full'; adapter loading is not implemented")

    set_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name_or_path"], revision=config["revision"]
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        revision=config["revision"],
        torch_dtype=dtypes[config["torch_dtype"]],
    )
    if training:
        model.train()
        model.config.use_cache = False
        if config["gradient_checkpointing"]:
            model.gradient_checkpointing_enable()
    else:
        model.eval()
        model.config.use_cache = True
    return model, tokenizer
