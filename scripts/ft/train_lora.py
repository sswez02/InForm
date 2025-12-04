from __future__ import annotations

import json
import inspect
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    LlamaTokenizerFast,
    LlamaForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

from src.ft.formatting import build_prompt


CONFIG_PATH = Path("ft_config.json")


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl_dataset(path: Path):
    return load_dataset("json", data_files=str(path), split="train")


def preprocess(example):
    instruction = example["instruction"]
    input_obj = example["input"]
    query = input_obj["query"]
    context = input_obj["context"]
    output = example["output"]

    prompt = build_prompt(instruction, query, context)
    full_text = prompt + output

    return {"text": full_text}


def main():
    cfg = load_config()

    train_path = Path("data/ft/splits/train.jsonl")
    val_path = Path("data/ft/splits/val.jsonl")

    model_name = cfg["model_name"]
    max_length = cfg["max_length"]

    # Load tokenizer + model
    tokenizer = LlamaTokenizerFast.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = LlamaForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )

    # LoRA config from file
    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Load datasets
    train_ds_raw = load_jsonl_dataset(train_path)
    val_ds_raw = load_jsonl_dataset(val_path)

    # Preprocess
    train_ds = train_ds_raw.map(preprocess)
    val_ds = val_ds_raw.map(preprocess)

    def tokenize_fn(batch):
        enc = tokenizer(
            batch["text"],
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    # Tokenize
    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)

    # Data collator
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training args from config
    training_kwargs = {
        "output_dir": "checkpoints/",
        "learning_rate": cfg["learning_rate"],
        "per_device_train_batch_size": cfg["batch_size"],
        "per_device_eval_batch_size": cfg["batch_size"],
        "gradient_accumulation_steps": cfg["grad_accum"],
        "fp16": torch.cuda.is_available(),
        "warmup_steps": 50,
        "num_train_epochs": cfg["epochs"],
        "logging_steps": 10,
        # Nice-to-have extras – will be dropped if not supported
        "evaluation_strategy": "steps",
        "eval_steps": 100,
        "save_steps": 200,
        "save_total_limit": 5,
        "report_to": "none",
    }

    # Look at the actual signature of TrainingArguments in install
    sig = inspect.signature(TrainingArguments.__init__)
    allowed_params = set(sig.parameters.keys())

    filtered_kwargs = {k: v for k, v in training_kwargs.items() if k in allowed_params}
    args = TrainingArguments(**filtered_kwargs)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    trainer.train()

    model.save_pretrained("model_lora")
    tokenizer.save_pretrained("model_lora")


if __name__ == "__main__":
    main()
