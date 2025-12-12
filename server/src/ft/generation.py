from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import LlamaTokenizerFast, AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM

from .formatting import build_prompt


class DomainLLM:
    """
    Wrapper around the fine-tuned domain LLM (base + LoRA)
    Loads a PEFT LoRA adapter and exposes a simple `generate_answer` API
    """

    def __init__(
        self,
        base_model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        lora_dir: str = "model_lora",
        device: str | None = None,
        max_new_tokens: int = 32,
        max_input_tokens: int = 512,
    ) -> None:
        self.base_model_name = base_model_name
        self.lora_dir = lora_dir
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens

        # Decide device (cuda if available, else cpu, unless explicitly overridden)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[DomainLLM] torch.cuda.is_available() = {torch.cuda.is_available()}")
        print(f"[DomainLLM] Using device = {self.device}")

        # Use tokenizer from LoRA dir if it exists (matches training),
        # otherwise fall back to base model tokenizer
        tok_source = lora_dir if Path(lora_dir).exists() else base_model_name
        self.tokenizer = LlamaTokenizerFast.from_pretrained(tok_source)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base + LoRA adapter together
        # AutoPeftModelForCausalLM will read adapter_config.json from `lora_dir`,
        # find the base model, and construct the full model correctly
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            lora_dir,
            dtype=torch.float32 if self.device == "cpu" else torch.float16,
            device_map=None if self.device == "cpu" else "auto",
        )

        if self.device == "cpu":
            self.model.to("cpu")

        self.model.eval()

    @torch.no_grad()
    def generate_answer(
        self,
        instruction: str,
        query: str,
        context_passages: List[Dict[str, Any]],
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        """
        Compose a prompt from instruction/query/context and generate a completion
        """
        prompt = build_prompt(instruction, query, context_passages)

        # Tokenize and move to device
        inputs = self.tokenizer(
            [prompt],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_input_tokens,
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Use sampling only if temperature > 0, otherwise greedy
        do_sample = temperature > 0.0

        print("[DomainLLM] Calling model.generate()...")
        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            temperature=temperature if do_sample else 0.0,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        print("[DomainLLM] model.generate() returned")

        # Keep only the generated continuation (strip the prompt tokens)
        generated = output_ids[0, input_ids.shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)

        return text.strip()
