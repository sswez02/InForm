from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import LlamaTokenizerFast, LlamaForCausalLM
from peft import PeftModel

from .formatting import build_prompt


class DomainLLM:
    """
    Wrapper around the fine-tuned domain LLM (base + LoRA)
    """

    def __init__(
        self,
        base_model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        lora_dir: str = "model_lora",
        device: str | None = None,
        max_new_tokens: int = 512,
    ) -> None:

        self.base_model_name = base_model_name
        self.lora_dir = lora_dir
        self.max_new_tokens = max_new_tokens
        self.tokenizer = LlamaTokenizerFast.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = LlamaForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if device is None else {"": device},
        )
        self.model = PeftModel.from_pretrained(self.model, lora_dir)
        self.model.eval()

    @torch.no_grad()
    def generate_answer(
        self,
        instruction: str,
        query: str,
        context_passages: List[Dict[str, Any]],
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> str:
        prompt = build_prompt(instruction, query, context_passages)

        inputs = self.tokenizer(
            [prompt],
            return_tensors="pt",
            padding=True,
        )

        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        # output_ids contains prompt + continuation
        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Keep only the generated continuation
        generated = output_ids[0, input_ids.shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)

        return text.strip()
