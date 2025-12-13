from __future__ import annotations

import os
from typing import List, Dict, Any

from dotenv import load_dotenv

from src.ft.formatting import build_prompt

load_dotenv()


class OpenAIDomainLLM:
    """
    Domain LLM that calls an OpenAI model.
    Drop-in replacement for the old GeminiDomainLLM.
    Uses the new Responses API (recommended for gpt-4.1-mini, gpt-4.1, etc.).
    """

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "OpenAI dependency is missing. Add 'openai' to requirements.txt."
            ) from e

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it in your environment "
                "or add it to your server/.env file."
            )

        self.client = OpenAI(api_key=api_key)
        self.model_name = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def generate_answer(
        self,
        instruction: str,
        query: str,
        context_passages: List[Dict[str, Any]],
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        prompt = build_prompt(instruction, query, context_passages)

        system_instruction = (
            "You are an evidence-based fitness and nutrition assistant. "
            "You must only answer using the provided study excerpts, and "
            "always include inline citations like [1], [2] that match the "
            "studies in the context."
        )

        temp = self.temperature if temperature is None else temperature
        tp = self.top_p if top_p is None else top_p

        response = self.client.responses.create(
            model=self.model_name,
            max_output_tokens=self.max_new_tokens,
            temperature=temp,
            top_p=tp,
            input=[
                {
                    "role": "system",
                    "content": system_instruction,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        text_chunks: list[str] = []

        for item in response.output[0].content:
            if hasattr(item, "text"):
                text_chunks.append(item.text)
            else:
                text_chunks.append(str(item))

        text = "".join(text_chunks)
        return (text or "").strip()
