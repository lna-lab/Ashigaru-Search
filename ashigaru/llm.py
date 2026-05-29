"""Minimal async OpenAI-compatible chat client (works with vLLM, llama.cpp server,
SGLang, OpenAI, etc.) — no `openai` dependency, just httpx."""
from __future__ import annotations
import httpx
from typing import Any


class LLMClient:
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self._client = httpx.AsyncClient(timeout=timeout)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        top_k: int | None = 80,
        repetition_penalty: float | None = 1.05,
        stop: list[str] | None = None,
    ) -> str:
        """Return the assistant message content. extra_body carries vLLM-only sampling
        params (top_k / repetition_penalty) which OpenAI ignores harmlessly."""
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop:
            payload["stop"] = stop
        extra = {}
        if top_k is not None:
            extra["top_k"] = top_k
        if repetition_penalty is not None:
            extra["repetition_penalty"] = repetition_penalty
        payload.update(extra)  # vLLM reads these top-level; OpenAI ignores unknowns? -> keep in extra_body too
        headers = {"Authorization": f"Bearer {self.api_key}"}
        r = await self._client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"] or ""

    async def aclose(self):
        await self._client.aclose()
