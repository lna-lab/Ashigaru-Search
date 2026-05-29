"""The 大将 (orchestrator): decompose a question into sub-questions, dispatch one 足軽
per sub-question concurrently against the vLLM fleet, then synthesize a cited answer.

The orchestrator model is pluggable (config) — it can be the same LFM2.5 fleet, a bigger
local NVFP4 model, or any OpenAI-compatible endpoint."""
from __future__ import annotations
import asyncio
import json
import re
from dataclasses import dataclass, field

from .config import Config
from .llm import LLMClient
from .registry import build_toolbox
from .toolproto import strip_think
from .worker import WorkerResult, run_ashigaru

_ARR_RE = re.compile(r"\[.*\]", re.DOTALL)

PLAN_SYSTEM = """You are the 大将 (commander) of a research fleet. Break the user's question \
into {n} or fewer SHARP, non-overlapping sub-questions that, once answered with web/local \
search, together fully answer it. Cover distinct facets (definitions, current state, numbers, \
comparisons, caveats). Output ONLY a JSON array of strings, nothing else."""

SYNTH_SYSTEM = """You are the 大将. Your 足軽 scouts each investigated one sub-question and \
returned findings with sources. Write the final answer to the user's ORIGINAL question:
- Synthesize across scouts; resolve overlaps; note disagreements/uncertainty.
- Ground every claim in the scouts' findings; do NOT add facts they didn't surface.
- Answer in the user's language. Be direct and well-structured.
- End with a "Sources" list (dedup the urls / chunk ids the scouts cited)."""


@dataclass
class ResearchResult:
    question: str
    answer: str
    subquestions: list[str] = field(default_factory=list)
    workers: list[WorkerResult] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


async def _plan(orch: LLMClient, question: str, cfg: Config) -> list[str]:
    msgs = [
        {"role": "system", "content": PLAN_SYSTEM.format(n=cfg.max_subquestions)},
        {"role": "user", "content": question},
    ]
    text = strip_think(await orch.chat(msgs, temperature=cfg.temperature, max_tokens=512))
    m = _ARR_RE.search(text)
    if m:
        try:
            arr = json.loads(m.group(0))
            subs = [str(s).strip() for s in arr if str(s).strip()]
            if subs:
                return subs[: cfg.max_subquestions]
        except Exception:
            pass
    return [question]  # fallback: single scout on the whole question


async def _synthesize(orch: LLMClient, question: str, workers: list[WorkerResult], cfg: Config) -> str:
    blocks = []
    for w in workers:
        src = "\n".join(f"    - {s}" for s in w.sources) or "    (none cited)"
        blocks.append(f"### Scout {w.index + 1}: {w.task}\n{w.findings}\nSources:\n{src}")
    user = f"ORIGINAL QUESTION:\n{question}\n\nSCOUT FINDINGS:\n" + "\n\n".join(blocks)
    msgs = [{"role": "system", "content": SYNTH_SYSTEM}, {"role": "user", "content": user}]
    return strip_think(await orch.chat(msgs, temperature=cfg.temperature, max_tokens=2048))


async def research(question: str, cfg: Config | None = None, on_event=None) -> ResearchResult:
    cfg = cfg or Config()
    orch = LLMClient(cfg.orch_base_url, cfg.orch_model, cfg.orch_api_key, cfg.request_timeout)
    worker_llm = LLMClient(cfg.worker_base_url, cfg.worker_model, cfg.worker_api_key, cfg.request_timeout)
    toolbox = build_toolbox(cfg)
    try:
        subs = await _plan(orch, question, cfg)
        if on_event:
            on_event("plan", {"subquestions": subs})

        sem = asyncio.Semaphore(cfg.max_concurrency)

        async def _one(i: int, q: str) -> WorkerResult:
            async with sem:
                if on_event:
                    on_event("worker_start", {"index": i, "task": q})
                return await run_ashigaru(worker_llm, toolbox, q, cfg, index=i, on_event=on_event)

        workers = await asyncio.gather(*[_one(i, q) for i, q in enumerate(subs)])

        if on_event:
            on_event("synthesize", {"workers": len(workers)})
        answer = await _synthesize(orch, question, list(workers), cfg)

        sources: list[str] = []
        for w in workers:
            for s in w.sources:
                if s not in sources:
                    sources.append(s)
        return ResearchResult(question, answer, subs, list(workers), sources)
    finally:
        await toolbox.aclose()
        await orch.aclose()
        await worker_llm.aclose()
