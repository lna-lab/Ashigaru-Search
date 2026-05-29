"""The Commander (orchestrator): decompose a question into sub-questions, dispatch one
Ashigaru scout per sub-question concurrently against the fleet, then synthesize a cited
answer.

The orchestrator model is pluggable (config) — it can be the same fleet, a bigger local
NVFP4 model, or any OpenAI-compatible endpoint."""
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

import math

_ARR_RE = re.compile(r"\[.*\]", re.DOTALL)
# leading "<n> " (half-width digits) OR "<S|M|L> " density tag, + space (full-width tolerated)
_NUM_RE = re.compile(r"^\s*(\d{1,2})[ 　]+(\S.*)$", re.DOTALL)
_SML_RE = re.compile(r"^\s*([SMLsml])[ 　]+(\S.*)$", re.DOTALL)
_SML_FRAC = {"s": 0.10, "m": 0.50, "l": 1.00}    # 1割 / 5割 / 10割 of the fleet

PLAN_SYSTEM = """You are the Commander of a research fleet. Break the user's question \
{constraint} SHARP, non-overlapping sub-questions that, once answered with web/local \
search, together fully answer it. Cover distinct facets (definitions, current state, numbers, \
comparisons, caveats). Output ONLY a JSON array of strings, nothing else."""


def parse_scout_count(question: str, default: int, fleet_size: int = 10) -> tuple[int, str, bool]:
    """A leading token + space picks the scout count (overrides default), clamped 1..12:
      - "3 compare X and Y"  -> (3, "compare X and Y", True)        explicit number
      - "M latest on X"      -> density tag: S=10% / M=50% / L=100% of `fleet_size`,
                                rounded UP (ceil), min 1 (fleet 10 -> S=1, M=5, L=10).
    Returns (count, cleaned_question, explicit)."""
    m = _NUM_RE.match(question)
    if m:
        return max(1, min(int(m.group(1)), 12)), m.group(2).strip(), True
    m = _SML_RE.match(question)
    if m:
        n = max(1, min(12, math.ceil(fleet_size * _SML_FRAC[m.group(1).lower()])))
        return n, m.group(2).strip(), True
    return default, question.strip(), False

SYNTH_SYSTEM = """You are the Commander. Your Ashigaru scouts each investigated one sub-question \
and returned findings with sources. Write the final answer to the user's ORIGINAL question:
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


async def _plan(orch: LLMClient, question: str, cfg: Config, exact: bool = False) -> list[str]:
    n = cfg.max_subquestions
    constraint = f"into EXACTLY {n}" if exact else f"into {n} or fewer"
    msgs = [
        {"role": "system", "content": PLAN_SYSTEM.format(constraint=constraint)},
        {"role": "user", "content": question},
    ]
    # reasoning models (e.g. LFM2.5) think at length before the array — give room so
    # the JSON actually gets emitted after </think> (else we fall back to 1 scout)
    text = strip_think(await orch.chat(msgs, temperature=cfg.temperature, max_tokens=3072))
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
    return strip_think(await orch.chat(msgs, temperature=cfg.temperature, max_tokens=3072))


async def research(question: str, cfg: Config | None = None, on_event=None) -> ResearchResult:
    cfg = cfg or Config()
    # a leading "<n> " in the question explicitly sets how many scouts to dispatch
    n, question, explicit = parse_scout_count(question, cfg.max_subquestions, cfg.fleet_size)
    cfg.max_subquestions = n
    orch = LLMClient(cfg.orch_base_url, cfg.orch_model, cfg.orch_api_key, cfg.request_timeout)
    worker_llm = LLMClient(cfg.worker_base_url, cfg.worker_model, cfg.worker_api_key, cfg.request_timeout)
    toolbox = build_toolbox(cfg)
    try:
        subs = await _plan(orch, question, cfg, exact=explicit)
        if on_event:
            on_event("plan", {"subquestions": subs, "requested": n if explicit else None})

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
