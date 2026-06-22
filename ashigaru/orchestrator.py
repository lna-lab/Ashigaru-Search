"""The Commander (orchestrator): decompose a question into sub-questions, dispatch one
Ashigaru scout per sub-question concurrently against the fleet, then synthesize a cited
answer.

The orchestrator model is pluggable (config) — it can be the same fleet, a bigger local
NVFP4 model, or any OpenAI-compatible endpoint."""
from __future__ import annotations
import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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


def _current_datetime_context() -> str:
    """A concise 'now' line appended to Commander prompts. Local NVFP4 models have a stale
    training cutoff — telling them today's date lets them phrase 'latest/current' sub-questions
    in the right year and judge whether sources are stale. Defaults to Asia/Tokyo; ``TZ`` env
    overrides."""
    tz_name = os.environ.get("TZ", "Asia/Tokyo")
    try:
        tz = ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError):
        tz, tz_name = ZoneInfo("Asia/Tokyo"), "Asia/Tokyo"
    now = datetime.now(tz)
    utc_now = datetime.now(ZoneInfo("UTC"))
    return (f"Today's date/time: {now.strftime('%Y-%m-%d %H:%M:%S')} ({tz_name}); "
            f"UTC: {utc_now.strftime('%Y-%m-%d %H:%M:%S')}.")


PLAN_SYSTEM = """You are the Commander of a research fleet. Break the user's question \
{constraint} SHARP, non-overlapping sub-questions that, once answered with web/local \
search, together fully answer it. Cover distinct facets (definitions, current state, numbers, \
comparisons, caveats). Output ONLY a JSON array of strings, nothing else."""


def _tier_count(tag: str, fleet_size: int) -> int:
    return max(1, min(12, math.ceil(fleet_size * _SML_FRAC[tag])))


def parse_scout_count(question: str, default: int, fleet_size: int = 10) -> tuple[int, str, str]:
    """A leading token + space picks the scout count (overrides default), clamped 1..12:
      - "3 compare X and Y"  -> (3, "compare X and Y", "num")       explicit number
      - "M latest on X"      -> density tag: S=10% / M=50% / L=100% of `fleet_size`,
                                rounded UP (ceil), min 1 (fleet 10 -> S=1, M=5, L=10).
    Returns (count, cleaned_question, kind) where kind ∈ {num, s, m, l, default}."""
    m = _NUM_RE.match(question)
    if m:
        return max(1, min(int(m.group(1)), 12)), m.group(2).strip(), "num"
    m = _SML_RE.match(question)
    if m:
        tag = m.group(1).lower()
        return _tier_count(tag, fleet_size), m.group(2).strip(), tag
    return default, question.strip(), "default"


def _escalate(kind: str, count: int, cfg: Config) -> tuple[str, int] | None:
    """Next density tier on a thin run. S→M→L; numeric/default → ×2 (cap 12). None = capped."""
    if kind in ("num", "default"):
        nc = min(12, count * 2)
        return ("num", nc) if nc > count else None
    ladder = sorted({_tier_count("s", cfg.fleet_size),
                     _tier_count("m", cfg.fleet_size),
                     _tier_count("l", cfg.fleet_size)})
    higher = [x for x in ladder if x > count]
    if not higher:
        return None
    nxt = higher[0]
    return ("l" if nxt == _tier_count("l", cfg.fleet_size) else "m"), nxt

_FAIL_RE = re.compile(
    r"\b(no results|couldn'?t find|could ?n'?t find|could not find|unable to find|"
    r"no (?:relevant )?(?:information|sources|data)|not enough (?:information|evidence)|"
    r"i (?:don'?t|do not) have)\b", re.IGNORECASE)

JUDGE_SYSTEM = """You are the Commander reviewing ONE scout's report against its sub-question. \
Decide if it actually answers the sub-question with concrete, source-grounded information \
(not vague, not empty, not "couldn't find"). Output ONLY JSON: \
{"ok": true|false, "hint": "if not ok, what to search or do differently"}."""

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
        {"role": "system", "content": PLAN_SYSTEM.format(constraint=constraint)
            + "\n\n" + _current_datetime_context()},
        {"role": "user", "content": question},
    ]
    # reasoning models (e.g. LFM2.5) think at length before the array — give room so
    # the JSON actually gets emitted after </think> (else we fall back to 1 scout)
    text = strip_think(await orch.chat(msgs, temperature=cfg.temperature, max_tokens=cfg.orch_max_tokens))
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
    msgs = [{"role": "system", "content": SYNTH_SYSTEM + "\n\n" + _current_datetime_context()},
            {"role": "user", "content": user}]
    return strip_think(await orch.chat(msgs, temperature=cfg.temperature, max_tokens=cfg.orch_max_tokens))


def _heuristic_ok(w: WorkerResult, cfg: Config) -> tuple[bool, str]:
    """Cheap no-LLM gate for 'no substance' reports."""
    f = (w.findings or "").strip()
    if len(f) < cfg.quality_min_chars:
        return False, "report too short / thin"
    if not w.sources:
        return False, "no sources cited"
    if _FAIL_RE.search(f):
        return False, "scout said it couldn't find enough"
    return True, ""


async def _judge(orch: LLMClient, subq: str, w: WorkerResult, cfg: Config) -> tuple[bool, str]:
    """Commander judges whether the report actually answers the sub-question."""
    msgs = [
        {"role": "system", "content": JUDGE_SYSTEM + "\n\n" + _current_datetime_context()},
        {"role": "user", "content": f"Sub-question: {subq}\n\nScout report:\n{w.findings}\n\n"
                                     f"Cited sources: {w.sources or '(none)'}"},
    ]
    txt = strip_think(await orch.chat(msgs, temperature=0.0, max_tokens=cfg.orch_max_tokens // 2))
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        return True, ""                       # judge unsure -> don't block
    try:
        d = json.loads(m.group(0))
        return bool(d.get("ok", True)), str(d.get("hint", ""))
    except Exception:
        return True, ""


async def _assess(orch: LLMClient, subs: list[str], workers: list[WorkerResult],
                  cfg: Config) -> tuple[int, int]:
    """How many scout reports are substantive? Returns (substantive_count, needed)."""
    async def _ok(subq: str, w: WorkerResult) -> bool:
        ok, _ = _heuristic_ok(w, cfg)
        if ok and cfg.quality_judge:
            ok, _ = await _judge(orch, subq, w, cfg)
        return ok
    flags = await asyncio.gather(*[_ok(subs[w.index], w) for w in workers])
    needed = max(1, math.ceil(0.5 * len(workers)))
    return sum(1 for f in flags if f), needed


async def research(question: str, cfg: Config | None = None, on_event=None) -> ResearchResult:
    cfg = cfg or Config()
    # a leading "<n> " / "S|M|L " in the question sets the scout count + density tier
    count, question, kind = parse_scout_count(question, cfg.max_subquestions, cfg.fleet_size)
    explicit = kind != "default"
    orch = LLMClient(cfg.orch_base_url, cfg.orch_model, cfg.orch_api_key, cfg.request_timeout)
    worker_llm = LLMClient(cfg.worker_base_url, cfg.worker_model, cfg.worker_api_key, cfg.request_timeout)
    toolbox = build_toolbox(cfg)
    sem = asyncio.Semaphore(cfg.max_concurrency)

    async def _one(i: int, q: str) -> WorkerResult:
        async with sem:
            if on_event:
                on_event("worker_start", {"index": i, "task": q})
            return await run_ashigaru(worker_llm, toolbox, q, cfg, index=i, on_event=on_event,
                                      orch=orch, overall=question)

    try:
        escalations = 0
        subs: list[str] = []
        workers: list[WorkerResult] = []
        while True:
            cfg.max_subquestions = count
            subs = await _plan(orch, question, cfg, exact=(explicit or escalations > 0))
            if on_event:
                on_event("plan", {"subquestions": subs,
                                  "requested": count if (explicit or escalations) else None,
                                  "escalation": escalations})
            workers = list(await asyncio.gather(*[_one(i, q) for i, q in enumerate(subs)]))

            if not cfg.escalate or escalations >= cfg.max_escalations:
                break
            good, needed = await _assess(orch, subs, workers, cfg)
            if good >= needed:
                break
            nxt = _escalate(kind, count, cfg)
            if nxt is None:
                break  # already at the top tier / cap
            new_kind, new_count = nxt
            if on_event:
                on_event("escalate", {"from": count, "to": new_count, "good": good,
                                      "needed": needed, "kind": new_kind})
            kind, count = new_kind, new_count
            escalations += 1

        if on_event:
            on_event("synthesize", {"workers": len(workers)})
        answer = await _synthesize(orch, question, workers, cfg)

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
