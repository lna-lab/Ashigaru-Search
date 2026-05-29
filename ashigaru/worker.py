"""The Ashigaru — one worker scout (ashigaru = a foot-soldier). Given a sub-question and a
toolbox, it runs a bounded tool-use loop (search → read → reason) and returns grounded
findings + sources. Works with any instruct model that follows the simple <tool>/<final>
protocol."""
from __future__ import annotations
import re
from dataclasses import dataclass, field

_URL_RE = re.compile(r'https?://[^\s)\]<>"\']+')

from .config import Config
from .llm import LLMClient
from .registry import ToolBox
from .toolproto import Action, parse_action, tool_result_message

WORKER_SYSTEM = """You are an Ashigaru, a focused research scout in a fleet. \
Investigate ONE sub-question thoroughly with the tools, then report concise findings WITH sources.

Available tools:
{tools}

Protocol — follow EXACTLY:
- To call a tool, output ONLY this (nothing else):
  <tool>{{"name":"<tool_name>","arguments":{{...}}}}</tool>
- When you have enough evidence, output your report:
  <final>
  3-8 sentence findings, grounded ONLY in what the tools returned.
  Sources:
  - <url or chunk id> — what it supports
  </final>

Rules:
- Start with web_search (and/or doc_search for local). Then fetch_url / read_chunk to actually READ the best sources before concluding.
- Fetch at least one source before your <final>. Never invent URLs or facts.
- Be efficient: at most {max_steps} tool calls. If evidence is thin, say so in <final>."""


@dataclass
class WorkerResult:
    index: int
    task: str
    findings: str
    sources: list[str] = field(default_factory=list)
    steps: int = 0
    ok: bool = True


async def run_ashigaru(llm: LLMClient, toolbox: ToolBox, task: str, cfg: Config,
                       index: int = 0, on_event=None) -> WorkerResult:
    sys_prompt = WORKER_SYSTEM.format(tools=toolbox.render_docs(), max_steps=cfg.worker_max_steps)
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"Sub-question to investigate:\n{task}\n\nBegin."},
    ]
    sources: list[str] = []

    def _note(url: str):
        if url and url not in sources:
            sources.append(url)

    last_text = ""
    for step in range(1, cfg.worker_max_steps + 1):
        text = await llm.chat(messages, temperature=cfg.temperature, max_tokens=2048)
        last_text = text
        act: Action = parse_action(text)

        if act.kind == "final":
            for u in _URL_RE.findall(act.text):     # also credit sources cited in the report
                _note(u.rstrip(".,);"))
            if on_event:
                on_event("worker_done", {"index": index, "steps": step})
            return WorkerResult(index, task, act.text.strip(), sources, step, ok=True)

        # tool call
        if on_event:
            on_event("worker_tool", {"index": index, "step": step, "tool": act.name, "args": act.args})
        result = await toolbox.run(act.name, act.args)
        if act.name == "fetch_url" and act.args.get("url"):
            _note(str(act.args["url"]))
        if act.name == "read_chunk" and act.args.get("id"):
            _note(str(act.args["id"]))
        messages.append({"role": "assistant", "content": text})
        messages.append(tool_result_message(act.name or "tool", result))

    # ran out of steps — force a final synthesis from what we have
    messages.append({"role": "user", "content": "Stop searching. Give your <final>…</final> report now, "
                                                 "grounded in what you found so far."})
    text = await llm.chat(messages, temperature=cfg.temperature, max_tokens=2048)
    act = parse_action(text)
    findings = act.text.strip() if act.kind == "final" else (act.text or last_text).strip()
    for u in _URL_RE.findall(findings):
        _note(u.rstrip(".,);"))
    if on_event:
        on_event("worker_done", {"index": index, "steps": cfg.worker_max_steps, "forced": True})
    return WorkerResult(index, task, findings, sources, cfg.worker_max_steps, ok=True)
