"""The Ashigaru — one worker scout (ashigaru = a foot-soldier). Given a sub-question and a
toolbox, it runs a bounded tool-use loop (search → read → reason) and returns grounded
findings + sources. Works with any instruct model that follows the simple <tool>/<final>
protocol.

Mid-search the scout can check in with the Commander ("found this lead — continue?"), who
orders 'continue' or 'regroup' (file an interim note and re-launch on a revised focus)."""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field

from .config import Config
from .llm import LLMClient
from .registry import ToolBox
from .toolproto import Action, parse_action, strip_think, tool_result_message

_URL_RE = re.compile(r'https?://[^\s)\]<>"\']+')

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
- {start_hint}
- Read at least one source/document in full before your <final>. Never invent URLs, ids, or facts.
- Be efficient: at most {max_steps} tool calls. If evidence is thin, say so in <final>."""

# fallback when a toolbox doesn't carry a start_hint (e.g. a test double)
_FALLBACK_START_HINT = ("Start with web_search (and/or doc_search for local), then fetch_url / "
                        "read_chunk to READ the best sources before concluding.")

SUPERVISOR_SYSTEM = """You are the Commander supervising a scout after each lead it reports. \
The scout is a small model and may dig aimlessly — your job is to prevent wasted effort and \
keep it on the overall goal. Given the goal, the scout's current sub-question, and the lead it \
just found, give the next order. Output ONLY JSON: \
{"action": "continue" | "regroup" | "return", "reason": "...", \
"new_focus": "if regroup, the revised sub-question / strategy"}.
- continue: ONLY if this lead is genuinely valuable AND more digging on it clearly helps the goal.
- regroup: a better angle exists — have the scout file what it has and re-launch on new_focus.
- return: enough is found, or further digging here is low-value — stop and write the report now.
Bias against wasteful digging: prefer return/regroup unless continuing is clearly worth it."""


@dataclass
class WorkerResult:
    index: int
    task: str
    findings: str
    sources: list[str] = field(default_factory=list)
    steps: int = 0
    ok: bool = True


async def _supervise(orch: LLMClient, overall: str, subq: str, evidence: str,
                     cfg: Config) -> tuple[str, str, str]:
    """Ask the Commander for a mid-search order. Returns (action, reason, new_focus);
    defaults to ('continue', …) whenever the verdict can't be parsed."""
    msgs = [
        {"role": "system", "content": SUPERVISOR_SYSTEM},
        {"role": "user", "content": f"Overall goal: {overall}\nScout sub-question: {subq}\n"
                                     f"Lead just found:\n{evidence[:1000]}"},
    ]
    txt = strip_think(await orch.chat(msgs, temperature=0.0, max_tokens=1024))
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        return "continue", "", ""
    try:
        d = json.loads(m.group(0))
        act = str(d.get("action", "continue")).lower()
        return (act if act in ("continue", "regroup", "return") else "continue"), \
               str(d.get("reason", "")), str(d.get("new_focus", ""))
    except Exception:
        return "continue", "", ""


async def run_ashigaru(llm: LLMClient, toolbox: ToolBox, task: str, cfg: Config,
                       index: int = 0, on_event=None, orch: LLMClient | None = None,
                       overall: str = "") -> WorkerResult:
    subq = task
    sys_prompt = WORKER_SYSTEM.format(tools=toolbox.render_docs(), max_steps=cfg.worker_max_steps,
                                      start_hint=getattr(toolbox, "start_hint", _FALLBACK_START_HINT))
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"Sub-question to investigate:\n{task}\n\nBegin."},
    ]
    sources: list[str] = []
    notes: list[str] = []          # interim reports filed when the Commander says 'regroup'
    checkins = 0

    def _note(url: str):
        if url and url not in sources:
            sources.append(url)

    def _finalize(text: str) -> str:
        text = text.strip()
        if notes:
            return ("Interim notes filed mid-search:\n"
                    + "\n".join(f"- {n}" for n in notes) + "\n\n" + text)
        return text

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
            return WorkerResult(index, subq, _finalize(act.text), sources, step, ok=True)

        # tool call
        if on_event:
            on_event("worker_tool", {"index": index, "step": step, "tool": act.name, "args": act.args})
        result = await toolbox.run(act.name, act.args)
        if act.name == "fetch_url" and act.args.get("url"):
            _note(str(act.args["url"]))
        if act.name == "read_chunk" and act.args.get("id"):
            _note(str(act.args["id"]))
        if act.name == "get_document":            # KURA-Emaki leaf read -> credit the doc id
            did = act.args.get("doc_id") or act.args.get("id")
            if did:
                _note(str(did))
        messages.append({"role": "assistant", "content": text})
        messages.append(tool_result_message(act.name or "tool", result))

        # --- mid-search check-in: scout reports the lead, Commander orders continue/regroup ---
        if (cfg.supervise and orch is not None and step >= cfg.supervise_after
                and checkins < cfg.max_checkins and step < cfg.worker_max_steps):
            checkins += 1
            action, reason, new_focus = await _supervise(orch, overall or subq, task, result, cfg)
            if on_event:
                on_event("worker_checkin", {"index": index, "step": step,
                                            "action": action, "focus": (new_focus or "")[:70]})
            if action == "return":
                break                              # 帰投 → write the report from what's gathered
            if action == "regroup":
                snippet = " ".join(result.split())[:240]
                notes.append(f"[{task[:50]}] {snippet}")
                task = new_focus or task           # 即帰投 → 一旦報告(note) → 新フォーカスで再出撃
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"New focus from the Commander: {task}\n\n"
                                                 f"(You already filed this lead: {snippet[:200]})\nBegin."},
                ]

    # ran out of steps — force a final synthesis from what we have
    messages.append({"role": "user", "content": "Stop searching. Give your <final>…</final> report now, "
                                                 "grounded in what you found so far."})
    text = await llm.chat(messages, temperature=cfg.temperature, max_tokens=2048)
    act = parse_action(text)
    findings = act.text.strip() if act.kind == "final" else (act.text or last_text).strip()
    for u in _URL_RE.findall(findings):
        _note(u.rstrip(".,);"))
    recalled = step < cfg.worker_max_steps      # came home early on a 'return' order
    if on_event:
        on_event("worker_done", {"index": index, "steps": step, "forced": not recalled,
                                 "recalled": recalled})
    return WorkerResult(index, subq, _finalize(findings), sources, step, ok=True)
