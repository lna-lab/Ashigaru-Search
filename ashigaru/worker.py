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
# a trailing model-written "Sources:" / "出典:" section (replaced by the harness-built map in turbo mode)
_SRC_SECTION_RE = re.compile(r'\n\s*(?:Sources?|出典|参考(?:文献)?|参照)\s*[:：].*$',
                             re.IGNORECASE | re.DOTALL)


def _attach_sources(text: str, registry):
    """足軽ターボ finalize: resolve the [Sn] ids the scout cited to VERBATIM URLs, replace its
    (URL-less) Sources section with a deterministic harness-built one, and return
    (rewritten_text, source_urls). The model never had to write a URL, so they can't be wrong."""
    refs = registry.refs_in(text)
    urls = registry.verbatim_sources(refs)
    if refs:
        body = _SRC_SECTION_RE.sub("", text).rstrip()
        text = f"{body}\n\nSources:\n{registry.source_map(refs)}"
    return text, urls

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
  {source_example}
  </final>

Rules:
- {start_hint}
- {source_rule}
- Be efficient: at most {max_steps} tool calls. If evidence is thin, say so in <final>."""

# how sources are cited — turbo (stable [Sn] ids) vs legacy (raw URLs)
_REF_SOURCE_EXAMPLE = "- [S1] — what it supports     (cite the source id; the harness fills in the URL)"
_REF_SOURCE_RULE = ("Read at least one source in full (fetch_url by id) before your <final>. "
                    "Cite each source by its [Sn] id only, e.g. [S1] — NEVER write a URL and never "
                    "invent an id; the harness attaches the exact URLs for you.")
_LEGACY_SOURCE_EXAMPLE = "- <url or chunk id> — what it supports"
_LEGACY_SOURCE_RULE = ("Read at least one source/document in full before your <final>. "
                       "Never invent URLs, ids, or facts.")

# fallback when a toolbox doesn't carry a start_hint (e.g. a test double)
_FALLBACK_START_HINT = ("Start with web_search (and/or doc_search for local), then fetch_url / "
                        "read_chunk to READ the best sources before concluding.")

# 蔵-recall mode: steer the (premium) Commander to read the fleet's OWN memory instead of the
# web — flat recall first, then NAVIGATE the knowledge graph if one is present. Other scouts
# cover the web, so this pass is pure prior-knowledge.
_RECALL_START_HINT = (
    "Consult ONLY our own local 蔵 (memory) — do NOT web_search; other scouts cover the web. "
    "Start with doc_search on the goal; then, if graph tools are present, use tree_overview and "
    "graph_neighbors / graph_related_docs to pivot to related entities and get_document / "
    "read_chunk to READ the most relevant notes in full. Report what we ALREADY know, or say "
    "plainly that memory holds nothing relevant.")

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
                       overall: str = "", recall: bool = False) -> WorkerResult:
    subq = task
    registry = getattr(toolbox, "sources", None)      # 足軽ターボ SourceRegistry, or None (legacy)
    # recall mode reads our 蔵 (memory/graph) instead of the web — see _RECALL_START_HINT.
    hint = _RECALL_START_HINT if recall else getattr(toolbox, "start_hint", _FALLBACK_START_HINT)
    sys_prompt = WORKER_SYSTEM.format(
        tools=toolbox.render_docs(), max_steps=cfg.worker_max_steps,
        start_hint=hint,
        source_example=_REF_SOURCE_EXAMPLE if registry is not None else _LEGACY_SOURCE_EXAMPLE,
        source_rule=_REF_SOURCE_RULE if registry is not None else _LEGACY_SOURCE_RULE)
    intro = ("From our OWN memory (蔵), surface what we already know about this goal:"
             if recall else "Sub-question to investigate:")
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"{intro}\n{task}\n\nBegin."},
    ]
    sources: list[str] = []
    notes: list[str] = []          # interim reports filed when the Commander says 'regroup'
    checkins = 0
    has_read = False               # has the scout opened a source in full yet?
    nudged_read = False            # we nudge "read before final" at most once (no loop)
    _READ_TOOLS = ("fetch_url", "read_chunk", "get_document")
    _can_read = any(toolbox.get(t) for t in _READ_TOOLS) if hasattr(toolbox, "get") else False

    def _note(url: str):
        if url and url not in sources:
            sources.append(url)

    def _finalize(text: str) -> str:
        text = text.strip()
        if notes:
            return ("Interim notes filed mid-search:\n"
                    + "\n".join(f"- {n}" for n in notes) + "\n\n" + text)
        return text

    # 足軽ターボ: a tiny scout often hallucinates a <final> instead of deciding to search, and
    # (measured) even an 8B rarely DECIDES to traverse the knowledge graph — it falls back to flat
    # search. So we HAND the structure over: seed the loop with automatic call(s) so the model
    # always starts from real evidence. In recall mode with a built scroll we additionally seed
    # tree_overview — the 蔵's bird's-eye map — so even a 1.2B rides the graph structure instead
    # of having to choose to. (Skipped for test doubles without a real toolbox.)
    if cfg.auto_first_search and hasattr(toolbox, "get"):
        seeds: list[tuple[str, dict]] = []
        if recall and toolbox.get("graph_neighbors"):
            # LEAD with the query's TYPED subgraph: the navigator resolves the question to its
            # most specific entity and returns its typed relations (incl. grounding anchors), so
            # even a 1.2B narrates the real edges instead of deciding to traverse — cognitive load
            # → mechanism. (We deliberately do NOT seed the whole-corpus tree_overview here: for a
            # focused "what connects to X" recall it is orientation noise that drowns the precise
            # subgraph; the scout can still call tree_overview itself to widen out.)
            seeds.append(("graph_neighbors", {"entity": task}))
        seed_order = ("doc_search", "web_search") if recall else ("web_search", "doc_search")
        primary = next((t for t in seed_order if toolbox.get(t)), None)
        if primary:
            seeds.append((primary, {"query": task}))
        for seed_tool, seed_args in seeds:
            seed_res = await toolbox.run(seed_tool, seed_args)
            if on_event:
                on_event("worker_tool", {"index": index, "step": 0, "tool": seed_tool,
                                         "args": seed_args, "auto": True})
            messages.append({"role": "assistant",
                             "content": f"<tool>{json.dumps({'name': seed_tool, 'arguments': seed_args})}</tool>"})
            messages.append(tool_result_message(seed_tool, seed_res))
        # Recall mode hands the answer over via the seeds (the typed subgraph + docs ARE the
        # answer to "what does the 蔵 know about X"). So collapse the scout's job to pure
        # narration — a tiny model that flails on the multi-step <tool> protocol can still read
        # back structure it was given. Cognitive load → mechanism, pushed to its limit.
        if recall and seeds:
            messages.append({"role": "user", "content":
                "You now have the 蔵's typed relations (graph_neighbors) and related documents "
                "above. Do NOT call any more tools. Using ONLY the information already handed to "
                "you, write your <final> report now: state what the memory connects this to, by "
                "relation type. If the relations are thin, say so honestly."})

    last_text = ""
    for step in range(1, cfg.worker_max_steps + 1):
        text = await llm.chat(messages, temperature=cfg.temperature, max_tokens=2048)
        last_text = text
        act: Action = parse_action(text)

        if act.kind == "final":
            # grounding gate: don't accept a <final> from a scout that never opened a source
            # in full (the tiny scout tends to satisfice after the seed search). Nudge ONCE.
            if _can_read and not has_read and not nudged_read:
                nudged_read = True
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content":
                    "Before your <final>: you have not READ any source in full yet. Open at "
                    "least one with fetch_url (by id) / read_chunk / get_document so your "
                    "findings are grounded in the actual content, then write your report."})
                continue
            findings = act.text
            if registry is not None:                # turbo: resolve [Sn] -> verbatim URLs
                findings, ref_urls = _attach_sources(findings, registry)
                for u in ref_urls:
                    _note(u)
            else:
                for u in _URL_RE.findall(findings):  # legacy: credit URLs the model wrote
                    _note(u.rstrip(".,);"))
            if on_event:
                on_event("worker_done", {"index": index, "steps": step})
            return WorkerResult(index, subq, _finalize(findings), sources, step, ok=True)

        # tool call
        if on_event:
            on_event("worker_tool", {"index": index, "step": step, "tool": act.name, "args": act.args})
        result = await toolbox.run(act.name, act.args)
        if act.name in _READ_TOOLS:
            has_read = True
        if act.name == "fetch_url":
            if registry is not None and (act.args.get("id") or act.args.get("ref") or act.args.get("source")):
                _note(registry.resolve(act.args.get("id") or act.args.get("ref") or act.args.get("source")) or "")
            elif act.args.get("url"):
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
    if registry is not None:
        findings, ref_urls = _attach_sources(findings, registry)
        for u in ref_urls:
            _note(u)
    else:
        for u in _URL_RE.findall(findings):
            _note(u.rstrip(".,);"))
    recalled = step < cfg.worker_max_steps      # came home early on a 'return' order
    if on_event:
        on_event("worker_done", {"index": index, "steps": step, "forced": not recalled,
                                 "recalled": recalled})
    return WorkerResult(index, subq, _finalize(findings), sources, step, ok=True)
