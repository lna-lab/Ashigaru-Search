"""Distil each tree node into a navigable card (topic label + summary + when-to-use +
entities). This is the BUILD-time "summarise many clusters" raid — the same Commander
fan-out the live fleet uses, here pointed inward at the corpus.

Bottom-up: leaves are summarised from their member chunks; parents are rolled up from their
children's cards. Every card passes a cheap quality gate (non-empty label, substantive
summary, no refusal); a thin card is re-distilled once with more context, and as a last
resort falls back to a heuristic card so a build NEVER hard-fails on a flaky small model.

Pass llm=None for a fully heuristic (zero-LLM, zero-network) build — used by the tests.
"""
from __future__ import annotations

import asyncio
import json
import re
from collections import Counter

from ..config import Config
from ..llm import LLMClient
from ..toolproto import strip_think
from .cluster import tokenize, _STOP
from .schema import Node

_MIN_SUMMARY = 40
_REFUSE = re.compile(r"\b(cannot|can'?t|couldn'?t|could not|no excerpts|not enough|unable|"
                     r"i (?:don'?t|do not) have)\b", re.IGNORECASE)
_BAD_LABEL = {"", "unknown", "untitled", "n/a", "none", "cluster"}

LEAF_SYSTEM = """You label and summarise ONE cluster of related document excerpts for a \
navigable knowledge map (a "scroll" scouts walk instead of keyword search).
Output ONLY JSON, no markdown fences:
{"label": "a <=6-word topic title",
 "summary": "2-4 sentences: what this cluster is about, grounded ONLY in the excerpts",
 "when_to_use": "one line: what kind of question should drill into this branch",
 "entities": ["key entity or term", "...up to 8"]}
Be concrete and faithful. Never invent facts that are not in the excerpts."""

PARENT_SYSTEM = """You write the bird's-eye card for a PARENT branch of a knowledge map, \
summarising its child sub-branches into one umbrella view.
Output ONLY JSON, no markdown fences:
{"label": "a <=6-word umbrella topic",
 "summary": "2-4 sentences describing what the whole branch covers, inferred from its children",
 "when_to_use": "one line: when a scout should open this branch",
 "entities": ["recurring key term", "...up to 8"]}"""


def _clean_entities(ents) -> list[str]:
    out: list[str] = []
    for e in ents or []:
        e = " ".join(str(e).split()).strip()
        if not e or len(e) > 60 or e.lower() in _STOP:
            continue
        if e not in out:
            out.append(e)
    return out[:8]


def _node_ok(node: Node) -> bool:
    if (node.name or "").strip().lower() in _BAD_LABEL:
        return False
    s = (node.summary or "").strip()
    return len(s) >= _MIN_SUMMARY and not _REFUSE.search(s)


def _apply(node: Node, parsed: dict) -> None:
    node.name = " ".join(str(parsed.get("label", "")).split())[:80] or node.name
    node.summary = str(parsed.get("summary", "")).strip() or node.summary
    node.when_to_use = " ".join(str(parsed.get("when_to_use", "")).split())[:300] or node.when_to_use
    ents = _clean_entities(parsed.get("entities"))
    if ents:
        node.entities = ents


def _heuristic(node: Node, texts: list[str]) -> None:
    """Last-resort, LLM-free card from raw text frequency."""
    toks: list[str] = []
    for t in texts:
        toks.extend(tokenize(t))
    common = [w for w, _ in Counter(toks).most_common(8)]
    if (node.name or "").strip().lower() in _BAD_LABEL:
        node.name = common[0].title() if common else node.node_id
    if len((node.summary or "").strip()) < _MIN_SUMMARY:
        joined = " ".join(" ".join(t.split()) for t in texts if t)
        node.summary = (joined[:300] or f"Cluster of {len(texts)} excerpts.").strip()
    if not node.when_to_use:
        node.when_to_use = f"Questions about {', '.join(common[:4]) or node.name}."
    if not node.entities:
        node.entities = common[:8]
    node.confidence = min(node.confidence or 0.3, 0.3)


def _extract_json_obj(txt: str) -> dict | None:
    """Robustly pull a JSON object out of an LLM reply: strip code fences, try the whole
    string, else scan balanced braces from the first '{' (greedy '{.*}' over-captures
    trailing prose/fences and silently drops the object)."""
    s = (txt or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s).strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    start = s.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(s[start:i + 1])
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
    return None


async def _chat_json(llm: LLMClient, system: str, user: str, *, temperature: float) -> dict | None:
    txt = strip_think(await llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature, max_tokens=1024))
    return _extract_json_obj(txt)


def _leaf_user(node: Node, chunk_text: dict[str, str], n: int) -> tuple[str, list[str]]:
    texts = [chunk_text.get(d, "") for d in node.doc_ids[:n]]
    body = "\n".join(f"[{i+1}] {' '.join(t.split())[:600]}" for i, t in enumerate(texts) if t)
    return f"Excerpts in this cluster:\n\n{body}", texts


def _parent_user(node: Node, nodes: dict[str, Node]) -> str:
    rows = []
    for cid in node.children:
        c = nodes.get(cid)
        if c:
            rows.append(f"- {c.name or cid}: {(' '.join(c.summary.split()))[:200]}")
    return "Child sub-branches:\n" + "\n".join(rows)


async def distill_node(llm: LLMClient | None, node: Node, nodes: dict[str, Node],
                       chunk_text: dict[str, str], cfg: Config) -> None:
    is_leaf = node.is_leaf
    texts_for_fallback = [chunk_text.get(d, "") for d in node.doc_ids[:10]] if is_leaf \
        else [nodes[c].summary or nodes[c].name for c in node.children if c in nodes]

    if llm is None:
        _heuristic(node, texts_for_fallback)
        return

    if is_leaf:
        user, _ = _leaf_user(node, chunk_text, n=6)
        system = LEAF_SYSTEM
    else:
        user, system = _parent_user(node, nodes), PARENT_SYSTEM

    parsed = await _chat_json(llm, system, user, temperature=cfg.temperature)
    if parsed:
        _apply(node, parsed)
        node.confidence = 1.0

    if not _node_ok(node):                              # one re-distill with more context
        if is_leaf:
            user, _ = _leaf_user(node, chunk_text, n=10)
        parsed = await _chat_json(llm, system, user, temperature=0.0)
        if parsed:
            _apply(node, parsed)
            node.confidence = 0.6

    if not _node_ok(node):                              # still thin -> heuristic backstop
        _heuristic(node, texts_for_fallback)


async def distill_tree(llm: LLMClient | None, nodes: dict[str, Node], root_id: str,
                       chunk_text: dict[str, str], cfg: Config, on_event=None,
                       concurrency: int | None = None) -> None:
    """Distil every node, deepest level first (children ready before parents)."""
    sem = asyncio.Semaphore(concurrency or cfg.max_concurrency)
    by_level: dict[int, list[str]] = {}
    for n in nodes.values():
        by_level.setdefault(n.level, []).append(n.node_id)

    done = 0
    total = len(nodes)
    for level in sorted(by_level, reverse=True):
        async def _one(nid: str):
            nonlocal done
            async with sem:
                await distill_node(llm, nodes[nid], nodes, chunk_text, cfg)
            done += 1
            if on_event:
                on_event("distill", {"node": nid, "level": level,
                                     "name": nodes[nid].name, "done": done, "total": total})
        await asyncio.gather(*[_one(nid) for nid in by_level[level]])
