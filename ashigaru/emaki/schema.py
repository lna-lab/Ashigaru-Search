"""Node model + SKILL.md / INDEX.md (de)serialisation for a KURA-Emaki scroll.

Each tree node materialises as a SKILL.md card (YAML frontmatter + a short navigable
body) plus an INDEX.md (its children or, at a leaf, its documents). The frontmatter
always carries `name` + `description` so the card is a VALID Anthropic Agent Skill
(github.com/anthropics/skills) — the scroll doubles as a portable skill library.

Dependency-free: a tiny single-line-value frontmatter writer/reader (no PyYAML), since we
own both ends. Values are folded to one line; `entities` is a comma list.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Node:
    """One node of the topic tree. Internal nodes hold `children`; leaves hold `doc_ids`."""
    node_id: str
    level: int = 0
    parent_id: str | None = None
    children: list[str] = field(default_factory=list)   # child node_ids (internal)
    doc_ids: list[str] = field(default_factory=list)     # member chunk ids (leaf)
    # distilled fields (filled by distill.py)
    name: str = ""
    summary: str = ""
    when_to_use: str = ""
    entities: list[str] = field(default_factory=list)
    num_documents: int = 0          # leaf-document count of the whole subtree
    confidence: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id, "level": self.level, "parent_id": self.parent_id,
            "children": list(self.children), "doc_ids": list(self.doc_ids),
            "name": self.name, "summary": self.summary, "when_to_use": self.when_to_use,
            "entities": list(self.entities), "num_documents": self.num_documents,
            "confidence": round(float(self.confidence), 4),
        }

    @staticmethod
    def from_dict(d: dict) -> "Node":
        return Node(
            node_id=str(d["node_id"]), level=int(d.get("level", 0)),
            parent_id=d.get("parent_id"),
            children=list(d.get("children", [])), doc_ids=list(d.get("doc_ids", [])),
            name=str(d.get("name", "")), summary=str(d.get("summary", "")),
            when_to_use=str(d.get("when_to_use", "")), entities=list(d.get("entities", [])),
            num_documents=int(d.get("num_documents", 0)),
            confidence=float(d.get("confidence", 0.0)),
        )


def _fold(s: str) -> str:
    """Collapse a value to a single safe line for frontmatter."""
    return " ".join(str(s).split()).strip()


def derive_description(node: Node) -> str:
    """A one-line `description` for Agent-Skills validity + navigation routing."""
    if node.when_to_use:
        return _fold(node.when_to_use)[:300]
    if node.summary:
        first = node.summary.replace("\n", " ").split(". ")[0]
        return _fold(first)[:300]
    kind = "leaf documents" if node.is_leaf else "sub-topics"
    return f"Knowledge branch '{node.name or node.node_id}' over {node.num_documents} documents ({kind})."


def format_skill_md(node: Node, description: str | None = None) -> str:
    """Render a node as an Anthropic-Agent-Skills-compatible SKILL.md card."""
    desc = _fold(description or derive_description(node))
    scope = "leaf" if node.is_leaf else "internal"
    fm = [
        "---",
        f"name: {_fold(node.name) or node.node_id}",
        f"description: {desc}",
        f"topic: {_fold(node.name)}",
        f"scope: {scope}",
        f"level: {node.level}",
        f"node_id: {node.node_id}",
        f"parent: {node.parent_id or ''}",
        f"num_documents: {node.num_documents}",
        f"confidence: {round(float(node.confidence), 3)}",
        f"entities: {', '.join(node.entities)}",
        "---",
    ]
    body = [
        "",
        "## Overview",
        node.summary.strip() or "(no summary)",
        "",
        "## When to navigate here",
        node.when_to_use.strip() or desc,
    ]
    if node.entities:
        body += ["", "## Key entities", *[f"- {e}" for e in node.entities]]
    body += ["", "## Contents"]
    if node.is_leaf:
        body.append(f"{len(node.doc_ids)} document(s) — read one with "
                    f'get_document(doc_id). The ids are listed in INDEX.md.')
    else:
        body.append(f"{len(node.children)} sub-branch(es) — drill into one with "
                    f'tree_open(node_id). The branches are listed in INDEX.md.')
    return "\n".join(fm + body) + "\n"


def parse_skill_md(text: str) -> dict:
    """Parse a SKILL.md card back into a flat dict (frontmatter keys + `_body`)."""
    out: dict = {}
    t = text.lstrip()
    if t.startswith("---"):
        end = t.find("\n---", 3)
        if end != -1:
            block = t[3:end].strip("\n")
            out["_body"] = t[end + 4:].lstrip("\n")
            for line in block.splitlines():
                if ":" not in line:
                    continue
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip()
    if "entities" in out:
        out["entities"] = [e.strip() for e in out["entities"].split(",") if e.strip()]
    for k in ("level", "num_documents"):
        if k in out:
            try:
                out[k] = int(out[k])
            except ValueError:
                pass
    if "confidence" in out:
        try:
            out["confidence"] = float(out["confidence"])
        except ValueError:
            pass
    return out


def format_index_md(node: Node, node_map: dict[str, Node],
                    doc_meta: dict[str, tuple[str, str]] | None = None) -> str:
    """Render a node's INDEX.md — its child branches (internal) or documents (leaf).

    doc_meta maps doc_id -> (source, snippet) for leaf listings.
    """
    head = [f"# Index — {node.name or node.node_id}  ({node.node_id})", ""]
    if node.is_leaf:
        head.append("Documents — read one in full with get_document(doc_id):")
        for did in node.doc_ids:
            src, snip = (doc_meta or {}).get(did, ("", ""))
            label = f" — {src}" if src else ""
            tail = f": {snip}" if snip else ""
            head.append(f"- {did}{label}{tail}")
    else:
        head.append("Sub-branches — drill into one with tree_open(node_id):")
        for cid in node.children:
            c = node_map.get(cid)
            if not c:
                continue
            desc = derive_description(c)
            head.append(f"- {cid} — {c.name or cid}: {desc}")
    return "\n".join(head) + "\n"
