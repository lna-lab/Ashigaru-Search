"""Edge-type registry — the runtime authority on which relationships may exist in the
knowledge graph. Loads edge_types.yaml (the ontology adopted from AIOS / LNA-ES v4.0).

Dependency-free: uses PyYAML if present, else a tiny parser for our fixed, flat YAML shape
(a list of `- name:` items with `key: value` lines). Keeping the ontology as editable YAML
mirrors AIOS so the vocabulary can be extended without touching code.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache

EDGE_TYPES_PATH = os.path.join(os.path.dirname(__file__), "edge_types.yaml")

# default entity/node types for LLM extraction (domain-general; override per corpus)
NODE_TYPES = ["PERSON", "ORG", "PLACE", "CONCEPT", "EVENT", "ARTIFACT", "TIME", "TERM"]


@dataclass(frozen=True)
class EdgeType:
    name: str
    inverse: str
    weight_default: float
    description: str
    ai_hint: str

    @property
    def is_symmetric(self) -> bool:
        return self.inverse == self.name


def _unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] in "\"'" and s[-1] == s[0]:
        return s[1:-1]
    return s


def _parse_edge_yaml(text: str) -> list[dict]:
    """Minimal parser for our edge_types.yaml (no PyYAML needed)."""
    try:
        import yaml  # use the real thing if available
        data = yaml.safe_load(text)
        return data.get("edge_types", []) if isinstance(data, dict) else []
    except Exception:
        pass
    items: list[dict] = []
    cur: dict | None = None
    in_list = False
    for raw in text.splitlines():
        if raw.strip().startswith("#") or not raw.strip():
            continue
        if raw.strip() == "edge_types:":
            in_list = True
            continue
        if not in_list:
            continue
        stripped = raw.strip()
        if stripped.startswith("- "):
            if cur:
                items.append(cur)
            cur = {}
            stripped = stripped[2:]
        if cur is not None and ":" in stripped:
            k, v = stripped.split(":", 1)
            cur[k.strip()] = _unquote(v)
    if cur:
        items.append(cur)
    return items


class EdgeRegistry:
    def __init__(self, path: str | None = None):
        self._types: dict[str, EdgeType] = {}
        self._load(path or EDGE_TYPES_PATH)

    def _load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for entry in _parse_edge_yaml(f.read()):
                name = entry.get("name")
                if not name:
                    continue
                try:
                    w = float(entry.get("weight_default", 0.5))
                except (TypeError, ValueError):
                    w = 0.5
                self._types[name] = EdgeType(
                    name=name, inverse=entry.get("inverse", name), weight_default=w,
                    description=entry.get("description", ""), ai_hint=entry.get("ai_hint", ""))

    def get(self, name: str) -> EdgeType | None:
        return self._types.get(name)

    def validate(self, name: str) -> bool:
        return name in self._types

    def canonical(self, name: str, default: str = "RELATED_TO") -> str:
        """Map a raw LLM label to a known edge type (upper-cased); fall back to a generic."""
        if not name:
            return default
        # normalise any separators (spaces, hyphens, dots) to underscores: "part-of" -> PART_OF
        up = re.sub(r"[^A-Z0-9]+", "_", str(name).strip().upper()).strip("_")
        return up if up in self._types else default

    def default_weight(self, name: str) -> float:
        et = self._types.get(name)
        return et.weight_default if et else 0.5

    def inverse_of(self, name: str) -> str | None:
        et = self._types.get(name)
        return et.inverse if et else None

    def ai_hint(self, name: str) -> str:
        et = self._types.get(name)
        return et.ai_hint if et else ""

    def names(self) -> set[str]:
        return set(self._types)

    def all_types(self) -> list[EdgeType]:
        return list(self._types.values())

    def format_vocabulary_for_prompt(self) -> str:
        lines = ["Relationship types (use these EXACT names):"]
        for et in self._types.values():
            sym = " (symmetric)" if et.is_symmetric else f" (inverse: {et.inverse})"
            lines.append(f"- {et.name}{sym}: {et.description}")
        return "\n".join(lines)


@lru_cache(maxsize=1)
def get_edge_registry() -> EdgeRegistry:
    return EdgeRegistry()
