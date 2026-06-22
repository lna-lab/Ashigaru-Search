"""Tool registry / toolbox. Tools are model-agnostic: each has a name, a one-line
description, a usage example, and an async `run(args) -> str`."""
from __future__ import annotations
import httpx
from dataclasses import dataclass
from typing import Awaitable, Callable

from .config import Config


_WEB_START_HINT = ("Start with web_search (and/or doc_search for local). Then fetch_url / "
                   "read_chunk to READ the best sources before concluding.")


def _emaki_start_hint(has_graph: bool) -> str:
    base = ("Start with tree_overview to see the corpus map, then tree_open(node_id) to drill "
            "into the most relevant branch, and get_document(doc_id) to read a leaf in full. "
            "Backtrack to a sibling branch if a path is unproductive.")
    if has_graph:
        base += (" Use graph_neighbors / graph_related_docs to pivot across related entities.")
    return base + " Use web_search / doc_search only as a fallback."


@dataclass
class Tool:
    name: str
    description: str
    usage: str                                   # shown to the model as a call example
    run: Callable[[dict], Awaitable[str]]


class ToolBox:
    def __init__(self, client: httpx.AsyncClient | None = None, sources=None):
        self._tools: dict[str, Tool] = {}
        self._client = client
        self.start_hint = _WEB_START_HINT       # how the worker should open its tool loop
        self.sources = sources                  # shared SourceRegistry (足軽ターボ); None = legacy URLs

    def add(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def render_docs(self) -> str:
        lines = []
        for t in self._tools.values():
            lines.append(f"- {t.name}: {t.description}\n    example: {t.usage}")
        return "\n".join(lines)

    async def run(self, name: str, args: dict) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"ERROR: unknown tool '{name}'. Available: {', '.join(self.names)}"
        try:
            return await tool.run(args or {})
        except Exception as e:  # never let a tool crash the worker loop
            return f"ERROR running {name}: {type(e).__name__}: {e}"

    async def aclose(self):
        if self._client is not None:
            await self._client.aclose()


def build_toolbox(cfg: Config) -> ToolBox:
    """Assemble the toolbox: web tools always; local-RAG tools if a BM25 index is configured;
    KURA-Emaki navigation tools if a built scroll is configured (and steer the scout to it)."""
    from .tools.web import make_web_tools
    from .tools.rag import make_rag_tools
    from .sources import SourceRegistry

    client = httpx.AsyncClient(timeout=cfg.request_timeout, follow_redirects=True,
                               headers={"User-Agent": "Ashigaru-Search/0.1 (+https://github.com/lna-lab/Ashigaru-Search)"})
    # one shared registry per run -> globally-unique [Sn] ids across the whole fleet
    sources = SourceRegistry() if cfg.ref_id_sources else None
    # default-deny egress gate (+ audit) so the fleet can't be steered at this box's own
    # loopback/LAN/metadata services. None = disabled (legacy behaviour).
    gate = None
    if cfg.egress_gate:
        from .egress import EgressGate, FileAuditLog, NullAudit
        audit = FileAuditLog(cfg.egress_audit) if cfg.egress_audit else NullAudit()
        allow = [h.strip() for h in (cfg.egress_allow or "").split(",") if h.strip()]
        gate = EgressGate(audit, allow, fetch_open=cfg.egress_fetch_open)
    box = ToolBox(client, sources=sources)
    for t in make_web_tools(cfg, client, sources, gate=gate):
        box.add(t)
    if cfg.x_search_enabled:
        from .tools.x_search import make_x_search_tools
        for t in make_x_search_tools(cfg, client, sources):
            box.add(t)
    if cfg.has_rag:
        for t in make_rag_tools(cfg):
            box.add(t)
    if cfg.has_emaki:
        from .tools.emaki import make_emaki_tools
        emaki_tools, has_graph = make_emaki_tools(cfg)
        for t in emaki_tools:
            box.add(t)
        box.start_hint = _emaki_start_hint(has_graph)
    return box
