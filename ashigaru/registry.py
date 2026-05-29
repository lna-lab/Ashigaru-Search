"""Tool registry / toolbox. Tools are model-agnostic: each has a name, a one-line
description, a usage example, and an async `run(args) -> str`."""
from __future__ import annotations
import httpx
from dataclasses import dataclass
from typing import Awaitable, Callable

from .config import Config


@dataclass
class Tool:
    name: str
    description: str
    usage: str                                   # shown to the model as a call example
    run: Callable[[dict], Awaitable[str]]


class ToolBox:
    def __init__(self, client: httpx.AsyncClient | None = None):
        self._tools: dict[str, Tool] = {}
        self._client = client

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
    """Assemble the toolbox: web tools always; local-RAG tools if an index is configured."""
    from .tools.web import make_web_tools
    from .tools.rag import make_rag_tools

    client = httpx.AsyncClient(timeout=cfg.request_timeout, follow_redirects=True,
                               headers={"User-Agent": "Ashigaru-Search/0.1 (+https://github.com/lna-lab/Ashigaru-Search)"})
    box = ToolBox(client)
    for t in make_web_tools(cfg, client):
        box.add(t)
    if cfg.has_rag:
        for t in make_rag_tools(cfg):
            box.add(t)
    return box
