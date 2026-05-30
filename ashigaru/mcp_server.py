"""MCP front door — exposes the Ashigaru fleet as a tool any MCP client (Claude Code,
Claude Desktop, …) can call to offload research to your local fleet.

Run:  ashigaru-mcp        (stdio transport)
Then register in your MCP client config, e.g. Claude Code:
  { "mcpServers": { "ashigaru": { "command": "ashigaru-mcp" } } }
"""
from __future__ import annotations

from .config import Config
from .orchestrator import research

try:
    from mcp.server.fastmcp import FastMCP
except Exception as e:  # pragma: no cover
    raise SystemExit("MCP SDK not installed. `pip install \"ashigaru-search[mcp]\"` or `pip install mcp`.") from e

mcp = FastMCP("ashigaru")


@mcp.tool()
async def deep_research(query: str, max_subquestions: int = 6) -> str:
    """Dispatch a fleet of local LLM scouts (ashigaru) to research a question across the web
    (SearXNG) and any configured local document corpus, then return a synthesized,
    source-cited answer.

    Args:
        query: the research question.
        max_subquestions: how many parallel scouts to fan out (1-12).
    """
    cfg = Config()
    cfg.max_subquestions = max(1, min(int(max_subquestions), 12))
    res = await research(query, cfg)
    src = "\n".join(f"- {s}" for s in res.sources) or "(no external sources cited)"
    return f"{res.answer}\n\n## Sources\n{src}"


@mcp.tool()
async def emaki_navigate(query: str, max_subquestions: int = 4) -> str:
    """Navigate a pre-built KURA-Emaki knowledge scroll (蔵絵巻) for a bounded LOCAL corpus.

    Scouts drill the topic tree (tree_overview -> tree_open -> get_document) and pivot across
    the ontology knowledge graph (graph_neighbors / graph_related_docs) instead of doing
    top-k retrieval — "don't retrieve, navigate". Build a scroll with
    `ashigaru-emaki <corpus_dir> <out_dir>` and set ASHIGARU_EMAKI=<out_dir>.

    Args:
        query: the research question.
        max_subquestions: how many parallel scouts to fan out (1-12).
    """
    cfg = Config()
    if not cfg.has_emaki:
        return ("ERROR: no KURA-Emaki scroll configured. Build one with "
                "`ashigaru-emaki <corpus_dir> <out_dir>` and set ASHIGARU_EMAKI=<out_dir>.")
    cfg.max_subquestions = max(1, min(int(max_subquestions), 12))
    res = await research(query, cfg)
    src = "\n".join(f"- {s}" for s in res.sources) or "(no sources cited)"
    return f"{res.answer}\n\n## Sources\n{src}"


def main():
    mcp.run()  # stdio


if __name__ == "__main__":
    main()
