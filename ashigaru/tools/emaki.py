"""KURA-Emaki navigation tools — exposed to the scouts through the same <tool>/<final>
protocol as every other tool, so navigating a knowledge scroll needs zero worker changes.

  tree_overview()        — root bird's-eye card + branch index
  tree_open(node_id)     — drill into a branch (its card + children, or a leaf's documents)
  get_document(doc_id)   — read a leaf document in full

If the scroll was built with a graph (--graph / --graph-llm), two more verbs let the scout
pivot sideways along typed relationships:

  graph_neighbors(entity)      — entities related to one, with edge types
  graph_related_docs(entity)   — documents that mention an entity
"""
from __future__ import annotations

from ..config import Config
from ..registry import Tool
from ..emaki.library import load_emaki


def make_emaki_tools(cfg: Config) -> tuple[list[Tool], bool]:
    """Return (tools, has_graph). has_graph tells the caller whether to advertise the graph
    verbs in the worker start hint, without coupling on tool-name string literals."""
    lib = load_emaki(cfg.emaki_tree)

    async def tree_overview(args: dict) -> str:
        return lib.overview()

    async def tree_open(args: dict) -> str:
        nid = str(args.get("node_id") or args.get("id") or args.get("node") or "").strip()
        if not nid:
            return "ERROR: tree_open needs a 'node_id' (see tree_overview / INDEX.md)."
        return lib.open(nid)

    async def get_document(args: dict) -> str:
        did = str(args.get("doc_id") or args.get("id") or "").strip()
        if not did:
            return "ERROR: get_document needs a 'doc_id' (see a leaf's INDEX.md)."
        return lib.get_document(did, cfg.fetch_char_limit)

    tools = [
        Tool("tree_overview",
             "Show the root bird's-eye view of the local knowledge scroll and its top branches.",
             '<tool>{"name":"tree_overview","arguments":{}}</tool>',
             tree_overview),
        Tool("tree_open",
             "Drill into a branch by node_id: returns its card + child branches (or leaf documents).",
             '<tool>{"name":"tree_open","arguments":{"node_id":"root.2"}}</tool>',
             tree_open),
        Tool("get_document",
             "Read a leaf document in full by doc_id (from a leaf's INDEX.md).",
             '<tool>{"name":"get_document","arguments":{"doc_id":"guide.md#3"}}</tool>',
             get_document),
    ]

    if lib.has_graph:
        from ..emaki.graph import load_graph
        # pass the embeddings endpoint so graph_neighbors resolves the query to its node by
        # MEANING (semantic) when the graph's nodes carry embeddings — not just by keyword.
        graph = load_graph(cfg.emaki_tree, embed_url=cfg.embed_url, embed_model=cfg.embed_model,
                           embed_query_instruct=cfg.embed_query_instruct)

        async def graph_neighbors(args: dict) -> str:
            ent = str(args.get("entity") or args.get("query") or args.get("id") or "").strip()
            if not ent:
                return "ERROR: graph_neighbors needs an 'entity'."
            hops = int(args.get("hops") or 1)
            return graph.neighbors(ent, hops=max(1, min(hops, 3)))

        async def graph_related_docs(args: dict) -> str:
            ent = str(args.get("entity") or args.get("query") or args.get("id") or "").strip()
            if not ent:
                return "ERROR: graph_related_docs needs an 'entity'."
            return graph.related_docs(ent)

        tools += [
            Tool("graph_neighbors",
                 "Find entities related to a given entity in the knowledge graph, with edge types.",
                 '<tool>{"name":"graph_neighbors","arguments":{"entity":"vLLM","hops":1}}</tool>',
                 graph_neighbors),
            Tool("graph_related_docs",
                 "List documents that mention an entity (then read one with get_document).",
                 '<tool>{"name":"graph_related_docs","arguments":{"entity":"vLLM"}}</tool>',
                 graph_related_docs),
        ]
    return tools, lib.has_graph
