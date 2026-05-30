"""KURA-Emaki (蔵絵巻) — turn a bounded LOCAL corpus into a navigable topic-tree of skill
cards the Ashigaru scouts WALK at query time, instead of (or alongside) top-k BM25/RAG.

「検索から探索へ」 — Don't retrieve, navigate. The *kura* (蔵, a storehouse of documents)
is compiled OFFLINE into an *emaki* (絵巻, a picture-scroll you unroll): an embed/cluster +
LLM-summarise topic tree materialised as Anthropic-Agent-Skills-compatible SKILL.md cards.
At serve time a scout reads the root bird's-eye card, drills coarse->fine into the most
relevant branch, reads leaf documents in full, and BACKTRACKS from dead branches — with no
vector DB / BM25 needed (BM25 stays attached as an optional hybrid fallback).

Clean-room: derived only from the published idea (arXiv:2604.14572 "Don't Retrieve,
Navigate"; lineage RAPTOR 2401.18059, GraphRAG 2404.16130, Voyager 2305.16291) and from the
Ashigaru architecture we own. NO upstream code copied (upstream repo is all-rights-reserved).
"""
from .schema import Node, format_skill_md, parse_skill_md, format_index_md
from .library import EmakiLibrary, load_emaki

__all__ = [
    "Node",
    "format_skill_md",
    "parse_skill_md",
    "format_index_md",
    "EmakiLibrary",
    "load_emaki",
]
