"""Local document RAG tools: doc_search + read_chunk over a BM25 index.

BM25 keeps the default zero-GPU and dependency-light (no embedding model download).
Build an index from a folder of .txt/.md/.pdf with:  `ashigaru-index <folder> <out.pkl>`
(see ashigaru.rag_index). Swap in embeddings later behind the same tool interface.
"""
from __future__ import annotations
import pickle

from ..config import Config
from ..registry import Tool

_TOKEN = __import__("re").compile(r"\w+", __import__("re").UNICODE)


def _tok(s: str) -> list[str]:
    return [w.lower() for w in _TOKEN.findall(s)]


def make_rag_tools(cfg: Config) -> list[Tool]:
    with open(cfg.rag_index, "rb") as f:
        idx = pickle.load(f)
    chunks: list[dict] = idx["chunks"]          # [{id, source, text}, ...]
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([_tok(c["text"]) for c in chunks])

    async def doc_search(args: dict) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            return "ERROR: doc_search needs a 'query'."
        k = int(args.get("k") or 5)
        scores = bm25.get_scores(_tok(query))
        order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:k]
        out = [f"Local document hits for: {query}"]
        for i in order:
            if scores[i] <= 0:
                continue
            c = chunks[i]
            snip = " ".join(c["text"].split())[:240]
            out.append(f"[{c['id']}] ({c['source']})  score={scores[i]:.1f}\n    {snip}")
        return "\n".join(out) if len(out) > 1 else f"No local matches for: {query}"

    async def read_chunk(args: dict) -> str:
        cid = str(args.get("id") or "").strip()
        for c in chunks:
            if str(c["id"]) == cid:
                return f"[{c['id']}] ({c['source']})\n{c['text']}"
        return f"ERROR: no chunk with id '{cid}'."

    return [
        Tool("doc_search",
             "Search the LOCAL document corpus (BM25). Returns chunk ids, sources and snippets.",
             '<tool>{"name":"doc_search","arguments":{"query":"...", "k":5}}</tool>',
             doc_search),
        Tool("read_chunk",
             "Read the full text of a local chunk by id (from doc_search).",
             '<tool>{"name":"read_chunk","arguments":{"id":"doc3#2"}}</tool>',
             read_chunk),
    ]
