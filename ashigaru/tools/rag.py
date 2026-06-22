"""Local document RAG tools: doc_search + read_chunk.

doc_search addresses the corpus by MEANING when an embeddings endpoint is configured
(``cfg.embed_url`` + chunks carrying an ``embedding``) — cosine over the query embedding, which
fixes the keyword-collision whiffs BM25 suffers (e.g. "無常" matching "記録"). Otherwise it falls
back to BM25 (zero-GPU default, and the fallback whenever the embed endpoint is unreachable).
Build a corpus from a folder with `ashigaru-index <folder> <out.pkl>` (see ashigaru.rag_index).
"""
from __future__ import annotations
import math
import pickle
from collections.abc import Mapping

from ..config import Config
from ..registry import Tool
from ..tok import tokenize as _tok   # CJK-aware tokenizer, shared with emaki (index==query)


async def _embed_query(cfg: Config, text: str) -> list[float]:
    """Embed one query string via an OpenAI-compatible /v1/embeddings endpoint."""
    import httpx

    url = cfg.embed_url.rstrip("/") + "/v1/embeddings"
    payload = {"model": cfg.embed_model or "embed", "input": [text]}
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def make_rag_tools(cfg: Config) -> list[Tool]:
    # Fail closed on a corrupt / wrong-shape corpus pickle rather than indexing garbage or
    # crashing the whole toolbox build. The contract is {"chunks": [{id, source, text}], ...}.
    with open(cfg.rag_index, "rb") as f:
        blob = pickle.load(f)
    if not isinstance(blob, Mapping) or not isinstance(blob.get("chunks"), list):
        raise ValueError(
            f"corpus pickle {cfg.rag_index!r} is not a {{'chunks': [...]}} mapping")
    # keep only well-formed chunks (a Mapping with an id and non-empty text); carry any embedding
    chunks: list[dict] = []
    for c in blob["chunks"]:
        if not isinstance(c, Mapping):
            continue
        if not c.get("id") or not (c.get("text") or "").strip():
            continue
        emb = c.get("embedding")
        chunks.append({"id": str(c["id"]), "source": c.get("source") or "",
                       "text": str(c["text"]),
                       "embedding": list(emb) if isinstance(emb, (list, tuple)) else None})

    # BM25Okapi divides by the corpus size, so it cannot be built over an empty corpus —
    # leave bm25 None and short-circuit doc_search (an empty index is valid: returns no hits).
    bm25 = None
    if chunks:
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi([_tok(c["text"]) or [""] for c in chunks])

    # Semantic addressing is available when an endpoint is configured AND the corpus is embedded.
    semantic_ready = bool(cfg.embed_url) and any(c["embedding"] for c in chunks)

    def _format(query: str, ranked: list[tuple[float, int]], mode: str) -> str:
        out = [f"Local document hits for: {query} ({mode})"]
        for score, i in ranked:
            c = chunks[i]
            snip = " ".join(c["text"].split())[:240]
            out.append(f"[{c['id']}] ({c['source']})  score={score:.2f}\n    {snip}")
        return "\n".join(out) if len(out) > 1 else f"No local matches for: {query}"

    async def doc_search(args: dict) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            return "ERROR: doc_search needs a 'query'."
        if bm25 is None:
            return f"No local corpus indexed (0 chunks). No matches for: {query}"
        k = int(args.get("k") or 5)

        if semantic_ready:
            try:  # address by MEANING; fall back to BM25 if the endpoint is unreachable
                qv = await _embed_query(cfg, (cfg.embed_query_instruct or "") + query)
                ranked = sorted(((_cosine(qv, c["embedding"]), i)
                                 for i, c in enumerate(chunks) if c["embedding"]),
                                reverse=True)[:k]
                ranked = [(s, i) for s, i in ranked if s > 0.0]
                if ranked:
                    return _format(query, ranked, "semantic")
            except Exception:
                pass  # fall through to BM25

        scores = bm25.get_scores(_tok(query))
        ranked = sorted(((float(scores[i]), i) for i in range(len(chunks)) if scores[i] > 0),
                        reverse=True)[:k]
        return _format(query, ranked, "BM25")

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
