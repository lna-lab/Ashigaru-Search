"""Ontology-typed knowledge graph over the corpus — the second navigable artifact.

The topic tree lets a scout drill top-down; the graph lets it pivot SIDEWAYS along typed
relationships (cross-branch evidence). Built two ways:

  cooccur (default, zero-GPU): salient terms per chunk become TERM nodes; terms sharing a
      chunk get a weighted CO_OCCURS edge. No LLM, no embeddings.
  llm (--graph-llm): the fleet extracts typed entities + typed relationships per chunk using
      the adopted LNA-ES edge ontology (edges.py) — richer, at a build-time token cost.

Emits graph.json (served in-process) + graph.cypher (openCypher MERGE statements, loadable
into Neo4j directly or PostgreSQL + Apache AGE — the "書架" backend Ken proposed, which can
also shelf multimodal payloads alongside the graph). Provenance MENTIONED_IN edges tie every
entity back to the documents in the 蔵, mirroring AIOS's RAW_SOURCE pattern.
"""
from __future__ import annotations

import asyncio
import hashlib

from ..config import Config
from ..llm import LLMClient
from .distill import _chat_json
from .edges import get_edge_registry, NODE_TYPES
from .schema import Node

# bounds so a big corpus can't produce an unusable graph (drops are reported, never silent)
MAX_ENTITIES = 4000
MAX_EDGES = 12000
MAX_DOCLINKS_PER_ENTITY = 25


def _eid(norm: str) -> str:
    return "ent_" + hashlib.sha1(norm.encode("utf-8")).hexdigest()[:10]


def _cy(s) -> str:
    # collapse \r \n \t and other control whitespace to single spaces, then escape for a
    # double-quoted openCypher string literal (a literal CR/LF inside quotes is a syntax error)
    s = " ".join(str(s).split())
    return s.replace("\\", "\\\\").replace('"', '\\"')


# ---------------------------------------------------------------------------
# extraction — co-occurrence (default)
# ---------------------------------------------------------------------------
def _extract_cooccur(chunks: list[dict], top_terms: int = 8, min_len: int = 3):
    from .cluster import build_vectors, _STOP
    vecs = build_vectors(chunks, "tfidf")
    ents: dict[str, dict] = {}
    edges: dict[tuple, dict] = {}
    for ci, c in enumerate(chunks):
        ranked = sorted(vecs[ci].items(), key=lambda kv: kv[1], reverse=True)
        terms: list[str] = []
        for term, _w in ranked:
            if len(term) < min_len or term in _STOP or term.isdigit():
                continue
            terms.append(term)
            if len(terms) >= top_terms:
                break
        for t in terms:
            e = ents.setdefault(t, {"text": t, "type": "TERM", "ontology": "",
                                    "doc_ids": set(), "count": 0})
            e["doc_ids"].add(c["id"])
            e["count"] += 1
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                a, b = sorted((terms[i], terms[j]))
                ed = edges.setdefault((a, b, "CO_OCCURS"), {"count": 0})
                ed["count"] += 1
    return ents, edges


# ---------------------------------------------------------------------------
# extraction — LLM ontology typing (opt-in)
# ---------------------------------------------------------------------------
def _extract_system() -> str:
    reg = get_edge_registry()
    return (
        "You extract a knowledge graph from a document excerpt. Identify the salient entities "
        "and the typed relationships between them. Output ONLY JSON, no markdown fences:\n"
        '{"entities":[{"id":"e1","text":"...","type":"CONCEPT","ontology":""}],'
        '"relationships":[{"src":"e1","tgt":"e2","type":"CAUSES"}]}\n\n'
        f"Entity types: {', '.join(NODE_TYPES)}\n"
        f"{reg.format_vocabulary_for_prompt()}\n"
        "Keep entity text in the original language. Prefer specific named entities over glue "
        "words. Never invent entities not in the excerpt.")


def _symmetric(reg, typ: str) -> bool:
    et = reg.get(typ)
    return bool(et and et.is_symmetric)


async def _extract_llm(chunks: list[dict], cfg: Config, on_event=None):
    reg = get_edge_registry()
    system = _extract_system()
    llm = LLMClient(cfg.worker_base_url, cfg.worker_model, cfg.worker_api_key, cfg.request_timeout)
    sem = asyncio.Semaphore(cfg.max_concurrency)
    done = {"n": 0}

    async def _one(c: dict):
        async with sem:
            parsed = await _chat_json(llm, system, f"Document excerpt:\n{c['text'][:1500]}",
                                      temperature=cfg.temperature)
        done["n"] += 1
        if on_event:
            on_event("graph", {"msg": f"extracting {done['n']}/{len(chunks)}"})
        return parsed

    try:
        per_chunk = await asyncio.gather(*[_one(c) for c in chunks])
    finally:
        await llm.aclose()

    ents: dict[str, dict] = {}
    edges: dict[tuple, dict] = {}
    for c, parsed in zip(chunks, per_chunk):
        if not isinstance(parsed, dict):          # a flaky model must never abort the build
            continue
        try:
            local: dict[str, str] = {}
            for e in (parsed.get("entities") or []):
                if not isinstance(e, dict):       # tolerate bare-string entity lists
                    continue
                text = " ".join(str(e.get("text", "")).split()).strip()
                if not text or len(text) > 60:
                    continue
                norm = text.lower()
                eid = str(e.get("id", "")).strip() or norm   # empty/missing id -> key by text
                local.setdefault(eid, norm)                   # duplicate id -> first writer wins
                ent = ents.setdefault(norm, {"text": text,
                                             "type": str(e.get("type", "TERM")).upper(),
                                             "ontology": str(e.get("ontology", "")),
                                             "doc_ids": set(), "count": 0})
                ent["doc_ids"].add(c["id"])
                ent["count"] += 1
            rels = parsed.get("relationships")
            eds = parsed.get("edges")
            rels = rels if isinstance(rels, list) else []
            eds = eds if isinstance(eds, list) else []
            for r in rels + eds:
                if not isinstance(r, dict):
                    continue
                s = local.get(str(r.get("src", r.get("s", ""))))
                t = local.get(str(r.get("tgt", r.get("t", ""))))
                if not s or not t or s == t:
                    continue
                typ = reg.canonical(r.get("type"))
                if _symmetric(reg, typ):
                    s, t = sorted((s, t))
                else:
                    inv = reg.inverse_of(typ)     # collapse directed inverse pairs (CAUSES/CAUSED_BY)
                    if inv and inv != typ and inv < typ:
                        s, t, typ = t, s, inv      # flip endpoints + use one canonical orientation
                ed = edges.setdefault((s, t, typ), {"count": 0})
                ed["count"] += 1
        except Exception:                          # one malformed response can't kill the build
            continue
    return ents, edges


# ---------------------------------------------------------------------------
# assemble + prune + emit
# ---------------------------------------------------------------------------
def _assemble(ents: dict[str, dict], edges: dict[tuple, dict], n_chunks: int, mode: str) -> dict:
    reg = get_edge_registry()
    dropped_ents = dropped_edges = 0

    # prune hyper-frequent cooccur terms (stopword-like) and order by salience
    keep: dict[str, dict] = {}
    for norm, e in ents.items():
        df = len(e["doc_ids"])
        if mode == "cooccur" and n_chunks >= 8 and df > 0.5 * n_chunks:
            dropped_ents += 1
            continue
        keep[norm] = e
    ranked = sorted(keep.items(), key=lambda kv: kv[1]["count"], reverse=True)
    if len(ranked) > MAX_ENTITIES:
        dropped_ents += len(ranked) - MAX_ENTITIES
        ranked = ranked[:MAX_ENTITIES]
    surviving = {norm for norm, _ in ranked}

    id_of = {norm: _eid(norm) for norm in surviving}
    nodes = [{"id": id_of[norm], "text": e["text"], "type": e["type"],
              "ontology": e.get("ontology", ""), "doc_ids": sorted(e["doc_ids"]),
              "count": e["count"], "degree": 0} for norm, e in ranked]
    node_by_id = {n["id"]: n for n in nodes}

    edge_min = 2 if (mode == "cooccur" and n_chunks >= 20) else 1
    out_edges = []
    for (s, t, typ), ed in edges.items():
        if s not in surviving or t not in surviving or ed["count"] < edge_min:
            dropped_edges += 1
            continue
        out_edges.append({"src": id_of[s], "tgt": id_of[t], "type": typ,
                          "weight": round(reg.default_weight(typ), 3), "count": ed["count"],
                          "ai_hint": reg.ai_hint(typ)})
    out_edges.sort(key=lambda e: (e["count"], e["weight"]), reverse=True)
    if len(out_edges) > MAX_EDGES:
        dropped_edges += len(out_edges) - MAX_EDGES
        out_edges = out_edges[:MAX_EDGES]

    for e in out_edges:
        node_by_id[e["src"]]["degree"] += 1
        node_by_id[e["tgt"]]["degree"] += 1

    return {
        "meta": {"mode": mode, "entity_count": len(nodes), "edge_count": len(out_edges),
                 "dropped_entities": dropped_ents, "dropped_edges": dropped_edges,
                 "lineage": "LNA-ES v4.0 edge ontology (AIOS)", "edge_min_count": edge_min},
        "nodes": nodes, "edges": out_edges,
    }


def _write_cypher(out_dir: str, g: dict, doc_source: dict[str, str], mode: str) -> None:
    import os
    label = ("co-occurrence graph (single CO_OCCURS relation, TERM nodes)" if mode == "cooccur"
             else "ontology-typed graph (LNA-ES v4.0 edge vocabulary)")
    lines = [
        f"// KURA-Emaki knowledge graph — {label} (clean-room)",
        f"// {g['meta']['entity_count']} entities, {g['meta']['edge_count']} edges, mode={mode}",
        f"// note: MENTIONED_IN provenance is capped at {MAX_DOCLINKS_PER_ENTITY} docs/entity here; "
        "graph.json keeps the full doc_ids per entity.",
        "// Neo4j:   cat graph.cypher | cypher-shell",
        "// Apache AGE (PostgreSQL):  wrap each statement as",
        "//   SELECT * FROM cypher('emaki', $$ <statement-without-trailing-semicolon> $$) as (v agtype);",
        "",
    ]
    for n in g["nodes"]:
        lines.append(
            f'MERGE (:Entity {{id:"{_cy(n["id"])}", text:"{_cy(n["text"])}", '
            f'type:"{_cy(n["type"])}", ontology:"{_cy(n["ontology"])}", count:{n["count"]}}});')
    seen_docs: set[str] = set()
    for n in g["nodes"]:
        for did in n["doc_ids"][:MAX_DOCLINKS_PER_ENTITY]:
            if did not in seen_docs:
                seen_docs.add(did)
                lines.append(f'MERGE (:Document {{doc_id:"{_cy(did)}", '
                             f'source:"{_cy(doc_source.get(did, ""))}"}});')
    for n in g["nodes"]:
        for did in n["doc_ids"][:MAX_DOCLINKS_PER_ENTITY]:
            lines.append(
                f'MATCH (e:Entity {{id:"{_cy(n["id"])}"}}),(d:Document {{doc_id:"{_cy(did)}"}}) '
                f'MERGE (e)-[:MENTIONED_IN]->(d);')
    for e in g["edges"]:
        lines.append(
            f'MATCH (a:Entity {{id:"{_cy(e["src"])}"}}),(b:Entity {{id:"{_cy(e["tgt"])}"}}) '
            f'MERGE (a)-[:{e["type"]} {{weight:{e["weight"]}, count:{e["count"]}, '
            f'ai_hint:"{_cy(e["ai_hint"])}"}}]->(b);')
    with open(os.path.join(out_dir, "graph.cypher"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


async def build_graph(chunks: list[dict], nodes: dict[str, Node], out_dir: str, *,
                      mode: str = "cooccur", cfg: Config | None = None, on_event=None) -> dict:
    import json
    import os
    cfg = cfg or Config()
    if mode == "llm":
        ents, edges = await _extract_llm(chunks, cfg, on_event)
    else:
        ents, edges = _extract_cooccur(chunks)
    g = _assemble(ents, edges, len(chunks), mode)
    with open(os.path.join(out_dir, "graph.json"), "w", encoding="utf-8") as f:
        json.dump(g, f, ensure_ascii=False, indent=1)
    _write_cypher(out_dir, g, {c["id"]: c["source"] for c in chunks}, mode)
    if on_event:
        on_event("graph_done", {"nodes": g["meta"]["entity_count"],
                                "edges": g["meta"]["edge_count"], "mode": mode,
                                "dropped_entities": g["meta"]["dropped_entities"],
                                "dropped_edges": g["meta"]["dropped_edges"]})
    return g["meta"]


# ---------------------------------------------------------------------------
# serve-time loader
# ---------------------------------------------------------------------------
class KnowledgeGraph:
    def __init__(self, path: str):
        import json
        import os
        with open(os.path.join(path, "graph.json"), "r", encoding="utf-8") as f:
            g = json.load(f)
        self.meta = g.get("meta", {})
        self.nodes = {n["id"]: n for n in g.get("nodes", [])}
        self._by_text: dict[str, str] = {}
        for n in g.get("nodes", []):
            self._by_text.setdefault(n["text"].lower(), n["id"])
        self.adj: dict[str, list[tuple]] = {}
        for e in g.get("edges", []):
            self.adj.setdefault(e["src"], []).append((e["tgt"], e["type"], e["weight"], "->"))
            self.adj.setdefault(e["tgt"], []).append((e["src"], e["type"], e["weight"], "<-"))

    def _resolve(self, q: str) -> str | None:
        q = (q or "").strip().lower()
        if not q:
            return None
        if q in self._by_text:
            return self._by_text[q]
        # Prefer the LONGEST entity name that appears in the query (or contains it) — so a whole
        # question like "「無常」という概念は何と繋がっているか" resolves to its most specific
        # mentioned entity ("無常"), not the first short term that happens to overlap. This lets
        # the harness seed graph_neighbors(<the whole question>) and hand over the right subgraph.
        best: str | None = None
        best_len = 0
        for text, nid in self._by_text.items():
            if text and (text in q or q in text) and len(text) > best_len:
                best, best_len = nid, len(text)
        return best

    def neighbors(self, entity: str, hops: int = 1, limit: int = 25) -> str:
        nid = self._resolve(entity)
        if nid is None:
            return f'No entity matching "{entity}" in the graph.'
        seen = {nid}
        frontier = [(nid, 0)]
        rows: list[str] = []
        while frontier and len(rows) < limit:
            cur, depth = frontier.pop(0)
            if depth >= hops:
                continue
            for tgt, typ, weight, direction in self.adj.get(cur, []):
                if tgt in seen:
                    continue
                seen.add(tgt)
                node = self.nodes.get(tgt, {})
                arrow = f"-[{typ}]{direction}" if direction == "->" else f"<-[{typ}]-"
                rows.append(f"- {node.get('text', tgt)} [{node.get('type','')}] "
                            f"{arrow} (w={weight}, hop {depth+1})")
                frontier.append((tgt, depth + 1))
                if len(rows) >= limit:
                    break
        head = self.nodes[nid]
        if not rows:
            return f'"{head["text"]}" has no recorded relationships in the graph.'
        return (f'Relationships of "{head["text"]}" [{head.get("type","")}]:\n'
                + "\n".join(rows))

    def related_docs(self, entity: str, limit: int = 15) -> str:
        nid = self._resolve(entity)
        if nid is None:
            return f'No entity matching "{entity}" in the graph.'
        node = self.nodes[nid]
        docs = node.get("doc_ids", [])[:limit]
        if not docs:
            return f'"{node["text"]}" is not linked to any document.'
        lines = [f'Documents mentioning "{node["text"]}" (read one with get_document(doc_id)):']
        lines += [f"- {d}" for d in docs]
        return "\n".join(lines)


def load_graph(path: str) -> KnowledgeGraph:
    return KnowledgeGraph(path)
