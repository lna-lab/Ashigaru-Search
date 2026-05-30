"""Build a KURA-Emaki scroll from a folder of documents.

    ashigaru-emaki <corpus_dir> <out_dir> [--branching 8] [--leaf-max 10] [--graph] ...

Pipeline: ingest (reuse the BM25 indexer's chunking) -> cluster (zero-GPU TF-IDF by
default) -> distil (fleet summarises each cluster) -> materialise SKILL.md/INDEX.md cards +
documents.json + entity_index.json (+ optional ontology knowledge graph). Point the serve
side at it with  ASHIGARU_EMAKI=<out_dir>  and the scouts will navigate instead of retrieve.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

import httpx

from ..config import Config
from ..llm import LLMClient
from ..rag_index import _read, _chunk, TEXT_EXT  # reuse the exact ingest/chunking
from .cluster import build_tree, subtree_doc_ids
from .distill import distill_tree
from .schema import Node, format_skill_md, format_index_md


def ingest(folder: str, chunk: int = 512, overlap: int = 64) -> list[dict]:
    """Walk a folder into {id, source, text} chunk records (same ids as ashigaru-index)."""
    chunks: list[dict] = []
    for root, _, files in os.walk(folder):
        for fn in sorted(files):
            path = os.path.join(root, fn)
            text = _read(path)
            if not text.strip():
                continue
            stem = os.path.relpath(path, folder)
            for n, piece in enumerate(_chunk(text.split(), chunk, overlap)):
                chunks.append({"id": f"{stem}#{n}", "source": stem, "text": piece})
    return chunks


def _materialize(out_dir: str, nodes: dict[str, Node], root_id: str, chunks: list[dict],
                 manifest: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    documents = {c["id"]: {"source": c["source"], "text": c["text"]} for c in chunks}

    def _w(name: str, obj):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)

    _w("documents.json", documents)
    _w("tree.json", [n.to_dict() for n in nodes.values()])

    # entity cross-index: entity -> all subtree doc ids of the node(s) that surfaced it
    ent: dict[str, set] = {}
    for nid, node in nodes.items():
        if not node.entities:
            continue
        docs = subtree_doc_ids(nodes, nid)
        for e in node.entities:
            ent.setdefault(e.lower(), set()).update(docs)
    _w("entity_index.json", {k: sorted(v) for k, v in ent.items()})

    # per-node cards
    for nid, node in nodes.items():
        nd = os.path.join(out_dir, "nodes", nid)
        os.makedirs(nd, exist_ok=True)
        doc_meta = {did: (documents[did]["source"],
                          " ".join(documents[did]["text"].split())[:80])
                    for did in node.doc_ids}
        with open(os.path.join(nd, "SKILL.md"), "w", encoding="utf-8") as f:
            f.write(format_skill_md(node))
        with open(os.path.join(nd, "INDEX.md"), "w", encoding="utf-8") as f:
            f.write(format_index_md(node, nodes, doc_meta))

    # top-level root copies -> the whole dir validates as a portable Anthropic Agent Skill
    root = nodes[root_id]
    with open(os.path.join(out_dir, "SKILL.md"), "w", encoding="utf-8") as f:
        f.write(format_skill_md(root))
    with open(os.path.join(out_dir, "INDEX.md"), "w", encoding="utf-8") as f:
        f.write(format_index_md(root, nodes,
                {d: (documents[d]["source"], " ".join(documents[d]["text"].split())[:80])
                 for d in root.doc_ids}))
    _w("manifest.json", manifest)


async def run_build(corpus: str, out: str, cfg: Config, *, chunk: int, overlap: int,
                    branching: int, leaf_max: int, max_depth: int, backend: str,
                    embed_model: str | None, no_llm: bool, graph_mode: str | None,
                    on_event=None) -> dict:
    if not os.path.isdir(corpus):
        raise SystemExit(f"Corpus path not found (or not a directory): {corpus} — check the path.")
    chunks = ingest(corpus, chunk, overlap)
    if not chunks:
        raise SystemExit(f"No readable documents under {corpus} "
                         f"(supported: {', '.join(sorted(TEXT_EXT))} + .pdf).")
    if on_event:
        on_event("ingest", {"chunks": len(chunks)})

    nodes, root_id = build_tree(chunks, branching_p=branching, leaf_max=leaf_max,
                                max_depth=max_depth, backend=backend, embed_model=embed_model)
    leaves = sum(1 for n in nodes.values() if n.is_leaf)
    if on_event:
        on_event("cluster", {"nodes": len(nodes), "leaves": leaves})

    chunk_text = {c["id"]: c["text"] for c in chunks}
    llm = None if no_llm else LLMClient(cfg.worker_base_url, cfg.worker_model,
                                        cfg.worker_api_key, cfg.request_timeout)
    try:
        await distill_tree(llm, nodes, root_id, chunk_text, cfg, on_event=on_event)
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.PoolTimeout) as e:
        print(f"  [warn] worker LLM unreachable at {cfg.worker_base_url} ({type(e).__name__}) — "
              f"building heuristic cards offline. Pass --no-llm to silence, or start your vLLM "
              f"server for distilled cards.", file=sys.stderr)
        await distill_tree(None, nodes, root_id, chunk_text, cfg, on_event=on_event)
    finally:
        if llm is not None:
            await llm.aclose()

    manifest = {
        "root_node_id": root_id, "node_count": len(nodes), "leaf_count": leaves,
        "doc_count": len(chunks), "branching_p": branching, "leaf_max": leaf_max,
        "max_depth": max_depth, "backend": backend, "embed_model": embed_model,
        "distilled": not no_llm, "has_graph": bool(graph_mode), "graph_mode": graph_mode,
        "generator": "ashigaru-emaki (KURA-Emaki)", "format_version": 1,
    }
    _materialize(out, nodes, root_id, chunks, manifest)

    if graph_mode:
        from .graph import build_graph                      # lazy: graph is opt-in
        try:
            g = await build_graph(chunks, nodes, out, mode=graph_mode, cfg=cfg, on_event=on_event)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.PoolTimeout) as e:
            if graph_mode != "llm":
                raise
            print(f"  [warn] worker LLM unreachable for --graph-llm ({type(e).__name__}) — "
                  f"falling back to the zero-infra co-occurrence graph.", file=sys.stderr)
            graph_mode = "cooccur"
            g = await build_graph(chunks, nodes, out, mode="cooccur", cfg=cfg, on_event=on_event)
        manifest["graph"], manifest["graph_mode"] = g, graph_mode
        with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=1)
    return manifest


def _reporter(quiet: bool):
    D, G, Y, C, R = "\033[2m", "\033[32m", "\033[33m", "\033[36m", "\033[0m"
    state = {"d": 0}

    def report(stage: str, info: dict):
        if quiet:
            return
        if stage == "ingest":
            print(f"{C}ingested {info['chunks']} chunks{R}", file=sys.stderr)
        elif stage == "cluster":
            print(f"{C}clustered into {info['nodes']} nodes ({info['leaves']} leaves){R}",
                  file=sys.stderr)
        elif stage == "distill":
            state["d"] = info["done"]
            end = "\n" if info["done"] == info["total"] else "\r"
            print(f"{D}distilling {info['done']}/{info['total']}  {info['name'][:40]:<40}{R}",
                  end=end, file=sys.stderr)
        elif stage == "graph":
            print(f"{Y}graph: {info.get('msg','')}{R}", file=sys.stderr)
        elif stage == "graph_done":
            drops = ""
            if info.get("dropped_entities") or info.get("dropped_edges"):
                drops = (f" (dropped {info.get('dropped_entities',0)} entities / "
                         f"{info.get('dropped_edges',0)} edges to bounds)")
            print(f"{G}graph: {info['nodes']} entities, {info['edges']} edges "
                  f"({info['mode']}){drops} -> graph.cypher{R}", file=sys.stderr)
    return report


def main():
    ap = argparse.ArgumentParser(
        prog="ashigaru-emaki",
        description="Build a KURA-Emaki (蔵絵巻) navigable knowledge scroll from a corpus.")
    ap.add_argument("corpus", help="folder of documents (.txt/.md/.rst/.pdf …)")
    ap.add_argument("out", help="output scroll directory (point ASHIGARU_EMAKI at it)")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--branching", type=int, default=8, help="topic-tree branching factor p")
    ap.add_argument("--leaf-max", type=int, default=10, help="max chunks in a leaf cluster")
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--embed", nargs="?", const="all-MiniLM-L6-v2", default=None,
                    metavar="MODEL", help="opt-in embedding backend (default: zero-GPU TF-IDF)")
    ap.add_argument("--no-llm", action="store_true",
                    help="skip LLM distillation (heuristic cards only)")
    ap.add_argument("--graph", action="store_true",
                    help="also build a co-occurrence ontology knowledge graph (zero-GPU)")
    ap.add_argument("--graph-llm", action="store_true",
                    help="build a typed ontology graph via LLM extraction (build-time tokens)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    cfg = Config()
    backend = "embed" if a.embed else "tfidf"
    graph_mode = "llm" if a.graph_llm else ("cooccur" if a.graph else None)
    manifest = asyncio.run(run_build(
        a.corpus, a.out, cfg, chunk=a.chunk, overlap=a.overlap, branching=a.branching,
        leaf_max=a.leaf_max, max_depth=a.max_depth, backend=backend, embed_model=a.embed,
        no_llm=a.no_llm, graph_mode=graph_mode, on_event=_reporter(a.quiet)))

    print(f"\n蔵絵巻 built: {manifest['node_count']} nodes / {manifest['leaf_count']} leaves / "
          f"{manifest['doc_count']} chunks -> {a.out}")
    if manifest.get("has_graph"):
        g = manifest.get("graph", {})
        drops = ""
        if g.get("dropped_entities") or g.get("dropped_edges"):
            drops = (f"  (dropped {g.get('dropped_entities',0)} entities / "
                     f"{g.get('dropped_edges',0)} edges to bounds)")
        print(f"  knowledge graph: {g.get('entity_count','?')} entities, "
              f"{g.get('edge_count','?')} edges -> {a.out}/graph.cypher{drops}")
    print(f"  serve:  ASHIGARU_EMAKI={a.out}  ashigaru \"your question\"")


if __name__ == "__main__":
    main()
