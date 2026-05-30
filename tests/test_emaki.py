"""No-network tests for KURA-Emaki: clustering, schema round-trip, edge registry, a full
zero-LLM build + serve (tree + co-occurrence graph + Cypher).
Run:  PYTHONPATH=. python3 tests/test_emaki.py
"""
import asyncio
import os
import tempfile

from ashigaru import Config
from ashigaru.emaki.schema import Node, format_skill_md, parse_skill_md
from ashigaru.emaki.cluster import build_tree
from ashigaru.emaki.edges import get_edge_registry
from ashigaru.emaki.build import run_build
from ashigaru.emaki.library import load_emaki
from ashigaru.emaki.graph import load_graph

TOPICS = {
    "ai":    "neural network deep learning gradient backprop tensor model training weights",
    "ocean": "ocean wave tide marine coral reef beach saltwater current lagoon",
    "music": "violin orchestra symphony melody concerto sonata cello tempo harmony",
}


def _chunks(per_topic=3):
    chunks = []
    for t, words in TOPICS.items():
        for i in range(per_topic):
            chunks.append({"id": f"{t}{i}", "source": f"{t}.md",
                           "text": f"{words} {words} ({t} document {i})"})
    return chunks


def test_schema_roundtrip():
    n = Node(node_id="root.2", level=1, parent_id="root", doc_ids=["a.md#0", "a.md#1"],
             name="Deep Learning", summary="Covers neural network training and gradients.",
             when_to_use="When asked about model training.", entities=["gradient", "tensor"],
             num_documents=2, confidence=0.9)
    parsed = parse_skill_md(format_skill_md(n))
    assert parsed["node_id"] == "root.2", parsed
    assert parsed["name"] == "Deep Learning", parsed
    assert parsed["level"] == 1 and parsed["num_documents"] == 2, parsed
    assert parsed["scope"] == "leaf", parsed
    assert "gradient" in parsed["entities"] and "tensor" in parsed["entities"], parsed
    assert parsed["description"], "description must be present (Agent-Skills validity)"
    print("✓ schema round-trip: SKILL.md <-> dict, name+description present")


def test_clustering_purity():
    chunks = _chunks(per_topic=3)
    nodes, root = build_tree(chunks, branching_p=3, leaf_max=2, max_depth=4)
    leaves = [n for n in nodes.values() if n.is_leaf]
    assert len(leaves) >= 3, f"expected >=3 leaves, got {len(leaves)}"
    impure = 0
    for lf in leaves:
        topics = {d.rstrip("0123456789") for d in lf.doc_ids}  # id prefix == topic
        if len(topics) > 1:
            impure += 1
    assert impure == 0, f"{impure} impure leaves (topics mixed): {[l.doc_ids for l in leaves]}"
    assert nodes[root].num_documents == len(chunks)
    print(f"✓ clustering: {len(leaves)} pure topical leaves from {len(chunks)} chunks")


def test_edge_registry():
    reg = get_edge_registry()
    assert reg.validate("CAUSES") and reg.validate("CO_OCCURS"), reg.names()
    assert reg.canonical("causes") == "CAUSES", "case-insensitive canonicalisation"
    assert reg.canonical("nonsense_rel") == "RELATED_TO", "unknown -> generic fallback"
    assert reg.inverse_of("CAUSES") == "CAUSED_BY", reg.inverse_of("CAUSES")
    assert reg.get("CONTRASTS").is_symmetric, "CONTRASTS is symmetric"
    assert len(reg.all_types()) >= 18, f"only {len(reg.all_types())} edge types loaded"
    print(f"✓ edge registry: {len(reg.all_types())} ontology relations, canonicalisation OK")


def test_build_and_serve():
    chunks = _chunks(per_topic=4)            # 12 chunks
    with tempfile.TemporaryDirectory() as tmp:
        corpus = os.path.join(tmp, "corpus")
        os.makedirs(corpus)
        for c in chunks:
            with open(os.path.join(corpus, c["id"] + ".md"), "w", encoding="utf-8") as f:
                f.write(c["text"])
        out = os.path.join(tmp, "scroll")
        cfg = Config()
        manifest = asyncio.run(run_build(
            corpus, out, cfg, chunk=512, overlap=64, branching=3, leaf_max=2, max_depth=4,
            backend="tfidf", embed_model=None, no_llm=True, graph_mode="cooccur"))

        # materialised files
        for fn in ("manifest.json", "tree.json", "documents.json", "SKILL.md", "INDEX.md",
                   "graph.json", "graph.cypher"):
            assert os.path.exists(os.path.join(out, fn)), f"missing {fn}"
        assert manifest["doc_count"] == 12, manifest
        assert manifest["leaf_count"] >= 3, manifest

        # serve: tree navigation
        lib = load_emaki(out)
        ov = lib.overview()
        assert "## Overview" in ov and "Index" in ov, ov[:200]
        root = lib.nodes[lib.root_id]
        assert root.children, "root should have branches"
        opened = lib.open(root.children[0])
        assert "node_id:" in opened, opened[:200]
        # fuzzy node open (slightly-off id still resolves)
        assert "node_id:" in lib.open(root.children[0].replace(".", "/")), "fuzzy id resolve"
        # read a leaf document
        some_doc = next(iter(lib.documents))
        doc = lib.get_document(some_doc)
        assert some_doc in doc and "neural" in doc or "ocean" in doc or "violin" in doc, doc[:120]
        # off-by-one / empty doc-id resolve safely (no crash, no arbitrary wrong file)
        stem = some_doc.split("#")[0]
        assert stem in lib.get_document(stem + "#999"), "off-by-one chunk -> same file"
        assert "No document" in lib.get_document(""), "empty doc id -> guidance, not arbitrary doc"

        # heuristic cards are non-empty even with no LLM
        assert all(lib.nodes[n].name for n in lib.nodes), "every node has a label"

        # serve: knowledge graph
        assert lib.has_graph
        kg = load_graph(out)
        assert kg.meta["entity_count"] > 0 and kg.meta["edge_count"] > 0, kg.meta
        ent_text = next(iter(kg.nodes.values()))["text"]
        nb = kg.neighbors(ent_text)
        assert "Relationships of" in nb or "no recorded" in nb, nb[:120]
        rd = kg.related_docs(ent_text)
        assert "Documents mentioning" in rd or "not linked" in rd, rd[:120]

        # Cypher sanity
        with open(os.path.join(out, "graph.cypher"), encoding="utf-8") as f:
            cy = f.read()
        assert "MERGE (:Entity {" in cy, "entity MERGE present"
        assert "MENTIONED_IN" in cy, "provenance edges present"
        stmts = [l for l in cy.splitlines() if l and not l.startswith("//")]
        assert stmts and all(l.rstrip().endswith(";") for l in stmts), "every statement ends with ;"
    print("✓ build+serve: zero-LLM scroll built, tree+graph navigable, Cypher valid")


def test_robustness():
    """Hardening from the adversarial review: JSON extraction + edge canonicalisation."""
    from ashigaru.emaki.distill import _extract_json_obj
    assert _extract_json_obj('```json\n{"a":1}\n```') == {"a": 1}, "fenced JSON"
    assert _extract_json_obj('{"a":1} then prose with {a brace}') == {"a": 1}, "trailing prose"
    assert _extract_json_obj('say {"a":{"b":2}} done') == {"a": {"b": 2}}, "nested + prose"
    assert _extract_json_obj("no json here") is None
    assert _extract_json_obj("[1,2,3]") is None, "top-level non-dict -> None"

    from ashigaru.emaki.edges import get_edge_registry
    reg = get_edge_registry()
    assert reg.canonical("part-of") == "PART_OF", "hyphen normalisation"
    assert reg.canonical("caused by") == "CAUSED_BY", "space normalisation"
    assert reg.canonical("transforms.into") == "TRANSFORMS_INTO", "dot normalisation"
    assert reg.canonical("???") == "RELATED_TO", "unknown -> generic"
    print("✓ robustness: JSON extraction (fences/prose/balanced), edge canonicalisation")


if __name__ == "__main__":
    test_schema_roundtrip()
    test_clustering_purity()
    test_edge_registry()
    test_robustness()
    test_build_and_serve()
    print("\nALL KURA-Emaki TESTS PASSED ✅")
