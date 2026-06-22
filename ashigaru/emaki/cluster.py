"""Recursive topic clustering — the spine of the scroll.

Default backend is ZERO-GPU and dependency-free: a hand-rolled TF-IDF + spherical k-means,
recursed into a branching topic tree. Deterministic (farthest-first seeding, no RNG) so a
build is reproducible. Tokenization is CJK-aware (shared `ashigaru.tok`): space-less Japanese
splits into single-character tokens so the topic tree can actually cluster Japanese corpora;
an optional `embed` backend (sentence-transformers) gives finer topical cohesion but is opt-in
to preserve Ashigaru's zero-GPU ethos.

This is independently authored clustering math (clean-room): the upstream Corpus2Skill repo
is all-rights-reserved and none of its code is used. The recursive-cluster-then-summarise
shape is the publicly-described idea (RAPTOR 2401.18059 / Corpus2Skill 2604.14572).
"""
from __future__ import annotations

import math

from ..tok import is_cjk_char
from ..tok import tokenize as _base_tokenize
from .schema import Node

# a light, language-agnostic-ish stopword set — keeps the tree from clustering on glue words
# (the AIOS reading-graph cypher showed "You/Access/The/For" noise nodes; we filter up front)
_STOP = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with", "as", "by",
    "at", "from", "is", "are", "was", "were", "be", "been", "being", "it", "its", "this",
    "that", "these", "those", "i", "you", "he", "she", "they", "we", "them", "his", "her",
    "their", "our", "your", "my", "me", "us", "do", "does", "did", "have", "has", "had",
    "not", "no", "yes", "if", "then", "else", "so", "than", "too", "very", "can", "could",
    "will", "would", "should", "may", "might", "must", "what", "which", "who", "whom",
    "how", "when", "where", "why", "all", "any", "some", "more", "most", "other", "such",
    "only", "own", "same", "about", "into", "over", "out", "up", "down", "off", "again",
}


def tokenize(text: str) -> list[str]:
    # CJK-aware base tokens, then drop glue: single-char CJK ideographs/kana are MEANINGFUL
    # (the unit for space-less languages) so they're kept; short ASCII runs / stopwords / pure
    # digits are dropped as noise — same intent as before, now correct for Japanese.
    return [w for w in _base_tokenize(text)
            if is_cjk_char(w) or (len(w) > 1 and w not in _STOP and not w.isdigit())]


# ---------------------------------------------------------------------------
# Sparse vector helpers (vectors are dict[term -> weight], L2-normalised)
# ---------------------------------------------------------------------------
def _dot(a: dict, b: dict) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(w * b.get(k, 0.0) for k, w in a.items())


def _normalize(v: dict) -> dict:
    n = math.sqrt(sum(w * w for w in v.values()))
    if n <= 0:
        return v
    return {k: w / n for k, w in v.items()}


def _centroid(vecs: list[dict], idxs: list[int]) -> dict:
    acc: dict = {}
    for i in idxs:
        for k, w in vecs[i].items():
            acc[k] = acc.get(k, 0.0) + w
    if idxs:
        inv = 1.0 / len(idxs)
        acc = {k: w * inv for k, w in acc.items()}
    return _normalize(acc)


def build_vectors(chunks: list[dict], backend: str = "tfidf",
                  embed_model: str | None = None) -> list[dict]:
    """Return one L2-normalised sparse vector per chunk."""
    if backend == "embed":
        return _embed_vectors([c["text"] for c in chunks], embed_model or "all-MiniLM-L6-v2")
    # --- TF-IDF (default, zero-GPU) ---
    toks = [tokenize(c["text"]) for c in chunks]
    n = len(toks)
    df: dict[str, int] = {}
    for tl in toks:
        for w in set(tl):
            df[w] = df.get(w, 0) + 1
    vecs: list[dict] = []
    for tl in toks:
        tf: dict[str, int] = {}
        for w in tl:
            tf[w] = tf.get(w, 0) + 1
        v = {w: (1.0 + math.log(c)) * (math.log((n + 1) / (df[w] + 1)) + 1.0)
             for w, c in tf.items()}
        vecs.append(_normalize(v))
    return vecs


def _embed_vectors(texts: list[str], model_name: str) -> list[dict]:
    from sentence_transformers import SentenceTransformer  # optional dep (extras: emaki)
    model = SentenceTransformer(model_name)
    arr = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                       show_progress_bar=False)
    return [{i: float(x) for i, x in enumerate(row) if abs(float(x)) > 1e-6} for row in arr]


# ---------------------------------------------------------------------------
# Deterministic spherical k-means
# ---------------------------------------------------------------------------
def _farthest_first(vecs: list[dict], idxs: list[int], k: int) -> list[dict]:
    """Pick k spread-out seeds deterministically (no RNG): first index, then repeatedly the
    point with the smallest max-similarity to the already-chosen seeds."""
    seeds = [dict(vecs[idxs[0]])]
    while len(seeds) < k:
        best_i, best_score = None, 2.0
        for i in idxs:
            sim = max(_dot(vecs[i], s) for s in seeds)   # similarity to nearest seed
            if sim < best_score:                          # want the most dissimilar point
                best_score, best_i = sim, i
        if best_i is None:
            break
        seeds.append(dict(vecs[best_i]))
    return seeds


def _kmeans(vecs: list[dict], idxs: list[int], k: int, iters: int = 12) -> list[list[int]]:
    """Spherical k-means over a subset of points. Returns non-empty clusters (lists of idx)."""
    k = max(2, min(k, len(idxs)))
    centroids = _farthest_first(vecs, idxs, k)
    assign: dict[int, int] = {}
    for _ in range(iters):
        groups: list[list[int]] = [[] for _ in range(len(centroids))]
        changed = False
        for i in idxs:
            sims = [(_dot(vecs[i], c), ci) for ci, c in enumerate(centroids)]
            best = max(sims, key=lambda t: (t[0], -t[1]))[1]   # tie -> lowest centroid index
            groups[best].append(i)
            if assign.get(i) != best:
                changed = True
                assign[i] = best
        groups = [g for g in groups if g]
        if not changed or len(groups) <= 1:
            return groups
        centroids = [_centroid(vecs, g) for g in groups]
    return [g for g in groups if g]


# ---------------------------------------------------------------------------
# Recursive tree build
# ---------------------------------------------------------------------------
def build_tree(chunks: list[dict], *, branching_p: int = 8, leaf_max: int = 10,
               max_depth: int = 6, backend: str = "tfidf",
               embed_model: str | None = None) -> tuple[dict[str, Node], str]:
    """Cluster `chunks` into a topic tree. Returns (node_map, root_id).

    A node is a leaf when it holds <= leaf_max chunks (or max_depth is hit, or a split fails
    to separate the points). Leaves carry doc_ids (chunk ids); internal nodes carry children.
    """
    vecs = build_vectors(chunks, backend=backend, embed_model=embed_model)
    chunk_ids = [c["id"] for c in chunks]
    nodes: dict[str, Node] = {}

    def _build(idxs: list[int], node_id: str, level: int, parent: str | None):
        node = Node(node_id=node_id, level=level, parent_id=parent)
        nodes[node_id] = node
        if len(idxs) <= leaf_max or level >= max_depth:
            node.doc_ids = [chunk_ids[i] for i in idxs]
            return
        parts = _kmeans(vecs, idxs, min(branching_p, len(idxs)))
        if len(parts) <= 1 or any(len(p) == len(idxs) for p in parts):
            node.doc_ids = [chunk_ids[i] for i in idxs]   # split made no progress -> leaf
            return
        for i, part in enumerate(parts):
            cid = f"{node_id}.{i}"
            node.children.append(cid)
            _build(part, cid, level + 1, node_id)

    _build(list(range(len(chunks))), "root", 0, None)
    finalize_counts(nodes, "root")
    return nodes, "root"


def finalize_counts(nodes: dict[str, Node], root_id: str) -> None:
    """Post-order: set num_documents = subtree leaf-doc count."""
    def _count(nid: str) -> int:
        node = nodes[nid]
        if node.is_leaf:
            node.num_documents = len(node.doc_ids)
        else:
            node.num_documents = sum(_count(c) for c in node.children)
        return node.num_documents
    _count(root_id)


def subtree_doc_ids(nodes: dict[str, Node], node_id: str) -> list[str]:
    """All leaf doc_ids beneath a node (used for the entity cross-index)."""
    node = nodes[node_id]
    if node.is_leaf:
        return list(node.doc_ids)
    out: list[str] = []
    for c in node.children:
        out.extend(subtree_doc_ids(nodes, c))
    return out
