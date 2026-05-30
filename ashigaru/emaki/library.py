"""Serve-time loader for a built scroll. The scout walks it with three verbs:
  overview()            -> the root bird's-eye card + its branch index
  open(node_id)         -> that node's card + its children (or, at a leaf, its documents)
  get_document(doc_id)  -> a leaf document in full

Node-id and doc-id lookups are fuzzy (prefix / name match) so a small model's slightly-off
reference still resolves instead of dead-ending.
"""
from __future__ import annotations

import json
import os

from .schema import Node, format_skill_md, format_index_md


class EmakiLibrary:
    def __init__(self, path: str):
        self.path = path
        self.manifest: dict = {}
        self.nodes: dict[str, Node] = {}
        self.root_id: str = "root"
        self.documents: dict[str, dict] = {}
        self._card_cache: dict[str, str] = {}
        self._index_cache: dict[str, str] = {}
        self._load()

    # ---- loading ----
    def _load(self) -> None:
        self.manifest = self._read_json("manifest.json", {})
        self.root_id = self.manifest.get("root_node_id", "root")
        for d in self._read_json("tree.json", []):
            n = Node.from_dict(d)
            self.nodes[n.node_id] = n
        self.documents = self._read_json("documents.json", {})

    def _read_json(self, name: str, default):
        fp = os.path.join(self.path, name)
        if not os.path.exists(fp):
            return default
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f)

    def _doc_meta(self, node: Node) -> dict[str, tuple[str, str]]:
        meta = {}
        for did in node.doc_ids:
            doc = self.documents.get(did, {})
            snip = " ".join((doc.get("text") or "").split())[:80]
            meta[did] = (doc.get("source", ""), snip)
        return meta

    # ---- card / index rendering (prefer the materialised .md, else format on the fly) ----
    def _card(self, node_id: str) -> str:
        if node_id in self._card_cache:
            return self._card_cache[node_id]
        fp = os.path.join(self.path, "nodes", node_id, "SKILL.md")
        if os.path.exists(fp):
            with open(fp, "r", encoding="utf-8") as f:
                txt = f.read()
        else:
            txt = format_skill_md(self.nodes[node_id])
        self._card_cache[node_id] = txt
        return txt

    def _index(self, node_id: str) -> str:
        if node_id in self._index_cache:
            return self._index_cache[node_id]
        fp = os.path.join(self.path, "nodes", node_id, "INDEX.md")
        if os.path.exists(fp):
            with open(fp, "r", encoding="utf-8") as f:
                txt = f.read()
        else:
            node = self.nodes[node_id]
            txt = format_index_md(node, self.nodes, self._doc_meta(node))
        self._index_cache[node_id] = txt
        return txt

    # ---- fuzzy resolution ----
    def _resolve_node(self, q: str) -> str | None:
        q = (q or "").strip()
        if q in self.nodes:
            return q
        norm = q.replace("/", ".").replace("-", ".").replace(" ", "")
        if norm in self.nodes:
            return norm
        pref = [nid for nid in self.nodes if nid.startswith(q) or nid.startswith(norm)]
        if len(pref) == 1:
            return pref[0]
        for nid, n in self.nodes.items():
            if n.name and n.name.lower() == q.lower():
                return nid
        return pref[0] if pref else None

    def _resolve_doc(self, q: str) -> str | None:
        q = (q or "").strip()
        if not q:
            return None
        if q in self.documents:
            return q
        # nearest existing chunk of the requested file: an off-by-one "file#5" resolves to the
        # closest real chunk of THAT file (largest n <= requested, else #0) — not an arbitrary doc
        stem = q.split("#")[0]
        same = sorted(d for d in self.documents if d.split("#")[0] == stem)
        if same:
            if "#" in q:
                try:
                    want = int(q.split("#", 1)[1])
                    le = [d for d in same if int(d.split("#", 1)[1]) <= want]
                    return le[-1] if le else same[0]
                except (ValueError, IndexError):
                    return same[0]
            return same[0]
        cands = [d for d in self.documents if q in d]
        return cands[0] if cands else None

    # ---- the three serve verbs ----
    def overview(self) -> str:
        rid = self.root_id
        if rid not in self.nodes:
            return ("This KURA-Emaki scroll is empty or incomplete (no tree.json / root node). "
                    "Rebuild it with `ashigaru-emaki <corpus_dir> <out_dir>`.")
        return self._card(rid) + "\n\n" + self._index(rid)

    def open(self, node_id: str) -> str:
        nid = self._resolve_node(node_id)
        if nid is None:
            top = self.nodes.get(self.root_id)
            kids = ", ".join(top.children[:12]) if top else ""
            return (f"No branch '{node_id}'. Call tree_overview first, then open a listed "
                    f"node_id. Root branches: {kids}")
        return self._card(nid) + "\n\n" + self._index(nid)

    def get_document(self, doc_id: str, char_limit: int = 8000) -> str:
        did = self._resolve_doc(doc_id)
        if did is None:
            return (f"No document '{doc_id}'. Use the doc_ids listed in a leaf's INDEX.md "
                    f"(open a leaf branch with tree_open first).")
        doc = self.documents[did]
        text = doc.get("text", "")
        src = doc.get("source", "")
        clipped = text[:char_limit]
        more = "" if len(text) <= char_limit else f"\n…[truncated {len(text) - char_limit} chars]"
        return f"[{did}] ({src})\n{clipped}{more}"

    @property
    def has_graph(self) -> bool:
        return os.path.exists(os.path.join(self.path, "graph.json"))


def load_emaki(path: str) -> EmakiLibrary:
    return EmakiLibrary(path)
